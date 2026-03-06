import os
import json
import re
import pandas as pd
from connections import get_vanna_instance, connect_database, get_columns_query, _engine

class AgenticSQLPipeline:
    def __init__(self):
        self.vn = get_vanna_instance()
        connect_database(self.vn)
    
    def get_table_columns(self, table_name):
        """Fetch actual column metadata from database to prevent hallucination."""
        try:
            from connections import _engine
            if _engine is None:
                return []
            query = get_columns_query(table_name)
            result = pd.read_sql(query, _engine)
            return result.to_dict('records')
        except Exception as e:
            print(f"Warning: Could not fetch columns for {table_name}: {e}")
            return []

    def classifier_agent(self, question):
        """
        Classifies the query into a domain to narrow down schema focus.
        """
        print("[Classifier Agent] Categorizing query...")
        prompt = f"""
        Question: {question}
        
        Available Domains:
        - FLIGHT: Queries about flights, status, schedules, aircraft codes.
        - BOOKING: Queries about bookings, prices, tickets, boarding passes, booking legs.
        - PASSENGER: Queries about passengers, accounts, frequent flyer levels, phone numbers.
        - AIRPORT: Queries about airport names, locations, timezones, countries.
        - MISC: General metadata or unknown.

        Task: Return ONLY the domain name in uppercase (e.g., FLIGHT).
        """
        response = self.vn.submit_prompt([{"role": "user", "content": prompt}])
        match = re.search(r'(FLIGHT|BOOKING|PASSENGER|AIRPORT|MISC)', response.upper())
        return match.group(1) if match else "MISC"

    def planner_agent(self, question, category="MISC"):
        print(f"[Planner Agent] Analyzing {category} question...")
        # Get potential context from ChromaDB
        training_data = self.vn.get_similar_question_sql(question)
        
        # Domain-driven table filtering to reduce context noise
        domain_tables = {
            "FLIGHT": ["flight", "aircraft", "airport"],
            "BOOKING": ["booking", "booking_leg", "boarding_pass", "passenger", "account"],
            "PASSENGER": ["account", "passenger", "frequent_flyer", "phone"],
            "AIRPORT": ["airport", "flight"],
            "MISC": [] # Fallback to all
        }
        
        # New: Get strict schema for all tables in the database to prevent hallucinating column names.
        # This provides a complete set of table and column information.
        from connections import _engine
        
        # Ensure engine is connected
        if _engine is None:
            connect_database(self.vn)
            from connections import _engine

        all_tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = '" + os.getenv('DB_SCHEMA', 'public') + "'"
        all_tables = pd.read_sql(all_tables_query, _engine)['table_name'].tolist()
        
        # Narrow down tables based on category to save LLM context space
        target_tables = domain_tables.get(category, [])
        if not target_tables:
            target_tables = all_tables
        
        table_context_str = ""
        for t in target_tables:
            if t in all_tables:
                cols = self.get_table_columns(t)
                if cols:
                    table_context_str += f"\nTable: {t}\nColumns: {cols}\n"

        prompt = f"""
        Category: {category}
        Question: {question}
        Available Context: {training_data}
        Strict Schema Reference:
        {table_context_str}
        
        Task: Create a step-by-step plan to solve this using SQL. 
        CRITICAL RULES:
        1. ONLY use tables and columns exactly as named in the 'Strict Schema Reference' above.
        2. DO NOT hallucinate columns or names. If you don't see a column like 'login_email', check if 'email' or 'login' exists in a related table.
        3. Identifiers guide:
           - To find a person by email, check 'booking.email' or 'account.login'.
           - To get boarding pass details, use 'boarding_pass' and join with 'passenger'.
           - To connect account to passenger, you may need a path like account -> booking -> passenger or account -> passenger.
        4. Focus on correctness: If a column doesn't exist, use the most plausible path using provided IDs.
        
        Plan format: 
        TABLES: [table1, table2]
        JOIN_LOGIC: [table1.col = table2.col]
        STEPS: [1. Filter by X, 2. Join Y, 3. Aggregate Z]
        """
        response = self.vn.submit_prompt([{"role": "user", "content": prompt}])
        return response

    def sql_agent(self, question, plan):
        """
        SQL Agent: Generates the actual SQL query based on the question and plan.
        """
        print("[SQL Agent] Generating SQL based on plan...")
        prompt = f"""
        Original Question: {question}
        Plan: {plan}
        
        Task: Write a standard SQL query to answer the question using the provided plan. 
        Use the schema-qualified format: "{os.getenv('DB_SCHEMA', 'public')}"."table_name"
        ONLY return the SQL code inside a code block.
        """
        response = self.vn.submit_prompt([{"role": "user", "content": prompt}])
        sql_match = re.search(r'```sql\n(.*?)\n```', response, re.DOTALL)
        if not sql_match:
             sql_match = re.search(r'```(.*?)```', response, re.DOTALL)
        
        return sql_match.group(1) if sql_match else response.strip()

    def validator_agent(self, question, sql):
        """
        Validator Agent: Checks the SQL for errors, runs it, and 'Self-Learns'.
        """
        print("[Validator Agent] Validating and refining SQL...")
        
        # 1. Syntax & Safety Check
        forbidden = ["drop", "delete", "update", "insert", "truncate"]
        if any(cmd in sql.lower() for cmd in forbidden):
            return None, "Blocked: Forbidden SQL command detected."

        try:
            # 2. Execution Check
            results = self.vn.run_sql(sql)
            
            # 3. Self-Learning: If successful, train Vanna automatically
            if results is not None and not results.empty:
                print("[Self-Learning] Query successful. Auto-training Vanna store...")
                self.vn.train(question=question, sql=sql)
            
            return results, None
        except Exception as e:
            # If it fails, we could potentially prompt the SQL agent to fix it (Self-Correction)
            return None, f"SQL Error: {str(e)}"

    def run(self, question):
        """
        Runs the full 4-Agent pipeline (Classifier -> Planner -> SQL -> Validator) with integrated caching.
        """
        # --- PHASE 0: Check Caching (via ChromaDB) ---
        print("[Cache Check] Searching for existing solutions...")
        cached_sql, cached_results = self.vn.get_cached_query(question)
        if cached_results is not None:
             print("[Fast Path] Returning cached results.")
             return cached_results

        # --- PHASE 1: Classification ---
        category = self.classifier_agent(question)

        # --- PHASE 2: Planning ---
        plan = self.planner_agent(question, category)
        
        # --- PHASE 3: SQL Generation ---
        sql = self.sql_agent(question, plan)
        
        # --- PHASE 4: Validation ---
        results, error = self.validator_agent(question, sql)
        
        if error:
            print(f"Pipeline Failed: {error}")
            return None
        
        print(f"\nFinal SQL: {sql} \n")
        print(f"Query executed successfully. Returning results...")
        return results

if __name__ == "__main__":
    agent = AgenticSQLPipeline()
    q = "Show me the top 5 airports by name in the US"
    df = agent.run(q)
    if df is not None:
        print(df.head())
