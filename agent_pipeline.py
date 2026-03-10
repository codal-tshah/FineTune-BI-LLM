import os
import json
import re
import time
import pandas as pd
import connections
from connections import (
    get_vanna_instance,
    connect_database,
    get_columns_query,
    get_schema_query,
)

class AgenticSQLPipeline:
    def __init__(self):
        self.vn = get_vanna_instance()
        connect_database(self.vn)
        self.schema_tables = self._load_schema_tables()
    
    def get_table_columns(self, table_name):
        """Fetch actual column metadata from database to prevent hallucination."""
        try:
            if connections._engine is None:
                return []
            query = get_columns_query(table_name)
            result = pd.read_sql(query, connections._engine)
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
        # Filter out potential <|thought|> or other special tokens
        clean_response = re.sub(r'<[^>]*>', '', response).strip().upper()
        match = re.search(r'(FLIGHT|BOOKING|PASSENGER|AIRPORT|MISC)', clean_response)
        category = match.group(1) if match else "MISC"
        print(f"[Classifier Agent] Result: {category}")
        return category

    def planner_agent(self, question, category="MISC"):
        print(f"[Planner Agent] Analyzing {category} question...")
        # Get potential context from ChromaDB
        training_data = self.vn.get_similar_question_sql(question)

        # 2. Fetch actual foreign key relationships from DB metadata (CRITICAL FIX)
        rel_text = "No explicit foreign key relationships found."
        try:
            if connections._engine is not None:
                rel_df = pd.read_sql(connections.get_relationships_query(), connections._engine)
                if not rel_df.empty:
                    rel_text = rel_df.to_string(index=False)
        except Exception as e:
            # Fallback check: try with schema if first attempt failed
            try:
                rel_df = pd.read_sql(connections.get_relationships_query(), connections._engine)
                rel_text = rel_df.to_string(index=False)
            except Exception:
                print(f"Warning: Could not fetch relationships: {e}")
        
        # Domain-driven table filtering to reduce context noise
        domain_tables = {
            "FLIGHT": ["flight", "aircraft", "airport"],
            "BOOKING": ["booking", "booking_leg", "boarding_pass", "passenger", "account"],
            "PASSENGER": ["account", "passenger", "frequent_flyer", "phone"],
            "AIRPORT": ["airport", "flight"],
            "MISC": [] # Fallback to all
        }
        
        all_tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = '" + os.getenv('DB_SCHEMA', 'public') + "'"
        all_tables = pd.read_sql(all_tables_query, connections._engine)['table_name'].tolist()
        
        # Narrow down tables based on category to save LLM context space
        target_tables = domain_tables.get(category, [])
        if not target_tables:
            target_tables = all_tables
        
        table_context_str = ""
        for t in target_tables:
            if t in all_tables:
                cols = self.get_table_columns(t)
                if cols:
                    # Fetch small sample of data to show LLM what's inside (e.g., if IDs are NULL)
                    sample_data = "No samples available."
                    try:
                        sample_query = connections.get_data_samples_query(t, limit=3)
                        sample_df = pd.read_sql(sample_query, connections._engine)
                        sample_data = sample_df.to_dict('records')
                    except Exception:
                        pass
                    
                    table_context_str += f"\nTable: {t}\nColumns: {cols}\nSample Data: {sample_data}\n"

        prompt = f"""
        Category: {category}
        Question: {question}
        Strict Relationship Reference (USE THESE FOR JOINS - DO NOT INVENT IDS):{rel_text}
        Available Context: {training_data}
        Strict Schema Reference:{table_context_str}
        
        Task: Create a step-by-step plan to solve this using SQL. 
        CRITICAL RULES:
        1. ONLY use tables and columns exactly as named in the 'Strict Schema Reference' above.
        2. DO NOT JOIN tables unless you need columns that are not available in the primary table. Check the 'Sample Data' to see if columns are likely to be populated.
        3. DATA WARNING: In the 'passenger' table, 'account_id' is often empty/NULL. Do NOT join 'passenger' and 'account' unless the question explicitly asks for account details (like login/email). If the question is just about passenger attributes (like age), use 'passenger' table ALONE.
        4. Identifiers guide:
           - To find a person by email, check 'booking.email' or 'account.login'.
           - To get boarding pass details, use 'boarding_pass' and join with 'passenger'.
        5. Focus on correctness: If a column doesn't exist, use the most plausible path using provided IDs.
        
        Plan format: 
        TABLES: [table1, table2]
        JOIN_LOGIC: [table1.col = table2.col or 'None']
        STEPS: [1. Filter by X, 2. Join Y, 3. Aggregate Z]
        
        6. Always qualify columns with table aliases when multiple tables share the same column name.
        """
        response = self.vn.submit_prompt([{"role": "user", "content": prompt}])
        return response
 
    def log_stage(self, stage, status, detail=None):
        """Structured workflow logging for CLI visibility."""
        detail_str = f" ({detail})" if detail else ""
        print(f"[Workflow] {stage}: {status}{detail_str}")

    def sql_agent(self, question, plan, previous_sql=None, error_message=None):
        """
        SQL Agent: Generates the actual SQL query based on the question and plan.
        If previous_sql and error_message are provided, it attempts to fix the SQL.
        """
        print("[SQL Agent] Generating SQL based on plan...")
        
        correction_prompt = ""
        if previous_sql and error_message:
            print(f"[SQL Agent] Attempting to fix previous error: {error_message}")
            correction_prompt = f"""
            The previous SQL query failed:
            SQL: {previous_sql}
            Error: {error_message}
            
            Please analyze the error and Provide a FIXED SQL query.
            """

        prompt = f"""
        Original Question: {question}
        Plan: {plan}
        {correction_prompt}
        
        Task: Write a single, valid standard SQL query to answer the question using the provided plan. 
        Use the schema-qualified format: "{os.getenv('DB_SCHEMA', 'public')}"."table_name"
        CRITICAL: ONLY return the SQL code. DO NOT provide multiple options. DO NOT add commentary. DO NOT return more than one SELECT statement.
        ONLY return the SQL code inside a single code block.
        """
        response = self.vn.submit_prompt([{"role": "user", "content": prompt}])
        
        # --- ROBUST CLEANING FOR DEEPSEEK GARBAGE ---
        # 1. Remove common LLM special tokens that cause syntax errors
        # (e.g., <|begin_of_sentence|>, <|thought|>, etc.)
        clean_response = re.sub(r'<[^>]*>', '', response)
        
        # 2. Extract the actual SQL from the code block
        sql_match = re.search(r'```sql\n(.*?)\n```', clean_response, re.DOTALL)
        if not sql_match:
             sql_match = re.search(r'```(.*?)```', clean_response, re.DOTALL)
        
        final_sql = sql_match.group(1) if sql_match else clean_response.strip()
        
        # 3. Last-resort cleanup: Delete all text before "SELECT" and after the last semicolon
        if "SELECT" in final_sql.upper():
            start_pos = final_sql.upper().find("SELECT")
            final_sql = final_sql[start_pos:]
            
        if "```" in final_sql:
            final_sql = final_sql.split("```")[0].strip()

        return final_sql.strip()

    def _load_schema_tables(self):
        """Fetches all table_names from the configured schema for SQL qualification."""
        if connections._engine is None:
            connect_database(self.vn)

        try:
            query = get_schema_query()
            tables = pd.read_sql(query, connections._engine)["table_name"].tolist()
            return sorted(set(tables), key=len, reverse=True)
        except Exception as exc:
            print(f"Warning: Unable to load schema table names: {exc}")
            return []

    def _ensure_schema_qualified(self, sql):
        """Rewrites unqualified table names to include the schema."""
        if not self.schema_tables:
            return sql

        schema = os.getenv("DB_SCHEMA", "public")
        qualified_sql = sql

        for table in self.schema_tables:
            pattern = (
                rf'(?<![\w\.\"])'
                + re.escape(table)
                + r'(?![\w\.\"])'
            )
            qualified_sql = re.sub(
                pattern,
                f'"{schema}"."{table}"',
                qualified_sql,
            )

        return qualified_sql

    def validator_agent(self, question, sql):
        """
        Validator Agent: Checks the SQL for errors, runs it, and 'Self-Learns'.
        """
        print("[Validator Agent] Validating and refining SQL...")
        
        # 1. Syntax & Safety Check
        forbidden = ["drop", "delete", "update", "insert", "truncate"]
        if any(cmd in sql.lower() for cmd in forbidden):
            return None, "Blocked: Forbidden SQL command detected.", False

        trained = False
        try:
            # 2. Execution Check
            results = self.vn.run_sql(sql)
            
            # 3. Self-Learning: If successful, train Vanna automatically
            if results is not None and not results.empty:
                print("[Self-Learning] Query successful. Auto-training Vanna store...")
                self.vn.train(question=question, sql=sql)
                trained = True
            
            return results, None, trained
        except Exception as e:
            # If it fails, we could potentially prompt the SQL agent to fix it (Self-Correction)
            error_msg = str(e)
            self.vn.log_failure(question, sql, error_msg)
            return None, f"SQL Error: {error_msg}", False

    def run(self, question):
        """
        Runs the full 4-Agent pipeline (Classifier -> Planner -> SQL -> Validator) with integrated caching.
        """
        latencies = {}
        start_time = time.time()

        # --- PHASE 0: Check Caching (via ChromaDB) ---
        p0_start = time.time()
        self.log_stage("Phase 0 - Cache", "Searching for a cached solution")
        cached_sql, cached_results = self.vn.get_cached_query(question)
        latencies["cache_check"] = time.time() - p0_start

        if cached_results is not None:
            self.log_stage("Phase 0 - Cache", "Cache hit", "Returning cached results without re-running agents")
            return cached_results

        # --- PHASE 1: Classification ---
        p1_start = time.time()
        category = self.classifier_agent(question)
        latencies["classification"] = time.time() - p1_start
        self.log_stage("Phase 1 - Classification", "Completed", f"Category={category}")

        # --- PHASE 2: Planning ---
        p2_start = time.time()
        plan = self.planner_agent(question, category)
        latencies["planning"] = time.time() - p2_start
        self.log_stage("Phase 2 - Planning", "Plan drafted", "Plan text stored for SQL agent")
        
        # --- PHASE 3: SQL Generation ---
        p3_start = time.time()
        sql = self.sql_agent(question, plan)
        sql = self._ensure_schema_qualified(sql)
        latencies["sql_generation"] = time.time() - p3_start
        self.log_stage("Phase 3 - SQL Generation", "SQL created", f"{sql.splitlines()[0] if sql else 'no SQL'}...")
        
        # --- PHASE 4: Validation & Potential Self-Correction ---
        p4_start = time.time()
        results, error, trained = self.validator_agent(question, sql)
        
        # Self-Correction Loop (One retry)
        if error and "SQL Error" in error:
            self.log_stage("Phase 4 - Validation", "Failed attempt, triggering self-correction", error)
            
            # Re-run Phase 3 with error context
            p3_retry_start = time.time()
            sql = self.sql_agent(question, plan, previous_sql=sql, error_message=error)
            sql = self._ensure_schema_qualified(sql)
            latencies["sql_generation_retry"] = time.time() - p3_retry_start
            
            # Re-run Phase 4
            results, error, trained = self.validator_agent(question, sql)
        
        latencies["validation"] = time.time() - p4_start

        if trained:
            self.log_stage("Phase 4 - Validation", "Self-learning triggered", "Stored new QA pair in Vanna cache")

        total_latency = time.time() - start_time
        latencies["total"] = total_latency

        # Save Metrics
        metrics = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "category": category,
            "sql": sql,
            "success": results is not None,
            "error": error,
            "latencies": latencies
        }
        with open("metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

        if error:
            self.log_stage("Phase 4 - Validation", "Failed", error)
            return None

        self.log_stage("Phase 4 - Validation", "Success", f"SQL executed in {total_latency:.2f}s")

        print(f"\nFinal SQL:\n{sql}\n")
        print("Query executed successfully. Returning results...")
        return results

if __name__ == "__main__":
    agent = AgenticSQLPipeline()
    q = "Show me the top 5 airports by name in the US"
    df = agent.run(q)
    if df is not None:
        print(df.head())
