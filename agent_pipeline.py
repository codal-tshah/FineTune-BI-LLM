import os
import json
import re
import time
import difflib
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
        self.schema_map = self._build_schema_map()
        # Cache expensive data at startup for speed
        self._cached_rel_text = self._load_relationships()
        self._cached_samples = self._load_samples()
        self._cached_all_tables = list(self.schema_map.keys())
    
    def get_table_columns(self, table_name):
        """Fetch actual column metadata from database to prevent hallucination."""
        try:
            if connections._engine is None:
                return []
            query = get_columns_query(table_name)
            result = pd.read_sql(query, connections._engine)
            # Return a simple list of column names for prompt clarity
            return result['column_name'].tolist()
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
        - FLIGHT: Queries about flight status, schedules, aircraft codes, or departures/arrivals.
        - BOOKING: Queries about bookings, prices, tickets, boarding passes, booking legs, or seat assignments.
        - PASSENGER: Queries about passengers, accounts, frequent flyer levels, phone numbers, or person details.
        - AIRPORT: Queries about airport names, locations, timezones, countries, or geographic codes.
        - MISC: General metadata, unknown, or systemic queries.

        Task: Return ONLY the domain name in uppercase (e.g., BOOKING). If you are unsure, return MISC.
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

        # Use cached relationships (loaded once at startup)
        rel_text = self._cached_rel_text
        
        # Domain-driven table filtering to reduce context noise
        domain_tables = {
            "FLIGHT": ["flight", "aircraft", "airport", "booking_leg"],
            "BOOKING": ["booking", "booking_leg", "boarding_pass", "passenger", "account"],
            "PASSENGER": ["account", "passenger", "frequent_flyer", "phone", "boarding_pass"],
            "AIRPORT": ["airport", "flight"],
            "MISC": [] # Fallback to all
        }
        
        all_tables = self._cached_all_tables
        
        # Hybrid table selection to prevent category-induced blindness
        target_tables = set(domain_tables.get(category, []))
        
        # 1. Add tables mentioned in training examples (context-aware)
        if training_data:
            for example in training_data:
                sql_words = re.findall(r'[\w\"]+', example.get('sql', ''))
                for word in sql_words:
                    word_clean = word.strip('"').lower()
                    if word_clean in all_tables:
                        target_tables.add(word_clean)
        
        # 2. Keyword-based table selection (failsafe)
        question_words = set(re.findall(r'\w+', question.lower()))
        for t in all_tables:
            t_clean = t.replace('_', ' ')
            if t in question_words or any(word in question.lower() for word in t_clean.split()):
                target_tables.add(t)
        
        # Fallback to all if still empty
        if not target_tables:
            target_tables = set(all_tables)
        
        # Build table context from cached data (no DB queries!)
        table_context_str = ""
        for t in sorted(list(target_tables)):
            if t in self.schema_map:
                cols = self.schema_map[t]
                sample_data = self._cached_samples.get(t, "No samples available.")
                table_context_str += f"\nTable: {t}\nColumns: {cols}\nSample Data: {sample_data}\n"

        # 3. Sanitize training context to prevent internal ID leakage
        clean_context = "No relevant examples found."
        if training_data:
            context_parts = []
            for item in training_data:
                q = item.get('question', 'N/A')
                s = item.get('sql', 'N/A')
                context_parts.append(f"Question: {q}\nSQL: {s}")
            clean_context = "\n---\n".join(context_parts)

        prompt = f"""
        Category: {category}
        Question: {question}
        Strict Relationship Reference (USE THESE FOR JOINS - DO NOT INVENT IDS):{rel_text}
        
        Potential Context (USE SELECTIVELY - DO NOT COPY VALUES OR IDS): {clean_context}
        Strict Schema Reference:{table_context_str}
        
        Task: Create a step-by-step plan to solve this using SQL. 
        CRITICAL RULES:
        1. ONLY use tables and columns exactly as named in the 'Strict Schema Reference' above.
        2. NO HALLUCINATION: DO NOT assume a table has a generic 'id' or 'uid' column. Look at the column names provided. If it has 'passenger_id', use that.
        3. DO NOT JOIN tables unless you need columns that are not available in the primary table. Check the 'Sample Data' to see if columns are likely to be populated.
        3. DATA WARNING: In the 'passenger' table, 'account_id' is often empty/NULL. Do NOT join 'passenger' and 'account' unless the question explicitly asks for account details (like login/email). If the question is just about passenger attributes (like age), use 'passenger' table ALONE.
        4. CONTEXT PRUNING: Only use retrieved 'Potential Context' if it is DIRECTLY RELEVANT to the current question. If the context contains filters (like age, date) that are NOT mentioned in the question, DISCARD THOSE FILTERS.
        5. JOIN ENFORCEMENT: Every table mentioned in TABLES must have a corresponding entry in JOIN_LOGIC (unless only one table is used). Ensure you have a clear path (e.g., A -> B -> C).
        6. Identifiers guide:
           - To find a person by email, check 'booking.email' or 'account.login'.
           - To get boarding pass details, use 'boarding_pass' and join with 'passenger'.
        7. Focus on correctness: If a column doesn't exist, use the most plausible path using provided IDs.
        
        Plan format: 
        TABLES: [table1, table2]
        JOIN_LOGIC: [table1.col = table2.col or 'None']
        STEPS: [1. Filter by X, 2. Join Y, 3. Aggregate Z]
        
        8. Always qualify columns with table aliases when multiple tables share the same column name.
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

        # Build a compact schema reference for the SQL Agent
        schema_ref = ""
        for table, cols in self.schema_map.items():
            schema_ref += f"\n  {table}: [{', '.join(cols)}]"

        prompt = f"""
        Original Question: {question}
        Plan: {plan}
        {correction_prompt}
        
        EXACT DATABASE SCHEMA (use ONLY these table and column names):{schema_ref}

        JOIN PATH REFERENCE (use THESE for multi-table queries):
        {self._cached_rel_text}

        Task: Write a single, valid standard SQL query to answer the question using the provided plan. 
        Use the schema-qualified format: "{os.getenv('DB_SCHEMA', 'public')}"."table_name"
        CRITICAL: 
        - ONLY use column names from the EXACT DATABASE SCHEMA above. Cross-check every column.
        - For multi-table queries, follow the JOIN PATH REFERENCE above to find the correct join columns.
        - DO NOT guess or hallucinate generic 'id' columns. If the table has 'passenger_id', use that.
        - ONLY return the SQL code inside a single code block. 
        - DO NOT provide multiple options. 
        - DO NOT add commentary. 
        - DO NOT return more than one SELECT statement.
        - DO NOT use parameterized queries (e.g., :age, :name, $1). Use literal values or remove the filter if the value is unknown.
        """
        response = self.vn.submit_prompt([{"role": "user", "content": prompt}])
        
        # --- ROBUST CLEANING FOR DEEPSEEK GARBAGE ---
        # 1. Remove common LLM special tokens that cause syntax errors
        clean_response = re.sub(r'<[^>]*>', '', response)
        
        # 2. STRIP UUIDs and internal malformed markers (Critical Fix)
        # Matches patterns like 30007612-d5b6... or :30007612...
        clean_response = re.sub(r':?[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', '', clean_response)
        # Strip colon-prefixed hex strings that often appear as hallucinated parameters
        clean_response = re.sub(r':[a-fA-F0-9]{8,}', '', clean_response)
        
        # 3. Extract the actual SQL from the code block
        sql_match = re.search(r'```sql\n(.*?)\n```', clean_response, re.DOTALL)
        if not sql_match:
             sql_match = re.search(r'```(.*?)```', clean_response, re.DOTALL)
        
        final_sql = sql_match.group(1) if sql_match else clean_response.strip()
        
        # 3. Last-resort cleanup: Delete all text before "SELECT" and after the first semicolon
        if "SELECT" in final_sql.upper():
            start_pos = final_sql.upper().find("SELECT")
            final_sql = final_sql[start_pos:]
            
        if "```" in final_sql:
            final_sql = final_sql.split("```")[0].strip()

        # 4. Multi-query guardrail: If there are multiple queries, take only the first one
        if ";" in final_sql:
            # Check if there's significant content after the first semicolon
            parts = final_sql.split(";")
            if len(parts) > 1 and "SELECT" in parts[1].upper():
                print("[SQL Agent] Warning: LLM returned multiple queries. Extracting the first one...")
                final_sql = parts[0] + ";"

        # 5. Auto-LIMIT: Prevent massive result sets (safety guard)
        if final_sql and 'LIMIT' not in final_sql.upper():
            # Strip trailing semicolon, add LIMIT, then re-add semicolon
            final_sql = final_sql.rstrip().rstrip(';')
            final_sql += "\nLIMIT 10;"
            print("[SQL Agent] Auto-added LIMIT 10 (no LIMIT clause found)")

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

    def _build_schema_map(self):
        """Pre-load {table_name: [column_names]} for fast SQL validation."""
        schema_map = {}
        for table in self.schema_tables:
            cols = self.get_table_columns(table)
            if cols:
                schema_map[table] = cols
        print(f"[Schema Map] Loaded {len(schema_map)} tables for SQL pre-validation.")
        return schema_map

    def _load_relationships(self):
        """Load FK relationships once at startup."""
        try:
            if connections._engine is not None:
                rel_df = pd.read_sql(connections.get_relationships_query(), connections._engine)
                if not rel_df.empty:
                    return rel_df.to_string(index=False)
        except Exception as e:
            print(f"Warning: Could not fetch relationships: {e}")
        return "No explicit foreign key relationships found."

    def _load_samples(self):
        """Load sample data for all tables once at startup."""
        samples = {}
        for table in self.schema_tables:
            try:
                sample_query = connections.get_data_samples_query(table, limit=3)
                sample_df = pd.read_sql(sample_query, connections._engine)
                samples[table] = sample_df.to_dict('records')
            except Exception:
                samples[table] = "No samples available."
        print(f"[Samples Cache] Loaded samples for {len(samples)} tables.")
        return samples

    def _validate_sql(self, sql):
        """
        Code-level SQL pre-validator. Catches and fixes common LLM errors:
        1. Fuzzy-match misspelled/truncated table names
        2. Validate column references against actual schema
        """
        if not self.schema_map or not sql:
            return sql

        schema = os.getenv("DB_SCHEMA", "public")
        valid_tables = list(self.schema_map.keys())
        fixed_sql = sql

        # --- Step 1: Fix misspelled/truncated table names ---
        # Find all table-like references: schema.table or standalone table names
        table_refs = re.findall(
            rf'(?:"{schema}"\."|{schema}\.)([a-zA-Z_][a-zA-Z0-9_]*)',
            fixed_sql
        )
        # Also find standalone table-like words after FROM/JOIN
        standalone_refs = re.findall(
            r'(?:FROM|JOIN)\s+(?:"?[a-zA-Z_]+"?\.)?"?([a-zA-Z_][a-zA-Z0-9_]*)"?',
            fixed_sql,
            re.IGNORECASE
        )
        all_table_refs = set(table_refs + standalone_refs)

        for ref in all_table_refs:
            ref_clean = ref.strip('"').lower()
            if ref_clean in valid_tables or ref_clean == schema:
                continue  # Already valid

            # Fuzzy match against actual tables
            matches = difflib.get_close_matches(ref_clean, valid_tables, n=1, cutoff=0.5)
            if matches:
                correct_table = matches[0]
                print(f"[SQL Pre-Validator] Fixing table: '{ref}' → '{correct_table}'")
                # Replace the misspelled table name (handle both quoted and unquoted)
                fixed_sql = re.sub(
                    rf'(?<![a-zA-Z_]){re.escape(ref)}(?![a-zA-Z_])',
                    correct_table,
                    fixed_sql
                )

        # --- Step 2: Validate column references (alias.column or table.column) ---
        # Build alias map from the SQL: alias -> table_name
        alias_map = {}
        alias_pattern = re.findall(
            r'(?:FROM|JOIN)\s+(?:"?[a-zA-Z_]+"?\.)?"?([a-zA-Z_][a-zA-Z0-9_]*)"?\s+(?:AS\s+)?([a-zA-Z_]\w*)',
            fixed_sql,
            re.IGNORECASE
        )
        for table_ref, alias in alias_pattern:
            table_clean = table_ref.strip('"').lower()
            alias_clean = alias.strip('"').lower()
            # Skip SQL keywords that look like aliases
            if alias_clean in ('on', 'where', 'group', 'order', 'limit', 'having', 'and', 'or', 'as', 'join', 'inner', 'left', 'right', 'full'):
                continue
            if table_clean in self.schema_map:
                alias_map[alias_clean] = table_clean

        # Find all alias.column or table.column references
        col_refs = re.findall(r'([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)', fixed_sql)
        for qualifier, column in col_refs:
            q_lower = qualifier.strip('"').lower()
            c_lower = column.strip('"').lower()

            # Skip schema qualifiers (e.g., postgres_air.table)
            if q_lower == schema:
                continue

            # Resolve alias to actual table name
            actual_table = alias_map.get(q_lower, q_lower)

            if actual_table not in self.schema_map:
                continue  # Can't validate unknown tables

            valid_columns = self.schema_map[actual_table]
            valid_columns_lower = [c.lower() for c in valid_columns]

            if c_lower not in valid_columns_lower:
                # Try fuzzy match
                matches = difflib.get_close_matches(c_lower, valid_columns_lower, n=1, cutoff=0.6)
                if matches:
                    correct_col = valid_columns[valid_columns_lower.index(matches[0])]
                    print(f"[SQL Pre-Validator] Fixing column: '{qualifier}.{column}' → '{qualifier}.{correct_col}'")
                    fixed_sql = fixed_sql.replace(
                        f"{qualifier}.{column}",
                        f"{qualifier}.{correct_col}"
                    )
                else:
                    print(f"[SQL Pre-Validator] Warning: Column '{qualifier}.{column}' not found in '{actual_table}'. Valid: {valid_columns}")

        return fixed_sql

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
        # Pre-validate SQL against actual schema (fuzzy fix tables/columns)
        sql = self._validate_sql(sql)
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
            sql = self._validate_sql(sql)
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
