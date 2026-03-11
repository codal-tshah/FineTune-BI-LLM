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

class AgenticSQLPipeline_V2:
    def __init__(self):
        self.vn = get_vanna_instance()
        connect_database(self.vn)
        self.schema_tables = self._load_schema_tables()
        self.schema_map = self._build_schema_map()
        # Cache expensive data at startup for speed
        self._cached_rel_text = self._load_relationships()
        self._cached_samples = self._load_samples()
        self._cached_all_tables = list(self.schema_map.keys())

        # MULTI-MODEL CONFIGURATION (Optimized for 16GB Mac)
        # Architect (3B) for planning/reasoning, SQL (6.7B) for coding
        self.models = {
            "architect": os.getenv("ARCHITECT_MODEL", "qwen2.5-coder:3b"),
            "sql": os.getenv("LLM_MODEL", "deepseek-coder:6.7b")
        }
        print("[Pipeline] Tiered Orchestration active:")
        print(f"  - Architect: {self.models['architect']}")
        print(f"  - SQL Engineer: {self.models['sql']}")
    
    def get_table_columns(self, table_name):
        """Fetch actual column metadata from database to prevent hallucination."""
        try:
            if connections.get_engine() is None:
                return []
            query = get_columns_query(table_name)
            result = pd.read_sql(query, connections.get_engine())
            return result['column_name'].tolist()
        except Exception as e:
            print(f"Warning: Could not fetch columns for {table_name}: {e}")
            return []

    def architect_agent(self, question):
        """
        Phase 1: Merges Classification & Planning.
        Uses a smaller, faster model (3B) for mapping question to schema.
        """
        print("[Architect Agent] Designing query strategy...")
        
        # 1. Hybrid Table Selection Logic (Robust)
        target_tables = set()
        all_tables = self._cached_all_tables
        
        # A. Keyword matching (fail-safe)
        question_words = set(re.findall(r'\w+', question.lower()))
        for t in all_tables:
            t_clean = t.replace('_', ' ')
            if t in question_words or any(word in question.lower() for word in t_clean.split()):
                target_tables.add(t)
        
        # B. Context-aware tables (from training examples)
        training_data = self.vn.get_similar_question_sql(question)
        if training_data:
            for item in training_data:
                sql_words = re.findall(r'[\w\"]+', item.get('sql', ''))
                for word in sql_words:
                    word_clean = word.strip('"').lower()
                    if word_clean in all_tables:
                        target_tables.add(word_clean)

        # C. Domain-Driven Expansion (Ensures bridges like booking_leg are included)
        domain_keywords = {
            "flight": ["flight", "aircraft", "airport", "booking_leg"],
            "booking": ["booking", "booking_leg", "passenger", "boarding_pass"],
            "passenger": ["passenger", "account", "frequent_flyer", "booking_leg"],
            "airport": ["airport"]
        }
        for domain, tables in domain_keywords.items():
            if domain in question.lower():
                for t in tables:
                    if t in all_tables:
                        target_tables.add(t)

        # Fallback if still empty
        if not target_tables:
            target_tables = set(all_tables[:10])

        # 2. Build Table Context & Mask Values
        table_context_str = ""
        for t in sorted(list(target_tables)):
            if t in self.schema_map:
                cols = self.schema_map[t]
                samples = self._cached_samples.get(t, "N/A")
                table_context_str += f"\nTable: {t}\nColumns: {cols}\nSamples: {samples}\n"

        clean_context = "No relevant examples found."
        if training_data:
            context_parts = []
            for item in training_data:
                s_masked = re.sub(r"'(.*?)'", "'<VALUE>'", item.get('sql', ''))
                context_parts.append(f"Q: {item.get('question', 'N/A')}\nSQL: {s_masked}")
            clean_context = "\n---\n".join(context_parts)

        prompt = f"""
        Question: {question}
        EXPLICIT SCHEMA REFERENCE:{table_context_str}
        STRICT RELATIONSHIPS (JOIN PATHS):{self._cached_rel_text}
        STRUCTURAL EXAMPLES (DO NOT COPY VALUES): {clean_context}
        
        Task: Create a step-by-step SQL JOIN plan.
        CRITICAL RULES:
        1. ONLY use tables/columns listed in EXPLICIT SCHEMA.
        2. NO HALLUCINATION: Do not assume columns exist.
        3. NO FILTER HALLUCINATION: Do not add WHERE values unless they are in the 'Question'.
        4. SHORTEST PATH: Join booking_leg and passenger directly using 'booking_id'.
        5. JOIN ENFORCEMENT: Every table in TABLES must have a JOIN_LOGIC entry.
        6. IDENTIFIER GUIDE:
           - Link flight and passenger via: flight -> booking_leg -> passenger.
           - Path: flight.flight_id = booking_leg.flight_id, booking_leg.booking_id = passenger.booking_id.
        
        Output Format:
        CATEGORY: [FLIGHT|BOOKING|PASSENGER|AIRPORT|MISC]
        TABLES: [table1, table2, table3]
        JOIN_LOGIC: [table1.id = table2.id, table2.id = table3.id]
        STEPS: [1. filter X, 2. join Y, 3. count Z]
        """
        response = self.vn.submit_prompt_with_model([{"role": "user", "content": prompt}], model=self.models["architect"])
        return response

    def sql_agent(self, question, plan, previous_sql=None, error_message=None):
        """
        Phase 2: SQL Generation. Uses the heavy 6.7B model for precise coding.
        Includes robust output cleaning and schema pruning.
        """
        print("[SQL Agent] Generating SQL based on plan...")
        
        correction_prompt = ""
        if previous_sql and error_message:
            correction_prompt = f"Previous attempt failed with error: {error_message}\nFailed SQL: {previous_sql}\nFix this error."

        # Dynamic Schema Pruning
        plan_tables = []
        table_section = re.search(r'TABLES:\s*\[?([\w\s,]+)\]?', plan, re.IGNORECASE)
        if table_section:
            plan_tables = [t.strip().lower() for t in table_section.group(1).split(',')]
        
        schema_ref = ""
        tables_found = 0
        for t in plan_tables:
            if t in self.schema_map:
                schema_ref += f"\n  {t}: [{', '.join(self.schema_map[t])}]"
                tables_found += 1
        
        # Fallback to full schema if parsing failed
        if tables_found == 0:
            for t, cols in self.schema_map.items():
                schema_ref += f"\n  {t}: [{', '.join(cols)}]"

        prompt = f"""
        Original Question: {question}
        Verified Plan: {plan}
        {correction_prompt}
        PRUNED SCHEMA REFERENCE (ONLY USE THESE):{schema_ref}
        JOIN PATH REFERENCE: {self._cached_rel_text}

        Task: Write a single, valid standard SQL query.
        - Use schema format: "{os.getenv('DB_SCHEMA', 'public')}"."table_name"
        - DO NOT USE CTEs (WITH clauses).
        - DO NOT HALLUCINATE VALUES like 'LAX'.
        - Follow the JOIN_LOGIC in the Plan strictly.
        - ONLY return SQL code.
        """
        response = self.vn.submit_prompt_with_model([{"role": "user", "content": prompt}], model=self.models["sql"])
        
        # --- ROBUST CLEANING (From original logic) ---
        sql = re.sub(r'<[^>]*>', '', response) # Remove thinking tokens
        # Strip UUIDs and internal hex markers
        sql = re.sub(r':?[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}', '', sql)
        sql = re.sub(r':[a-fA-F0-9]{8,}', '', sql)
        
        sql_match = re.search(r'```sql\n(.*?)\n```', sql, re.DOTALL | re.IGNORECASE)
        if not sql_match: sql_match = re.search(r'```(.*?)```', sql, re.DOTALL)
        final_sql = sql_match.group(1).strip() if sql_match else sql.strip()

        if "SELECT" in final_sql.upper():
            start_pos = final_sql.upper().find("SELECT")
            final_sql = final_sql[start_pos:]
        
        if "```" in final_sql:
            final_sql = final_sql.split("```")[0].strip()

        if ";" in final_sql:
            parts = final_sql.split(";")
            final_sql = parts[0] + ";"
        
        # Auto-LIMIT
        if final_sql and 'LIMIT' not in final_sql.upper():
            final_sql = final_sql.rstrip().rstrip(';') + "\nLIMIT 10;"

        return final_sql.strip()

    def validator_agent(self, question, sql):
        """Phase 3: Code-level safety + Execution + Self-Learning."""
        print("[Validator Agent] Executing SQL...")
        
        # Safety Check
        forbidden = ["drop", "delete", "update", "insert", "truncate"]
        if any(cmd in sql.lower() for cmd in forbidden):
            return None, "Blocked: Forbidden command.", False

        try:
            results = self.vn.run_sql(sql)
            if results is not None and not results.empty:
                print("[Self-Learning] Success. Auto-training...")
                self.vn.train(question=question, sql=sql)
                return results, None, True
            return results, None, False
        except Exception as e:
            error_msg = str(e)
            self.vn.log_failure(question, sql, error_msg)
            return None, error_msg, False

    def run(self, question):
        latencies = {}
        start_time = time.time()

        # --- Phase 0: Cache ---
        p0_start = time.time()
        self.log_stage("Phase 0 - Cache", "Checking semantic cache")
        _, cached_results = self.vn.get_cached_query(question)
        latencies["cache"] = time.time() - p0_start
        if cached_results is not None:
            return cached_results

        # --- Phase 1: Architect (3B) ---
        p1_start = time.time()
        plan = self.architect_agent(question)
        latencies["architect"] = time.time() - p1_start
        self.log_stage("Phase 1 - Architect", "Strategy designed")

        # --- Phase 2: SQL Engineer (6.7B) ---
        p2_start = time.time()
        sql = self.sql_agent(question, plan)
        sql = self._ensure_schema_qualified(sql)
        sql = self._validate_sql(sql)
        latencies["sql_gen"] = time.time() - p2_start
        self.log_stage("Phase 2 - SQL", "SQL Generated")

        # --- Phase 3: Validation ---
        p3_start = time.time()
        results, error, trained = self.validator_agent(question, sql)
        
        # Self-Correction Loop
        if error:
            self.log_stage("Phase 3 - Validator", "Error found, retrying...", error)
            p2_retry = time.time()
            sql = self.sql_agent(question, plan, previous_sql=sql, error_message=error)
            sql = self._ensure_schema_qualified(sql)
            sql = self._validate_sql(sql)
            results, error, trained = self.validator_agent(question, sql)
            latencies["sql_retry"] = time.time() - p2_retry

        latencies["validation"] = time.time() - p3_start
        total_time = time.time() - start_time
        latencies["total"] = total_time

        # Metrics
        metrics = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "sql": sql,
            "success": results is not None,
            "error": error,
            "latencies": latencies
        }
        with open("metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        
        if error:
            print(f"🛑 Error: {error}")
            return None
            
        self.log_stage("Phase 3 - Validator", "Success", f"Total: {total_time:.1f}s")
        return results

    def log_stage(self, stage, status, detail=None):
        detail_str = f" ({detail})" if detail else ""
        print(f"[Workflow] {stage}: {status}{detail_str}")

    def _load_schema_tables(self):
        try:
            query = get_schema_query()
            tables = pd.read_sql(query, connections.get_engine())["table_name"].tolist()
            return sorted(set(tables), key=len, reverse=True)
        except Exception:
            return []

    def _build_schema_map(self):
        schema_map = {}
        for table in self.schema_tables:
            cols = self.get_table_columns(table)
            if cols:
                schema_map[table] = cols
        return schema_map

    def _load_relationships(self):
        try:
            rel_df = pd.read_sql(connections.get_relationships_query(), connections.get_engine())
            if not rel_df.empty:
                return rel_df.to_string(index=False)
            return "None"
        except Exception:
            return "None"

    def _load_samples(self):
        samples = {}
        for table in self.schema_tables:
            try:
                sample_df = pd.read_sql(connections.get_data_samples_query(table, limit=3), connections.get_engine())
                samples[table] = sample_df.to_dict('records')
            except Exception:
                samples[table] = "N/A"
        return samples

    def _validate_sql(self, sql):
        """
        Code-level SQL pre-validator. Catches and fixes common LLM errors:
        1. Fuzzy-match misspelled/truncated table names
        2. Validate column references against actual schema (Alias-aware)
        """
        if not self.schema_map or not sql:
            return sql

        schema = os.getenv("DB_SCHEMA", "public")
        valid_tables = list(self.schema_map.keys())
        fixed_sql = sql

        # --- Step 1: Fix misspelled/truncated table names ---
        table_refs = re.findall(rf'(?:"{schema}"\."|{schema}\.)([a-zA-Z_]\w*)', fixed_sql)
        standalone_refs = re.findall(r'(?:FROM|JOIN)\s+(?:"?[\w]+"?\.)?"?([a-zA-Z_]\w*)"?', fixed_sql, re.IGNORECASE)
        all_table_refs = set(table_refs + standalone_refs)

        for ref in all_table_refs:
            ref_clean = ref.strip('"').lower()
            if ref_clean in valid_tables or ref_clean == schema:
                continue
            matches = difflib.get_close_matches(ref_clean, valid_tables, n=1, cutoff=0.5)
            if matches:
                correct_table = matches[0]
                print(f"[SQL Pre-Validator] Fixing table: '{ref}' → '{correct_table}'")
                fixed_sql = re.sub(rf'(?<![a-zA-Z_]){re.escape(ref)}(?![a-zA-Z_])', correct_table, fixed_sql)

        # --- Step 2: Validate column references (alias.column or table.column) ---
        alias_map = {}
        # Improved alias extraction: MATCHING "table AS alias" or "table alias"
        alias_pattern = re.findall(
            r'(?:FROM|JOIN)\s+(?:"?[\w]+"?\.)?"?([a-zA-Z_]\w*)"?\s+(?:AS\s+)?([a-zA-Z_]\w*)',
            fixed_sql,
            re.IGNORECASE
        )
        for table_ref, alias in alias_pattern:
            table_clean = table_ref.strip('"').lower()
            alias_clean = alias.strip('"').lower()
            if alias_clean in ('on', 'where', 'group', 'order', 'limit', 'and', 'join'):
                continue
            if table_clean in self.schema_map:
                alias_map[alias_clean] = table_clean

        # Find all qualifier.column references
        col_refs = re.findall(r'([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)', fixed_sql)
        for qualifier, column in col_refs:
            q_lower = qualifier.strip('"').lower()
            c_lower = column.strip('"').lower()
            if q_lower == schema:
                continue

            actual_table = alias_map.get(q_lower, q_lower)
            if actual_table in self.schema_map:
                valid_cols = self.schema_map[actual_table]
                valid_cols_lower = [c.lower() for c in valid_cols]
                
                if c_lower not in valid_cols_lower:
                    matches = difflib.get_close_matches(c_lower, valid_cols_lower, n=1, cutoff=0.6)
                    if matches:
                        correct_col = valid_cols[valid_cols_lower.index(matches[0])]
                        print(f"[SQL Pre-Validator] Fixing column: '{qualifier}.{column}' → '{qualifier}.{correct_col}'")
                        fixed_sql = fixed_sql.replace(f"{qualifier}.{column}", f"{qualifier}.{correct_col}")
        
        return fixed_sql

    def _ensure_schema_qualified(self, sql):
        if not self.schema_tables: return sql
        schema = os.getenv("DB_SCHEMA", "public")
        qualified_sql = sql
        for table in self.schema_tables:
            pattern = rf'(?<![\w\.\"]){re.escape(table)}(?![\w\.\"])'
            qualified_sql = re.sub(pattern, f'"{schema}"."{table}"', qualified_sql)
        return qualified_sql

if __name__ == "__main__":
    agent = AgenticSQLPipeline_V2()
    # Test complex query
    print(agent.run("names of all passengers on flight id 100"))
