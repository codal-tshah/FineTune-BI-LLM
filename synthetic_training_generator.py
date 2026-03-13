import os
import json
import re
from decimal import Decimal
from datetime import datetime, date
import pandas as pd
from dotenv import load_dotenv
from connections import get_vanna_instance, connect_database, get_schema_query, get_columns_query, get_relationships_query

# Load environment variables
load_dotenv()

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (datetime, date, pd.Timestamp)):
            return obj.isoformat()
        if isinstance(obj, pd.Timedelta):
            return str(obj)
        if pd.isna(obj):
            return None
        return super(EnhancedJSONEncoder, self).default(obj)

def generate_synthetic_data(num_examples=100):
    """
    Generates synthetic training data in batches to prevent LLM truncation.
    Rotating complexity ensures a well-rounded training set (Joins, Aggregates, Advanced).
    
    Why this is useful for your LLM:
    Vanna.ai uses 'Semantic Search' (RAG). When a user asks a question, it finds the most 
    similar question-SQL pair from this training data and gives it to the SQL Engineer 
    as a 'Reference Example'. The better this data is, the smarter your SQL Engineer becomes.
    """
    vn = get_vanna_instance()
    connect_database(vn)
    
    batch_size = 10 # Small batches prevent the LLM from cutting off JSON or hallucinating
    total_batches = (num_examples // batch_size) + (1 if num_examples % batch_size > 0 else 0)

    # 1. Setup metadata
    existing_questions = set()
    existing_data = []
    if os.path.exists('generated_training_data.json'):
        try:
            with open('generated_training_data.json', 'r') as f:
                existing_data = json.load(f)
                existing_questions = {item['question'].lower().strip() for item in existing_data}
        except Exception: pass

    schema_query = get_schema_query()
    tables_df = vn.run_sql(schema_query)
    schema_context = []
    for table in tables_df['table_name']:
        cols_df = vn.run_sql(get_columns_query(table))
        cols = cols_df['column_name'].tolist()
        schema_context.append(f"Table: {table}, Columns: {', '.join(cols)}")
    
    rel_query = get_relationships_query()
    relationships = ""
    if rel_query:
        rel_df = vn.run_sql(rel_query)
        relationships = "\nRelationships (JOIN PATHS):\n" + "\n".join([
            f"{row['table_name']}.{row['column_name']} = {row['foreign_table_name']}.{row['foreign_column_name']}"
            for _, row in rel_df.iterrows()
        ])

    context_str = "\n".join(schema_context) + relationships
    schema_name = os.getenv("DB_SCHEMA", "public")
    
    generated_data = []
    success_count = 0

    # 2. Complexity Rotation Categories
    categories = [
        "BASIC: Select, Filter (WHERE), Ordering, and Counting single tables.",
        "JOINS: Multi-table queries focusing on foreign key relationships.",
        "AGGREGATES: Focus on GROUP BY, HAVING, and statistical functions (AVG, SUM, MIN, MAX).",
        "ADVANCED: Focus on DISTINCT, Window Functions, Subqueries, and Complex business logic."
    ]

    print(f"🚀 Starting generation of {num_examples} examples in {total_batches} batches...")

    for i in range(total_batches):
        if success_count >= num_examples: break
        
        current_cat = categories[i % len(categories)]
        remaining = num_examples - success_count
        current_batch_count = min(batch_size, remaining)
        
        print(f"\n--- Batch {i+1}/{total_batches} | Strategy: {current_cat.split(':')[0]} ---")
        
        # Blacklist recently generated so we don't repeat
        recent_blacklist = list(existing_questions)[-50:]
        
        prompt = f"""
        You are a Senior PostgreSQL Architect. Generate exactly {current_batch_count} DIFFERENT questions and SQL queries.
        
        CRITICAL RULE: EVERY table name MUST be prefixed with the schema: "{schema_name}"."table_name".
        FAILURE to use the schema prefix will make the query invalid.
        
        CATEGORY FOCUS: {current_cat}
        SCHEMA TABLES: {context_str}
        AVOID THESE: {recent_blacklist}

        OUTPUT FORMAT:
        Return ONLY a JSON array. 
        [
          {{"question": "...", "sql": "SELECT ... FROM \"{schema_name}\".\"table\" ..."}}
        ]
        """

        try:
            raw_response = vn.submit_prompt([{"role": "user", "content": prompt}])
            
            # --- ROBUST JSON EXTRACTION ---
            # Try to find the block between [ and ]
            json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)
            else:
                cleaned = re.sub(r'```json\s*(.*?)\s*```', r'\1', raw_response, flags=re.DOTALL).strip()
            
            batch_pairs = json.loads(cleaned)
            
            for item in batch_pairs:
                if success_count >= num_examples: break
                
                q = item.get('question', '').strip()
                sql = item.get('sql', '').strip()
                
                if not q or not sql or q.lower() in existing_questions:
                    continue

                # Safety check
                if any(cmd in sql.lower() for cmd in ["drop", "delete", "update", "insert"]):
                    continue
                
                # --- AUTO-FIXER: If LLM forgot the schema, inject it ---
                tables = tables_df['table_name'].tolist()
                for table in tables:
                    # Look for table names not already qualified with a dot or double quote
                    pattern = rf'(?<![\w\.\"]){re.escape(table)}(?![\w\.\"])'
                    if re.search(pattern, sql):
                        sql = re.sub(pattern, f'"{schema_name}"."{table}"', sql)

                try:
                    # VALIDATE AGAINST DATABASE
                    results = vn.run_sql(sql)
                    if results is not None:
                        # TRAIN VANNA
                        vn.train(question=q, sql=sql)
                        success_count += 1
                        existing_questions.add(q.lower())
                        
                        generated_data.append({
                            "question": q,
                            "sql": sql,
                            "preview": results.head(2).to_dict(orient='records')
                        })
                        print(f"  ✅ ({success_count}/{num_examples}) Trained: {q[:70]}...")
                except Exception as e:
                    print(f"  ❌ SQL Error: {str(e)[:120]}...")

        except Exception as e:
            print(f"  ⚠️ Batch {i+1} failed: {e}")

    # 4. Save results to historical file
    final_data = existing_data + generated_data
    with open('generated_training_data.json', 'w') as f:
        json.dump(final_data, f, indent=4, cls=EnhancedJSONEncoder)
        
    print(f"\n✨ DONE! Newly Generated: {success_count} | Total Knowledge Base: {len(final_data)} pairs.")

if __name__ == "__main__":
    generate_synthetic_data(num_examples=100)
