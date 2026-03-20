import os
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
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

# Global locks for thread safety
train_lock = threading.Lock()

def process_batch(batch_idx, batch_count, strategy, context_str, schema_name, vn, tables_list, existing_questions):
    """Worker function to process a single batch of synthetic queries."""
    successes = []
    print(f"\n[Batch {batch_idx}] Strategy: {strategy.split(':')[0]} | Starting...")
    
    # Avoid recent duplicates
    recent_blacklist = list(existing_questions)[-50:]
    
    prompt = f"""
    You are a Senior PostgreSQL Architect. Generate exactly {batch_count} DIFFERENT questions and SQL queries.
    
    CRITICAL RULE: EVERY table name MUST be prefixed with the schema: "{schema_name}"."table_name".
    
    CATEGORY FOCUS: {strategy}
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
        json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
        cleaned = json_match.group(0) if json_match else raw_response
        batch_pairs = json.loads(cleaned)
        
        for item in batch_pairs:
            q = item.get('question', '').strip()
            sql = item.get('sql', '').strip()
            if not q or not sql: continue

            # Auto-Fixer: Inject schema if missing
            for table in tables_list:
                pattern = rf'(?<![\w\.\"]){re.escape(table)}(?![\w\.\"])'
                if re.search(pattern, sql, re.IGNORECASE):
                    sql = re.sub(pattern, f'"{schema_name}"."{table}"', sql, flags=re.IGNORECASE)

            try:
                # Validation
                results = vn.run_sql(sql)
                if results is not None:
                    # Thread-safe training
                    with train_lock:
                        vn.train(question=q, sql=sql)
                        existing_questions.add(q.lower())
                    
                    successes.append({
                        "question": q,
                        "sql": sql,
                        "preview": results.head(2).to_dict(orient='records')
                    })
                    print(f"  [Batch {batch_idx}] ✅ Success: {q[:50]}...")
            except Exception as e:
                # print(f"  [Batch {batch_idx}] ❌ SQL Error: {str(e)[:100]}...")
                pass

    except Exception as e:
        print(f"  [Batch {batch_idx}] ⚠️ Batch failed: {e}")
    
    return successes

def generate_synthetic_data(num_examples=100, max_workers=4):
    """
    Generates synthetic training data using multi-threading for speed.
    """
    vn = get_vanna_instance()
    connect_database(vn)
    
    batch_size = 10
    total_batches = (num_examples // batch_size) + (1 if num_examples % batch_size > 0 else 0)

    # Setup metadata (identical to previous version)
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
    tables_list = tables_df['table_name'].tolist()
    
    schema_context = []
    for table in tables_list:
        cols_df = vn.run_sql(get_columns_query(table))
        cols = cols_df['column_name'].tolist()
        schema_context.append(f"Table: {table}, Columns: {', '.join(cols)}")
    
    rel_query = get_relationships_query()
    relationships = ""
    if rel_query:
        rel_df = vn.run_sql(rel_query)
        relationships = "\nRelationships:\n" + "\n".join([
            f"{row['table_name']}.{row['column_name']} = {row['foreign_table_name']}.{row['foreign_column_name']}"
            for _, row in rel_df.iterrows()
        ])

    context_str = "\n".join(schema_context) + relationships
    schema_name = os.getenv("DB_SCHEMA", "public")
    
    categories = [
        "BASIC: Select, Filter, Count.",
        "JOINS: Multi-table queries (flight + aircraft, passenger + frequent_flyer).",
        "AGGREGATES: GROUP BY, HAVING, complex statistics.",
        "ADVANCED: Window Functions, DISTINCT, CASE WHEN logic."
    ]

    print(f"🚀 Starting parallel generation ({max_workers} threads) for {num_examples} examples...")

    all_new_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(total_batches):
            remaining = num_examples - (i * batch_size)
            if remaining <= 0: break
            
            futures.append(executor.submit(
                process_batch, 
                i+1, 
                min(batch_size, remaining), 
                categories[i % len(categories)],
                context_str,
                schema_name,
                vn,
                tables_list,
                existing_questions
            ))
        
        for future in futures:
            all_new_data.extend(future.result())

    # Save results
    final_data = existing_data + all_new_data
    with open('generated_training_data.json', 'w') as f:
        json.dump(final_data, f, indent=4, cls=EnhancedJSONEncoder)
        
    print(f"\n✨ DONE! Parallel generation complete.")
    print(f"Newly Trained: {len(all_new_data)} | Total knowledge: {len(final_data)} pairs.")

if __name__ == "__main__":
    generate_synthetic_data(num_examples=100, max_workers=4)
