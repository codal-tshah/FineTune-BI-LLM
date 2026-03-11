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
        return super(EnhancedJSONEncoder, self).default(obj)

def generate_synthetic_data(num_examples=100):
    vn = get_vanna_instance()
    connect_database(vn)
    
    # 0. Load Existing Questions to prevent duplicates
    existing_questions = set()
    existing_data = []
    if os.path.exists('generated_training_data.json'):
        try:
            with open('generated_training_data.json', 'r') as f:
                existing_data = json.load(f)
                existing_questions = {item['question'].lower().strip() for item in existing_data}
        except Exception as e:
            print(f"Warning: Could not read existing training data: {e}")
            existing_data = []

    # 1. Extract Schema Info for the Prompt
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
        relationships = "\nRelationships:\n" + "\n".join([
            f"{row['table_name']}.{row['column_name']} = {row['foreign_table_name']}.{row['foreign_column_name']}"
            for _, row in rel_df.iterrows()
        ])

    context_str = "\n".join(schema_context) + relationships
    schema = os.getenv("DB_SCHEMA", "public")
    
    print(f"Starting synthetic data generation for {num_examples} examples...")
    
    # Send existing questions to LLM to prevent overlap in the generation phase
    exclude_list = list(existing_questions)[:20]
    
    prompt = f"""
    You are a database expert for a Business Intelligence tool.
    Based on the following database schema, generate {num_examples} diverse natural language questions and their corresponding SQL queries.
    
    IMPORTANT: DO NOT generate questions similar to these existing ones: {exclude_list}
    
    CRITICAL: You MUST use schema-qualified table names in the SQL.
    Every table reference should be in the format: \"{schema}\".\"table_name\"
    
    Schema:
    {context_str}
    
        Guidelines:
        1. Questions should range from simple (count, filter) to complex (joins, aggregations, DISTINCT, GROUP BY, window functions, subqueries).
        2. Use ONLY the tables and columns provided as shown in the schema.
        3. When multiple tables share column names, always qualify the column with a table alias (e.g., `a.account_id` vs `account_id`).
        4. Ensure the SQL is valid for {os.getenv('DB_TYPE', 'postgres')}.
        5. Provide the output in a clean JSON format: 
             [
                 {{"question": "How many passengers are there?", "sql": "SELECT count(*) FROM \"{schema}\".\"passenger\""}}
             ]
        6. No text before or after the JSON. Just the array.
    """

    generated_data = []
    success_count = 0
    pairs = []

    # 2. Call LLM to generate pairs
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant and database expert. You only output valid JSON arrays."},
            {"role": "user", "content": prompt}
        ]
        raw_response = vn.submit_prompt(messages)
    
        cleaned_response = raw_response.strip()
        if "```json" in cleaned_response:
            cleaned_response = cleaned_response.split("```json")[-1].split("```")[0].strip()
        elif "```" in cleaned_response:
            cleaned_response = cleaned_response.split("```")[-1].split("```")[0].strip()
        
        start_idx = cleaned_response.find("[")
        if start_idx != -1:
            cleaned_response = cleaned_response[start_idx:]
            
        try:
            pairs = json.loads(cleaned_response)
        except json.JSONDecodeError:
            last_brace = cleaned_response.rfind("}")
            if last_brace != -1:
                try:
                    pairs = json.loads(cleaned_response[:last_brace+1] + "]")
                except:
                    print("Failed to parse JSON even after recovery attempt.")
                    return
            else:
                return

    except Exception as e:
        print(f"Error generating questions from LLM: {str(e)}")
        return

    # 3. Validate and Train
    for item in pairs:
        question = item.get('question', '').strip()
        sql = item.get('sql')
        
        if not question or not sql:
            continue

        # --- DUPLICATE CHECK ---
        if question.lower() in existing_questions:
            print(f"Skipping: '{question}' already exists in training data.")
            continue
            
        # Safeguards
        forbidden = ["drop", "delete", "update", "insert", "truncate", "alter"]
        if any(cmd in sql.lower() for cmd in forbidden):
            print(f"Skipping forbidden query: {sql}")
            continue
            
        print(f"Testing: {question}")
        
        try:
            results = vn.run_sql(sql)
            if results is None or results.empty:
                print(f"Skipping: Query returned no data.")
                continue
                
            vn.train(question=question, sql=sql)
            success_count += 1
            existing_questions.add(question.lower())
            
            generated_data.append({
                "question": question,
                "sql": sql,
                "sample_results": results.head(5).to_dict(orient='records')
            })
            print(f"Success! Trained {success_count}/{num_examples}")
            
        except Exception as e:
            print(f"SQL Validation Error for '{question}': {str(e)}")
            continue

    # 4. Save and Update
    final_data = existing_data + generated_data
    with open('generated_training_data.json', 'w') as f:
        json.dump(final_data, f, indent=4, cls=EnhancedJSONEncoder)
        
    print(f"\nGeneration complete!")
    print(f"Total NEWLY trained this run: {success_count}")
    print(f"Total historical training records: {len(final_data)}")
    print(f"Review data updated in 'generated_training_data.json'")

if __name__ == "__main__":
    generate_synthetic_data(num_examples=100)
