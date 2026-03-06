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

def generate_synthetic_data(num_examples=10):
    vn = get_vanna_instance()
    connect_database(vn)
    
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
    
    generated_data = []
    success_count = 0
    
    prompt = f"""
    You are a database expert for a Business Intelligence tool.
    Based on the following database schema, generate {num_examples} diverse natural language questions and their corresponding SQL queries.
    
    IMPORTANT: You MUST use schema-qualified table names in the SQL.
    Every table reference should be in the format: "{schema}"."table_name"
    
    Schema:
    {context_str}
    
    Guidelines:
    1. Questions should range from simple (count, filter) to complex (joins, aggregations, subqueries).
    2. Use ONLY the tables and columns provided as shown in the schema.
    3. Ensure the SQL is valid for {os.getenv('DB_TYPE', 'postgres')}.
    4. Provide the output in a clean JSON format: 
       [
         {{"question": "How many passengers are there?", "sql": "SELECT count(*) FROM {schema}.passenger"}}
       ]
    5. No text before or after the JSON. Just the array.
    """

    # 2. Call LLM to generate pairs
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant and database expert. You only output valid JSON arrays."},
            {"role": "user", "content": prompt}
        ]
        raw_response = vn.submit_prompt(messages)
    
        # Clean up the response
        cleaned_response = raw_response.strip()
        if "```json" in cleaned_response:
            cleaned_response = cleaned_response.split("```json")[-1].split("```")[0].strip()
        elif "```" in cleaned_response:
            cleaned_response = cleaned_response.split("```")[-1].split("```")[0].strip()
        
        start_idx = cleaned_response.find("[")
        if start_idx != -1:
            cleaned_response = cleaned_response[start_idx:]
            
        # Try to parse or fix partial JSON
        try:
            pairs = json.loads(cleaned_response)
        except json.JSONDecodeError:
            # Attempt to close brackets if truncated
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
        question = item.get('question')
        sql = item.get('sql')
        
        if not question or not sql:
            continue
            
        # Safeguards
        forbidden = ["drop", "delete", "update", "insert", "truncate", "alter"]
        if any(cmd in sql.lower() for cmd in forbidden):
            print(f"Skipping forbidden query: {sql}")
            continue
            
        print(f"Testing: {question}")
        
        try:
            # 4. Execute to check validity and data
            results = vn.run_sql(sql)
            
            if results is None or results.empty:
                print(f"Skipping: Query returned no data.")
                continue
                
            # 5. Store in Vanna
            vn.train(question=question, sql=sql)
            success_count += 1
            
            # 6. Log for manual review
            generated_data.append({
                "question": question,
                "sql": sql,
                "sample_results": results.head(5).to_dict(orient='records')
            })
            print(f"Success! Trained {success_count}/{num_examples}")
            
        except Exception as e:
            print(f"SQL Validation Error for '{question}': {str(e)}")
            continue

    # 7. Save to JSON file
    with open('generated_training_data.json', 'w') as f:
        json.dump(generated_data, f, indent=4, cls=EnhancedJSONEncoder)
        
    print(f"\nGeneration complete!")
    print(f"Total trained: {success_count}")
    print(f"Review data saved to 'generated_training_data.json'")

if __name__ == "__main__":
    generate_synthetic_data(num_examples=10)
