import vanna as vn
from connections import get_vanna_instance, connect_database

# Initialize dynamic vanna and database connection
vn = get_vanna_instance()
connect_database(vn)

def ask_with_validation(question: str):
    """Generates SQL, validates it, and runs the query."""
    print(f"Generating SQL for: {question}")
    
    # --- 1 & 2. SQL Generation Logic (Internal to Vanna) ---
    sql = vn.generate_sql(question)
    print(f"Generated SQL: {sql}")
    
    # --- 3. SQL Validation Layer ---
    if not sql or not sql.lower().strip().startswith("select"):
        print("Invalid SQL generated or model was unable to generate a select query.")
        return None
    
    # --- 4. SQL Execution ---
    try:
        results = vn.run_sql(sql)
        return results
    except Exception as e:
        print(f"SQL Execution Error: {str(e)}")
        # Optional: Feed the error back into the model to fix the query
        return None

if __name__ == "__main__":
    # Test Question
    q = "Show me the top 5 airports by name"
    df = ask_with_validation(q)
    if df is not None:
        print(df)