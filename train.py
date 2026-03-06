import vanna as vn
from connections import get_vanna_instance, connect_database, get_schema_query, get_columns_query, get_data_samples_query, get_relationships_query

# Initialize dynamic vanna and database connection
vn = get_vanna_instance()
connect_database(vn)

# Train schema automatically using dynamic queries
schema_query = get_schema_query()
df = vn.run_sql(schema_query)
print("Tables to train: ", df)

for table in df['table_name']:
    columns_query = get_columns_query(table)
    schema = vn.run_sql(columns_query)
    
    vn.train(
        documentation=f"Table {table} has columns {schema.to_dict()}"
    )
    
    # --- 1. Train on Data Samples (Dynamic) ---
    sample_query = get_data_samples_query(table)
    sample_data = vn.run_sql(sample_query)
    vn.train(
        documentation=f"Sample data for table {table}: {sample_data.head(5).to_dict()}"
    )
    print(f"Trained on data samples for table {table}")

# --- 2. Train on Relationships (Dynamic) ---
rel_query = get_relationships_query()
if rel_query:
    rel_data = vn.run_sql(rel_query)
    for _, row in rel_data.iterrows():
        vn.train(
            documentation=f"The table {row['table_name']} is related to {row['foreign_table_name']} via {row['column_name']} = {row['foreign_column_name']}"
        )
    print("Trained on table relationships.")

# --- 3. Train on Example SQL Queries (Suggested) ---
# For a robust BI system, add domain-specific SQL examples.
# Here we're adding some generic metadata-based SQL patterns.
example_questions = [
    (f"Show me all data from {df['table_name'].iloc[0]}", f"SELECT * FROM {df['table_name'].iloc[0]}"),
    (f"Count total records in {df['table_name'].iloc[0]}", f"SELECT count(*) FROM {df['table_name'].iloc[0]}"),
]
for q, sql in example_questions:
    vn.train(question=q, sql=sql)

print("Comprehensive Training completed")