import vanna as vn
from connections import get_vanna_instance, connect_database, get_schema_query, get_columns_query

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
    print(f"Trained on schema for table {table}")

print("Training completed")