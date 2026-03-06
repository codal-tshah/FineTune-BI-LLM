import vanna as vn
from vanna.legacy.ollama import Ollama
from vanna.legacy.chromadb import ChromaDB_VectorStore

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

vn = MyVanna(config={'model': 'deepseek-coder:6.7b', 'path': './vanna_storage'})

vn.connect_to_postgres(
    host='localhost',
    dbname='postgres_air',
    user='postgres',
    password='postgres',
    port=5432
)

# Train schema automatically
df = vn.run_sql("SELECT table_name FROM information_schema.tables WHERE table_schema='postgres_air'")
print("df: ",df)

for table in df['table_name']:
    schema = vn.run_sql(f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = '{table}'
    """)
    
    vn.train(
        documentation=f"Table {table} has columns {schema.to_dict()}"
    )
    print(f"Trained on schema for table {table}")

print("Training completed")