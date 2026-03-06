import os
from dotenv import load_dotenv
import vanna as vn
from vanna.legacy.ollama import Ollama
from vanna.legacy.chromadb import ChromaDB_VectorStore

# Load environment variables
load_dotenv()

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)

def get_vanna_instance():
    config = {
        'model': os.getenv("LLM_MODEL", "deepseek-coder:6.7b"),
        'path': os.getenv("VEC_STORAGE_PATH", "./vanna_storage")
    }
    return MyVanna(config=config)

def connect_database(vn_instance):
    db_type = os.getenv("DB_TYPE", "postgres").lower()
    
    if db_type == "postgres":
        vn_instance.connect_to_postgres(
            host=os.getenv("DB_HOST", "localhost"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            port=int(os.getenv("DB_PORT", 5432))
        )
    elif db_type == "sqlite":
        vn_instance.connect_to_sqlite(os.getenv("DB_NAME"))
    elif db_type == "mysql":
        # Note: vanna.connect_to_mysql implementation might vary or need an adapter
        vn_instance.connect_to_mysql(
            host=os.getenv("DB_HOST"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            port=int(os.getenv("DB_PORT", 3306))
        )
    else:
        raise ValueError(f"Unsupported DB_TYPE: {db_type}")

def get_schema_query():
    db_type = os.getenv("DB_TYPE", "postgres").lower()
    schema = os.getenv("DB_SCHEMA", "public")
    
    if db_type == "postgres":
        return f"SELECT table_name FROM information_schema.tables WHERE table_schema='{schema}'"
    elif db_type == "mysql":
        return f"SELECT table_name FROM information_schema.tables WHERE table_schema='{os.getenv('DB_NAME')}'"
    elif db_type == "sqlite":
        return "SELECT name as table_name FROM sqlite_master WHERE type='table'"
    return ""

def get_columns_query(table_name):
    db_type = os.getenv("DB_TYPE", "postgres").lower()
    schema = os.getenv("DB_SCHEMA", "public")
    
    if db_type == "postgres":
        return f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}' AND table_schema = '{schema}'
        """
    elif db_type == "mysql":
        return f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table_name}' AND table_schema = '{os.getenv('DB_NAME')}'
        """
    elif db_type == "sqlite":
         return f"PRAGMA table_info({table_name})"
    return ""

def get_data_samples_query(table_name, limit=10):
    """Generates a query to fetch sample data for a table."""
    db_type = os.getenv("DB_TYPE", "postgres").lower()
    schema = os.getenv("DB_SCHEMA", "public")
    
    if db_type == "postgres":
        return f'SELECT * FROM "{schema}"."{table_name}" LIMIT {limit}'
    elif db_type == "mysql":
        return f'SELECT * FROM `{schema}`.`{table_name}` LIMIT {limit}'
    else:
        return f"SELECT * FROM {table_name} LIMIT {limit}"

def get_relationships_query():
    """Generates a query to fetch foreign key relationships (Postgres specific)."""
    db_type = os.getenv("DB_TYPE", "postgres").lower()
    schema = os.getenv("DB_SCHEMA", "public")
    
    if db_type == "postgres":
        return f"""
            SELECT
                tc.table_name, 
                kcu.column_name, 
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name 
            FROM 
                information_schema.table_constraints AS tc 
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                  ON ccu.constraint_name = tc.constraint_name
                  AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema='{schema}';
        """
    # Placeholder for other DBs
    return ""
