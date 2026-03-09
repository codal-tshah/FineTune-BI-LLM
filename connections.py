import os
import json
from dotenv import load_dotenv
import vanna as vn
from vanna.legacy.ollama import Ollama
from vanna.legacy.chromadb import ChromaDB_VectorStore
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Load environment variables
load_dotenv()

# Global engine for connection pooling
_engine = None

class MyVanna(ChromaDB_VectorStore, Ollama):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        Ollama.__init__(self, config=config)
        show_prompts = os.getenv("VANNA_SHOW_PROMPTS", "false").lower() in ("1", "true", "yes")
        self._suppress_vanna_prompts = not show_prompts
        self._prompt_log_notice_printed = False

    def log(self, message: str, title: str = "Info"):
        """Suppress verbose Ollama logging unless the CLI opt-in flag is set."""

        if self._suppress_vanna_prompts and isinstance(message, str):
            suppressed_prefixes = (
                "Ollama parameters:",
                "Prompt Content:",
                "Ollama Response:",
            )
            if message.startswith(suppressed_prefixes):
                if not self._prompt_log_notice_printed:
                    super().log(
                        "Vanna prompt logging suppressed (set VANNA_SHOW_PROMPTS=1 to reveal)",
                        title=title,
                    )
                    self._prompt_log_notice_printed = True
                return

        super().log(message, title)

    def get_cached_query(self, question: str):
        """
        Modified to handle semantic similarity correctly using ChromaDB's search.
        Even if the phrasing changes (e.g., 'show' vs 'list'), if it's semantically 
        similar to an existing question, return the cached SQL.
        """
        # Threshold for semantic matching (0.0 means identical, higher means more permissive)
        # Note: Vanna/ChromaDB typically returns results sorted by distance if exposed.
        # Since we use get_similar_question_sql, it returns a list of results.
        
        results = self.get_similar_question_sql(question)
        if not results:
            return None, None
        
        # We manually check the top result. 
        # For a truly robust system, we would check the 'distance' if the API provided it.
        # Given we can't see the raw similarity score easily through this Vanna wrapper,
        # we do a slightly smarter check for key tokens.
        
        best_match = results[0]
        match_q = best_match['question'].lower()
        query_q = question.lower()
        
        # Normalize: Remove common SQL-NL "fluff" words to check for core intent
        fluff = ["show", "list", "get", "me", "all", "please", "can", "you", "tell", "following"]
        def normalize(text):
            for word in fluff:
                text = text.replace(f" {word} ", " ")
                if text.startswith(word + " "): text = text[len(word)+1:]
            return " ".join(text.split())

        if normalize(match_q) == normalize(query_q):
             print(f"[Semantic Cache Hit] Found similar query: '{best_match['question']}'")
             print(f"SQL: {best_match['sql']}")
             try:
                 df = self.run_sql(best_match['sql'])
                 return best_match['sql'], df
             except:
                 return None, None
        
        return None, None

    def train_structured_schema(self, table_name, schema_df, relationships=None):
        """
        Embeds a structured version of the schema for better retrieval accuracy.
        """
        columns = schema_df.to_dict(orient='records')
        structured_doc = {
            "table_name": table_name,
            "columns": columns,
            "relationships": relationships or []
        }
        # Store as a formatted JSON document string to preserve structure
        self.add_documentation(json.dumps(structured_doc))

def get_vanna_instance():
    config = {
        'model': os.getenv("LLM_MODEL", "deepseek-coder:6.7b"),
        'path': os.getenv("VEC_STORAGE_PATH", "./vanna_storage")
    }
    return MyVanna(config=config)

def connect_database(vn_instance):
    global _engine
    db_type = os.getenv("DB_TYPE", "postgres").lower()
    
    if db_type == "postgres":
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASS")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", 5432)
        dbname = os.getenv("DB_NAME")
        
        # Initialize SQLAlchemy engine with connection pooling
        if _engine is None:
            connection_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
            _engine = create_engine(
                connection_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30
            )

        vn_instance.connect_to_postgres(
            host=host,
            dbname=dbname,
            user=user,
            password=password,
            port=int(port)
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
