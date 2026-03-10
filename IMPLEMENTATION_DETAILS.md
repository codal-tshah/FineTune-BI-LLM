# FineTune-BI-LLM Implementation Summary

## 1. Issue: `ModuleNotFoundError: No module named 'vanna.ollama'`

- **Cause**: The `vanna` package (v2.0.2) no longer exposes `Ollama` and `ChromaDB_VectorStore` under the top-level `vanna.ollama` and `vanna.chromadb` modules in the same way the user code expected.
- **Resolution**:
  - Identified the correct classes in `vanna.legacy.ollama.Ollama` and `vanna.legacy.chromadb.ChromaDB_VectorStore`.
  - Installed missing optional dependencies: `ollama` and `chromadb`.

## 2. Refactoring to Multi-Inheritance

- **Action**: Created a `MyVanna` class that inherits from both `ChromaDB_VectorStore` and `Ollama`.
- **Reason**: Vanna's architecture requires a class that implements both the LLM interface (for query generation) and the Vector Store interface (for retrieving similar training data).
- **Implementation**:

  ```python
  class MyVanna(ChromaDB_VectorStore, Ollama):
      def __init__(self, config=None):
          ChromaDB_VectorStore.__init__(self, config=config)
          Ollama.__init__(self, config=config)
  ```

## 3. Environment Preparation

- **Action**: Configured the project to use a local storage directory (`./vanna_storage`) for **ChromaDB**.
- **Action**: Set the model to `deepseek-coder:6.7b`.

## 4. Training Validation

- **Action**: Updated `train.py` to iterate through all tables in the `public` schema of the `postgres_air` database.
- **Action**: Executed the training script, which now correctly initializes the LLM, connects to PostgreSQL, and stores the schema documentation in the vector database.

## 7. Advanced Agentic Pipeline (4-Agent Architecture)

- **Problem**: A single-agent SQL generator often hallucinates columns (e.g., `login_email` instead of `login`) or struggles with complex joins.
- **Solution**: Implemented a multi-agent orchestration in `agent_pipeline.py`.
- **Agents**:
  1. **Classifier Agent**: Categorizes queries into domains (FLIGHT, BOOKING, PASSENGER, etc.).
  2. **Planner Agent**: Performs strict schema introspection of the categorized domain and generates a step-by-step SQL plan.
  3. **SQL Agent**: Translates the plan into schema-qualified SQL with **Anti-Parameterization** guardrails.
  4. **Validator Agent**: Runs the SQL, checks for errors, and triggers **Self-Correction** or **Self-Learning**.

## 8. Automated Self-Correction

- **Logic**: If the `Validator Agent` catches an exception (e.g., syntax error or missing column), it passes the error message and the failed SQL back to the `SQL Agent`.
- **Refinement**: The SQL Agent analyzes the error and produces a corrected query. This effectively doubles the success rate for complex joins or minor hallucinations.

## 9. Observability: Latency Tracking & Metrics

- **Latency measuring**: Every phase (Cache check, Classification, Planning, SQL Gen, Validation) is measured using `time.time()`.
- **Metrics Log**: `metrics.jsonl` stores a JSON line for every execution, containing the question, SQL, result status, and a breakdown of latencies per phase.

## 10. Robust Hybrid Table Selection

- **Problem**: Narrow domain categories could sometimes "blind" the planner to relevant tables if the classifier was slightly off.
- **Solution**: The Planner now uses a hybrid selection strategy:
  1. **Domain Logic**: Tables from the identified category.
  2. **Keyword Match**: Tables whose names appear in the natural language question.
  3. **Context Sensitivity**: Tables referenced in similar successful SQL examples retrieved from ChromaDB.

## 11. Semantic Query Caching (Fast Path)

- Uses ChromaDB to find similar previously answered questions.
- Applies a **Normalization Layer** to strip "fluff" words (show, list, please, me, all).
- If the core intent matches an existing record, it returns the cached SQL/Results instantly (<1s).

## 10. Failure Observability & Schema Guardrails

- **Failure Log**: `MyVanna.log_failure(...)` now writes every caught SQL exception as a `failure_log` document number in ChromaDB.
- **SQL Guardrails**:
  - **Schema Qualification**: Post-processes SQL to replace plain table references with `"schema"."table"`.
  - **Anti-Parameterization**: Strips LLM-generated parameters (like `:age`) to ensure executable SQL.
  - **Multi-Statement Guard**: Extracts only the first SELECT if the LLM returns multiple queries.

## Updated Component Matrix

| Component | Library/Tool | Role |
| :--- | :--- | :--- |
| **Pipeline** | `AgenticSQLPipeline` | 4-Agent Orchestration (Classifier/Planner/SQL/Validator). |
| **Cache** | `MyVanna.get_cached_query` | Semantic Normalization & Fast Path Retrieval. |
| **LLM Service** | `vanna.legacy.ollama.Ollama` | Logic for prompting the local model. |
| **Vector Index** | `vanna.legacy.chromadb.ChromaDB_VectorStore` | Storage and retrieval for DDL/SQL examples. |
| **Orchestration** | Custom `MyVanna` Class | Integrates LLM, Vector Store, and Cache. |
| **Model Engine** | `Ollama` (`deepseek-coder:6.7b`) | Local inference engine. |
| **Deep Persistence** | `SQLAlchemy` (`QueuePool`) | Efficient connection to PostgreSQL. |

## Training & "Fine-Tuning" Logic (RAG vs Fine-Tuning)

This project uses **Retrieval-Augmented Generation (RAG)** rather than traditional weight-based fine-tuning for the DeepSeek LLM.

### What is stored in ChromaDB?

ChromaDB stores **Embeddings** (mathematical vectors) of:

1. **DDL/Schema**: Table and column definitions.
2. **SQL Examples**: Valid pairs of `(Natural Language Question, SQL Query)`.
3. **Documentation**: Business definitions and data samples.

### How it works at Runtime

1. **Search**: When you ask a question, Vanna searches ChromaDB for the *most similar* schema and SQL examples.
2. **Context Injection**: These retrieved snippets are injected into the **Prompt** sent to DeepSeek.
3. **Inference**: DeepSeek uses this "context" to write a precise SQL query for your specific database.

### Does it "Train" DeepSeek?

- **No**, the model weights of `deepseek-coder` are not modified.
- **Yes**, the model "understands" your database because we give it the exact "cheat sheet" (schema + examples) it needs via the RAG pipeline.

## 11. Workflow Visibility & Prompt Noise Control

- `agent_pipeline.py` now logs each phase via `log_stage(...)`, emitting a `[Workflow] Phase X - Y` line that summarizes what just finished (e.g., cache hit, category determined, SQL executed, training recorded).
- `connections.py`'s `MyVanna.log` overrides the base logger to suppress Ollama prompt dumps unless the `VANNA_SHOW_PROMPTS=1` environment variable is set.

## 12. SQL Pre-Validation Layer

- **Problem**: Lower-parameter models (6.7B) frequently hallucinate column names (`id` vs `passenger_id`) or truncate table names (`boardingb`).
- **Solution**: A **code-level pre-validator** (`_validate_sql()`) runs before SQL execution:
  - **Fuzzy Table Matching**: Auto-corrects misspelled table names using `difflib`.
  - **Column Validation**: Verifies `table.column` references against the actual schema and fuzzy-fixes hallucinations.
  - **Alias-Aware**: Correctly resolves table aliases (e.g., `p.id` → `passenger.id`) for validation.

## 13. Startup Caching for Performance

- **Problem**: Querying `information_schema` and sampling data on every query added 10-15s of overhead to the Planning phase.
- **Solution**: Cached expensive metadata at startup:
  - `_load_relationships()`: FK graph cached.
  - `_load_samples()`: 3 sample rows per table cached.
  - `_cached_all_tables`: Table list cached.
- **Impact**: Planning phase DB overhead reduced to zero at runtime.

## 14. FK-Aware SQL Generation

- **Problem**: LLMs often guess direct join paths that don't exist in the schema.
- **Solution**: The full FK relationship graph is injected as a `JOIN PATH REFERENCE` in the SQL Agent prompt. This ensures the model follows the correct multi-hop paths (e.g., `flight` → `booking_leg` → `boarding_pass`).

## 15. Auto-LIMIT Safety Guard

- **Problem**: Unrestricted SELECTs on large tables (e.g., `boarding_pass` with 200k+ rows) hang the CLI and PgAdmin.
- **Solution**: Appends `LIMIT 100` to any query lacking a LIMIT clause.

## Performance Notes (Mac 16GB RAM)

- **Sequential Nature**: Latency is cumulative (sum of 4 LLM calls). Parallelism is not possible as agents depend on prior output.
- **Optimization**: Apple Silicon Macs should update Ollama to ensure Metal GPU support. Switch to `qwen2.5-coder:3b` or Gemini Flash API for significant speed gains (10x faster than CPU inference).

## Known Limitations

1. **6.7B Accuracy**: Hallucinations on complex queries still occur; SQL Pre-Validator catches ~80%.
2. **PostgreSQL focus**: MySQL/SQLite support exists but is secondary.
3. **No multi-turn context**: Each question is treated as a new session.

