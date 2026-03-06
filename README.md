# FineTune-BI-LLM

A project to fine-tune and implement a Business Intelligence (BI) tool using Vanna.ai, Ollama, and local PostgreSQL to generate SQL from natural language.

## Architecture

- **LLM**: `deepseek-coder:6.7b` running locally via **Ollama**.
- **Vector Store**: **ChromaDB** for storing training data (DDL, SQL examples, documentation).
- **Framework**: **Vanna.ai** (Legacy Implementation for local integration).
- **Database**: **PostgreSQL** (Postgres Air dataset).

## Features

- **Automated Schema Training**: Extracts table schemas from PostgreSQL and trains the Vanna model.
- **Local Inference**: Uses Ollama for generating SQL queries without external API calls.
- **Persistent Storage**: Stores training embeddings locally in `./vanna_storage`.

## Project Structure

- `app.py`: Main application entry point for querying.
- `train.py`: Training script to ingest database schema and documentation.
- `requirements.txt`: Project dependencies.
- `vanna_storage/`: Local ChromaDB vector database.

## Setup Instructions

1.  **Install dependencies**:
    ```bash
    pip install vanna ollama chromadb psycopg2-binary
    ```
2.  **Ensure Ollama is running**:
    ```bash
    ollama serve
    ollama pull deepseek-coder:6.7b
    ```
3.  **Run Training**:
    ```bash
    python train.py
    ```
4.  **Run Application**:
    ```bash
    python app.py
    ```
