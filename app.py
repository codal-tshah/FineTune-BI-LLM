"""
Main Entry Point: Integrated AgenticSQLPipeline

This module provides a unified interface for:
1. Multi-Agent Orchestration (Planner → SQL → Validator)
2. Self-Learning (Automatic training on successful queries)
3. Schema Awareness (Direct database metadata to prevent hallucination)
4. Connection Pooling (High-performance PostgreSQL connections)
"""

import argparse
from typing import List

from agent_pipeline import AgenticSQLPipeline

def ask_question(question: str):
    """
    Main interface: Ask a natural language question, get results.
    
    Uses the 3-Agent Pipeline:
    - Planner Agent: Creates a step-by-step SQL plan
    - SQL Agent: Generates the actual SQL query
    - Validator Agent: Executes, validates, and auto-trains on success
    
    Args:
        question: Natural language query (e.g., "Show me top 5 airports")
    
    Returns:
        pandas.DataFrame with query results, or None if error
    """
    agent = AgenticSQLPipeline()
    results = agent.run(question)
    return results

DEFAULT_TEST_QUESTIONS: List[str] = [
    "Show me the top 5 airports by name in the US",
    "How many flights are scheduled from each airport?",
    "What is the most common aircraft used?"
]


def _print_results(result):
    if result is None:
        print("🛑 No results returned")
        return

    try:
        print(result)
    except Exception:
        print(str(result))


def interactive_loop():
    print("💬 Enter a question (type 'quit' or 'exit' to stop)")
    while True:
        raw = input("Question: ").strip()
        if raw.lower() in {"quit", "exit"}:
            print("👋 Bye!")
            break

        if not raw:
            continue

        _print_results(ask_question(raw))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Agentic SQL pipeline for answers")
    parser.add_argument("-q", "--question", help="One-off question to run")
    parser.add_argument("-t", "--test", action="store_true", help="Run the built-in test questions")
    args = parser.parse_args()

    if args.question:
        print(f"📝 Question: {args.question}")
        _print_results(ask_question(args.question))
    elif args.test:
        print("🧪 Running test questions...\n")
        for qm in DEFAULT_TEST_QUESTIONS:
            print(f"{'='*60}\nQuestion: {qm}\n{'='*60}")
            _print_results(ask_question(qm))
            print()
    else:
        interactive_loop()


if __name__ == "__main__":
    main()