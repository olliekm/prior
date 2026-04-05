#!/usr/bin/env python3
"""
Prior - Research Literature Analysis Tool

Usage:
    python main.py "your research question here"
    python main.py --interactive
    python main.py --init-db
"""
import argparse
import json
import sys
import time
from db.vector import init_db
from core.graph import build_graph


def run_analysis(question: str, verbose: bool = True) -> dict:
    """Run the full analysis pipeline on a research question."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Research Question: {question}")
        print(f"{'='*60}\n")

    start = time.time()
    graph = build_graph()

    initial_state = {
        "question": question,
        "sub_queries": [],
        "papers": [],
        "claims": [],
        "report": None,
    }

    result = graph.invoke(initial_state)
    elapsed = time.time() - start

    if verbose:
        print(f"\n{'='*60}")
        print(f"Analysis completed in {elapsed:.1f}s")
        print(f"{'='*60}\n")

    return result


def print_report(result: dict):
    """Pretty-print the analysis report."""
    if not result.get("report"):
        print("No report generated.")
        return

    try:
        report = json.loads(result["report"])
        print(json.dumps(report, indent=2))
    except json.JSONDecodeError:
        print(result["report"])


def interactive_mode():
    """Run in interactive REPL mode."""
    print("\nPrior - Interactive Mode")
    print("Type your research question and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        result = run_analysis(question)
        print_report(result)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Prior - Research Literature Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py "What are the latest advances in neural architecture search?"
    python main.py --interactive
    python main.py --init-db
        """,
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Research question to analyze",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize the database and exit",
    )
    parser.add_argument(
        "-o", "--output",
        help="Write JSON report to file",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    if args.init_db:
        print("Initializing database...")
        init_db()
        print("Database initialized.")
        return

    if args.interactive:
        interactive_mode()
        return

    if not args.question:
        parser.print_help()
        sys.exit(1)

    result = run_analysis(args.question, verbose=not args.quiet)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Report written to {args.output}")
    else:
        print_report(result)


if __name__ == "__main__":
    main()
