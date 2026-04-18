#!/usr/bin/env python3
"""
Ahoum CLI — score a conversation turn from the command line.

Usage:
  # Score a single turn against all facets
  python src/cli.py score --speaker User --text "I feel hopeless and can't get out of bed."

  # Score against a specific category only
  python src/cli.py score --speaker User --text "..." --category emotion

  # Score against N random facets (quick test)
  python src/cli.py score --speaker User --text "..." --n 10

  # List available facets
  python src/cli.py facets
  python src/cli.py facets --category safety

  # Test Ollama connection
  python src/cli.py ping
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from scorer import score_turn, OLLAMA_MODEL, OLLAMA_BASE_URL, BATCH_SIZE

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'facets_cleaned.csv')


def load_facets(category: str = None, n: int = None, observable_only: bool = True) -> list[dict]:
    rows = []
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if observable_only and row.get("observable_in_text", "True") == "False":
                continue
            if category and row["category"] != category:
                continue
            rows.append({
                "facet_id": int(row["facet_id"]),
                "facet_name": row["facet_name"],
                "category": row["category"],
            })
    if n:
        import random
        rows = random.sample(rows, min(n, len(rows)))
    return rows


def cmd_ping(_args):
    """Test that Ollama is reachable and the model responds."""
    from openai import OpenAI
    print(f"Connecting to: {OLLAMA_BASE_URL}")
    print(f"Model: {OLLAMA_MODEL}")
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    try:
        resp = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": 'Reply with exactly: {"status":"ok"}'}],
            max_tokens=20,
            temperature=0,
        )
        print(f"Response: {resp.choices[0].message.content}")
        print("✅ Ollama connection OK")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)


def cmd_facets(args):
    """List available facets."""
    facets = load_facets(category=args.category)
    categories = {}
    for f in facets:
        categories.setdefault(f["category"], []).append(f["facet_name"])

    for cat, names in sorted(categories.items()):
        print(f"\n── {cat.upper()} ({len(names)}) ──")
        for name in names:
            print(f"  {name}")
    print(f"\nTotal: {len(facets)} facets")


def cmd_score(args):
    """Score a turn and print results."""
    facets = load_facets(category=args.category, n=args.n)
    if not facets:
        print("No facets matched. Check --category value.")
        sys.exit(1)

    model = args.model or OLLAMA_MODEL
    batch_size = args.batch_size or BATCH_SIZE

    turn_text = f"{args.speaker}: {args.text}"
    print(f"\nScoring: \"{turn_text[:80]}...\"")
    print(f"Model: {model}  |  Facets: {len(facets)}  |  Batch size: {batch_size}")
    print(f"Estimated LLM calls: {-(-len(facets) // batch_size)}\n")  # ceiling div

    scores = score_turn(turn_text, facets, batch_size=batch_size, model=model)

    # Sort by score descending
    scores.sort(key=lambda s: (-s["score"], -s["confidence"]))

    # Print table
    print(f"{'Facet':<35} {'Score':>5}  {'Conf':>5}  Rationale")
    print("─" * 90)
    for s in scores:
        bar = "█" * s["score"] + "░" * (5 - s["score"])
        print(f"{s['facet_name']:<35} {bar}  {s['confidence']:>4.0%}  {s['rationale'][:50]}")

    # Summary stats
    avg = sum(s["score"] for s in scores) / len(scores)
    avg_conf = sum(s["confidence"] for s in scores) / len(scores)
    print(f"\nAvg score: {avg:.2f}/5  |  Avg confidence: {avg_conf:.0%}  |  Facets scored: {len(scores)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"turn": turn_text, "scores": scores}, f, indent=2)
        print(f"\nSaved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Ahoum Conversation Evaluation CLI")
    sub = parser.add_subparsers(dest="cmd")

    # ping
    sub.add_parser("ping", help="Test Ollama connection")

    # facets
    f_parser = sub.add_parser("facets", help="List available facets")
    f_parser.add_argument("--category", help="Filter by category (emotion, safety, cognitive, ...)")

    # score
    s_parser = sub.add_parser("score", help="Score a conversation turn")
    s_parser.add_argument("--speaker", default="User", help="Speaker name")
    s_parser.add_argument("--text", required=True, help="Turn text to score")
    s_parser.add_argument("--category", help="Score only this facet category")
    s_parser.add_argument("--n", type=int, help="Score N random facets (quick test)")
    s_parser.add_argument("--model", help=f"Ollama model (default: {OLLAMA_MODEL})")
    s_parser.add_argument("--batch-size", type=int, dest="batch_size",
                          help=f"Facets per LLM call (default: {BATCH_SIZE})")
    s_parser.add_argument("--output", "-o", help="Save results to JSON file")

    args = parser.parse_args()

    if args.cmd == "ping":
        cmd_ping(args)
    elif args.cmd == "facets":
        cmd_facets(args)
    elif args.cmd == "score":
        cmd_score(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
