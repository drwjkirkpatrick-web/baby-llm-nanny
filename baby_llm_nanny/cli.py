"""CLI entry point for baby-llm-nanny.

Usage:
    baby-llm-nanny [OPTIONS] [MODEL]

Options:
    --model MODEL         Model to test (default: qwen2.5:3b)
    --categories CATS    Comma-separated categories to run (default: all)
    --verbose            Show full details for all prompts (not just failures)
    --json PATH          Save JSON report to file
    --temperature FLOAT   Temperature for generation (default: 0.0)
    --seed INT           Random seed (default: 42)
    --host HOST          Ollama host (default: localhost)
    --port PORT          Ollama port (default: 11434)
    --timeout SECONDS    Per-prompt timeout (default: 120)
    --list-models        List available models and exit
    --list-prompts       List all prompts and exit
    --system-prompt NAME System prompt strategy (none, careful, expert)
    --version            Show version and exit
"""

from __future__ import annotations

import sys
import argparse
from typing import Optional

from . import __version__
from .prompts import PROMPTS, get_prompts_by_category, list_categories
from .runner import check_ollama, list_models, run_prompt_set
from .evaluator import evaluate
from .report import build_results, format_terminal_report, save_json_report


# ─────────────────────────────────────────────────────────────────────
# System prompts — strategies to improve small model output
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "none": None,  # No system prompt — baseline
    "careful": (
        "You are a careful assistant. Think step by step. "
        "If you are not sure about something, say 'I don't know'. "
        "Do not make up facts. Be precise with numbers."
    ),
    "expert": (
        "You are an expert reasoning assistant. "
        "For math problems, work through each step carefully before giving the final answer. "
        "For factual questions, only answer if you are confident. "
        "If a question references something you don't recognize, it may not exist — say so. "
        "For coding, write clean, correct Python. Follow instructions exactly."
    ),
    "concise": (
        "Answer concisely. Follow output format instructions exactly. "
        "If asked for just a number, output only the number. "
        "If asked for just a word, output only that word."
    ),
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="baby-llm-nanny",
        description="🍼 Hallucination and quality screening for small local LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model", nargs="?", default="qwen2.5:3b",
        help="Model to test (default: qwen2.5:3b)",
    )
    parser.add_argument(
        "--categories", default="all",
        help="Comma-separated categories to run (default: all)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show full details for all prompts",
    )
    json_group = parser.add_argument_group("output")
    json_group.add_argument(
        "--json", default=None,
        help="Save JSON report to file path",
    )
    run_group = parser.add_argument_group("model config")
    run_group.add_argument(
        "--temperature", type=float, default=0.0,
        help="Temperature for generation (default: 0.0)",
    )
    run_group.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    run_group.add_argument("--host", default="localhost", help="Ollama host (default: localhost)")
    run_group.add_argument("--port", type=int, default=11434, help="Ollama port (default: 11434)")
    run_group.add_argument("--timeout", type=int, default=120, help="Per-prompt timeout (default: 120)")
    run_group.add_argument(
        "--system-prompt", default="none",
        help="System prompt strategy (none, careful, expert, concise)",
    )
    info_group = parser.add_argument_group("info")
    info_group.add_argument("--list-models", action="store_true", help="List available models and exit")
    info_group.add_argument("--list-prompts", action="store_true", help="List all prompts and exit")
    info_group.add_argument("--version", action="version", version=f"baby-llm-nanny {__version__}")
    return parser


def cmd_list_models(args) -> int:
    models = list_models(args.host, args.port)
    if not models:
        print("No models found (is Ollama running?)")
        return 1
    print("Available models:")
    for m in models:
        print(f"  • {m}")
    return 0


def cmd_list_prompts(args) -> int:
    cats = list_categories()
    print(f"Total prompts: {len(PROMPTS)}")
    print(f"Categories: {', '.join(cats)}")
    print()
    for cat in cats:
        cat_prompts = get_prompts_by_category(cat)
        print(f"  {cat} ({len(cat_prompts)}):")
        for p in cat_prompts:
            print(f"    {p.id}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    # Info-only commands
    if args.list_models:
        return cmd_list_models(args)
    if args.list_prompts:
        return cmd_list_prompts(args)

    # Select prompts
    if args.categories == "all":
        prompts_to_run = list(PROMPTS)
    else:
        cats = [c.strip() for c in args.categories.split(",")]
        prompts_to_run = []
        for c in cats:
            cat_prompts = get_prompts_by_category(c)
            if not cat_prompts:
                print(f"Warning: unknown category '{c}'")
            prompts_to_run.extend(cat_prompts)

    if not prompts_to_run:
        print("No prompts to run. Check --categories.")
        return 1

    # Check Ollama is running
    if not check_ollama(args.host, args.port):
        print(f"❌ Cannot connect to Ollama at {args.host}:{args.port}")
        print("   Start it with: ollama serve")
        return 1

    system_prompt = SYSTEM_PROMPTS.get(args.system_prompt, None)
    sp_name = args.system_prompt if system_prompt else "none"

    print(f"🍼 baby-llm-nanny v{__version__}")
    print(f"   Model:          {args.model}")
    print(f"   Prompts:        {len(prompts_to_run)}")
    print(f"   Categories:     {', '.join(sorted(set(p.category for p in prompts_to_run)))}")
    print(f"   System prompt:  {sp_name}")
    print(f"   Temperature:    {args.temperature}")
    print(f"   Seed:           {args.seed}")
    print()

    # Run prompts
    print("Running prompts...")
    responses = run_prompt_set(
        args.model, prompts_to_run,
        host=args.host, port=args.port, timeout=args.timeout,
        temperature=args.temperature, seed=args.seed,
        show_progress=True,
    )

    # Evaluate
    print("\nEvaluating...")
    evaluations = []
    for resp, tp in zip(responses, prompts_to_run):
        if resp.error:
            from .evaluator import EvalResult
            evaluations.append(EvalResult(
                passed=False, score=0.0,
                detail=f"Model error: {resp.error}"
            ))
        else:
            evaluations.append(evaluate(resp.response, tp))

    # Build report
    report = build_results(
        model=args.model,
        prompts=prompts_to_run,
        responses=responses,
        evaluations=evaluations,
        system_prompt_name=sp_name,
        temperature=args.temperature,
        seed=args.seed,
    )

    # Print terminal report
    print(format_terminal_report(report, verbose=args.verbose))

    # Save JSON if requested
    if args.json:
        path = save_json_report(report, args.json)
        print(f"JSON report saved to: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())