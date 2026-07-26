"""CLI entry point for baby-llm-nanny.

Usage:
    baby-llm-nanny [OPTIONS] [MODEL]

Options:
    --compare M1,M2,M3    Compare multiple models side-by-side
    --categories CATS    Comma-separated categories to run (default: all)
    --difficulty DIFF    Filter by difficulty: easy, medium, hard
    --verbose            Show full details for all prompts (not just failures)
    --json PATH          Save JSON report to file
    --csv PATH           Save CSV report to file
    --html PATH          Save self-contained HTML report to file
    --temperature FLOAT   Temperature for generation (default: 0.0)
    --seed INT           Random seed (default: 42)
    --host HOST          Ollama host (default: localhost)
    --port PORT          Ollama port (default: 11434)
    --timeout SECONDS    Per-prompt timeout (default: 120)
    --system-prompt NAME System prompt strategy (none, careful, expert, concise)
    --retries N          Run each prompt N times for consistency analysis
    --save-history       Save results to SQLite DB for trend tracking
    --history            Show historical trend report and exit
    --list-models        List available models and exit
    --list-prompts       List all prompts and exit
    --no-color           Disable ANSI color output
    --version            Show version and exit

Live Code Review:
    --review              Enable live code review loop (coding prompts only)
    --max-iterations N    Max generate→test→fix iterations (default: 3)
    --review-prompt TEXT  Custom coding prompt to review (bypasses prompt bank)
    --review-tests-file F Path to a .py file with test_cases list and function_name
    --review-json PATH    Save review results as JSON
"""

from __future__ import annotations

import sys
import argparse
from typing import Optional

from . import __version__
from .prompts import PROMPTS, get_prompts_by_category, get_prompts_by_difficulty, list_categories
from .runner import check_ollama, list_models, query_model, run_prompt_set
from .evaluator import evaluate, EvalResult
from .report import (
    build_results, format_terminal_report, format_comparison_table,
    save_json_report, save_csv_report, save_html_report, set_color_enabled,
)
from .history import (
    init_db, save_run_to_db, format_trend_report, list_runs,
    save_retry_results, get_retry_stats,
)
from .reviewer import (
    review_code, review_coding_prompts, format_review_report,
    save_review_json, ReviewResult,
)


# ─────────────────────────────────────────────────────────────────────
# System prompts — strategies to improve small model output
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "none": None,
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
        "--compare", default=None,
        help="Comma-separated models to compare side-by-side (e.g. qwen2.5:3b,gemma2:2b)",
    )
    parser.add_argument(
        "--categories", default="all",
        help="Comma-separated categories to run (default: all)",
    )
    parser.add_argument(
        "--difficulty", default=None,
        help="Filter by difficulty: easy, medium, hard",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show full details for all prompts",
    )
    out_group = parser.add_argument_group("output formats")
    out_group.add_argument("--json", default=None, help="Save JSON report to file")
    out_group.add_argument("--csv", default=None, help="Save CSV report to file")
    out_group.add_argument("--html", default=None, help="Save HTML report to file")
    out_group.add_argument("--no-color", action="store_true", help="Disable ANSI colors")

    run_group = parser.add_argument_group("model config")
    run_group.add_argument("--temperature", type=float, default=0.0, help="Temperature (default: 0.0)")
    run_group.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    run_group.add_argument("--host", default="localhost", help="Ollama host")
    run_group.add_argument("--port", type=int, default=11434, help="Ollama port")
    run_group.add_argument("--timeout", type=int, default=120, help="Per-prompt timeout")
    run_group.add_argument("--system-prompt", default="none",
                           help="System prompt (none, careful, expert, concise)")

    adv_group = parser.add_argument_group("advanced")
    adv_group.add_argument("--retries", type=int, default=1,
                           help="Run each prompt N times for consistency analysis (default: 1)")
    adv_group.add_argument("--save-history", action="store_true",
                           help="Save results to SQLite DB for trend tracking")
    adv_group.add_argument("--history", action="store_true",
                           help="Show historical trend report and exit")

    review_group = parser.add_argument_group("live code review")
    review_group.add_argument("--review", action="store_true",
                               help="Enable live code review loop (coding prompts only)")
    review_group.add_argument("--max-iterations", type=int, default=3,
                              help="Max generate→test→fix iterations (default: 3)")
    review_group.add_argument("--review-prompt", default=None,
                              help="Custom coding prompt to review (bypasses prompt bank)")
    review_group.add_argument("--review-tests-file", default=None,
                              help="Path to .py file with test_cases and function_name")
    review_group.add_argument("--review-json", default=None,
                              help="Save review results as JSON")

    info_group = parser.add_argument_group("info")
    info_group.add_argument("--list-models", action="store_true", help="List available models")
    info_group.add_argument("--list-prompts", action="store_true", help="List all prompts")
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
            diff_tag = f" [{p.difficulty}]" if p.difficulty != "medium" else ""
            print(f"    {p.id}{diff_tag}")
    return 0


def select_prompts(args) -> list:
    """Select prompts based on --categories and --difficulty filters."""
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

    # Filter by difficulty
    if args.difficulty:
        prompts_to_run = [p for p in prompts_to_run if p.difficulty == args.difficulty]

    return prompts_to_run


def run_single_model(model, prompts_to_run, args, sp_name, system_prompt):
    """Run evaluation for a single model. Returns RunReport."""
    print(f"   Running {model}...")
    responses = run_prompt_set(
        model, prompts_to_run,
        host=args.host, port=args.port, timeout=args.timeout,
        temperature=args.temperature, seed=args.seed,
        show_progress=True,
    )

    # Evaluate
    evaluations = []
    for resp, tp in zip(responses, prompts_to_run):
        if resp.error:
            evaluations.append(EvalResult(
                passed=False, score=0.0,
                detail=f"Model error: {resp.error}"
            ))
        else:
            evaluations.append(evaluate(resp.response, tp))

    # Build report
    report = build_results(
        model=model,
        prompts=prompts_to_run,
        responses=responses,
        evaluations=evaluations,
        system_prompt_name=sp_name,
        temperature=args.temperature,
        seed=args.seed,
    )

    # Retry/consistency analysis
    if args.retries > 1:
        print(f"\n   Running {args.retries} retries for consistency analysis...")
        run_id = None
        if args.save_history:
            run_id = save_run_to_db(report)

        for tp in prompts_to_run:
            retry_results = []
            for attempt in range(1, args.retries + 1):
                # Use different seeds for retries
                resp = query_model(
                    model, tp.prompt,
                    host=args.host, port=args.port, timeout=args.timeout,
                    temperature=args.temperature, seed=args.seed + attempt,
                )
                if resp.error:
                    eval_res = EvalResult(passed=False, score=0.0, detail=resp.error)
                else:
                    eval_res = evaluate(resp.response, tp)
                retry_results.append({
                    "attempt": attempt,
                    "passed": eval_res.passed,
                    "score": eval_res.score,
                    "response": resp.response,
                })

            if run_id:
                save_retry_results(run_id, tp.id, retry_results, db_path=None)

            # Report consistency
            passes = sum(1 for r in retry_results if r["passed"])
            scores = [r["score"] for r in retry_results]
            if len(set(scores)) > 1:
                print(f"   ⚠️  {tp.id}: inconsistent across retries "
                      f"(pass {passes}/{len(retry_results)}, "
                      f"score range {min(scores):.1f}-{max(scores):.1f})")

    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)

    # Handle colors
    if args.no_color:
        set_color_enabled(False)

    # Info-only commands
    if args.list_models:
        return cmd_list_models(args)
    if args.list_prompts:
        return cmd_list_prompts(args)

    # History report
    if args.history:
        model_filter = args.model if args.model != "qwen2.5:3b" else None
        print(format_trend_report(model=model_filter))
        return 0

    # Select prompts
    prompts_to_run = select_prompts(args)
    if not prompts_to_run:
        print("No prompts to run. Check --categories / --difficulty.")
        return 1

    # Check Ollama is running
    if not check_ollama(args.host, args.port):
        print(f"❌ Cannot connect to Ollama at {args.host}:{args.port}")
        print("   Start it with: ollama serve")
        return 1

    system_prompt = SYSTEM_PROMPTS.get(args.system_prompt, None)
    sp_name = args.system_prompt if system_prompt else "none"

    # ──────────────────────────────────────────────────────────────
    # Live Code Review mode
    # ──────────────────────────────────────────────────────────────
    if args.review or args.review_prompt:
        from .reviewer import review_code, review_coding_prompts, format_review_report, save_review_json

        print(f"🍼 baby-llm-nanny v{__version__} — 🔬 Live Code Review")
        print(f"   Model:          {args.model}")
        print(f"   Max iterations: {args.max_iterations}")
        print(f"   Temperature:    {args.temperature}")
        print()

        review_results = []

        if args.review_prompt:
            # Custom single-prompt review mode
            # Load tests from file if provided, otherwise run without tests (just execute)
            function_name = "solution"
            test_cases = []
            code_pattern = ""

            if args.review_tests_file:
                import importlib.util
                spec = importlib.util.spec_from_file_location("tests_module", args.review_tests_file)
                tests_mod = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(tests_mod)
                    test_cases = getattr(tests_mod, "test_cases", [])
                    function_name = getattr(tests_mod, "function_name", "solution")
                    code_pattern = getattr(tests_mod, "code_pattern", "")
                except Exception as e:
                    print(f"❌ Could not load tests file: {e}")
                    return 1

            rr = review_code(
                model=args.model,
                prompt=args.review_prompt,
                function_name=function_name,
                test_cases=test_cases,
                code_pattern=code_pattern,
                prompt_id="custom",
                host=args.host, port=args.port, timeout=args.timeout,
                temperature=args.temperature, seed=args.seed,
                max_iterations=args.max_iterations,
                show_progress=True,
            )
            review_results.append(rr)
        else:
            # Review all coding prompts from the prompt bank
            coding_prompts = get_prompts_by_category("coding")
            if not coding_prompts:
                print("No coding prompts found.")
                return 1
            print(f"   Reviewing {len(coding_prompts)} coding prompts...\n")
            review_results = review_coding_prompts(
                args.model, coding_prompts,
                host=args.host, port=args.port, timeout=args.timeout,
                temperature=args.temperature, seed=args.seed,
                max_iterations=args.max_iterations,
                show_progress=True,
            )

        # Print review report
        print(format_review_report(review_results, args.model))

        # Save review JSON
        if args.review_json:
            path = save_review_json(review_results, args.model, args.review_json)
            print(f"Review JSON saved to: {path}")

        return 0

    # ──────────────────────────────────────────────────────────────
    # Multi-model comparison mode
    # ──────────────────────────────────────────────────────────────
    if args.compare:
        models = [m.strip() for m in args.compare.split(",")]
        reports = []

        for model in models:
            print(f"\n{'─' * 64}")
            print(f"  🍼 Evaluating {model}")
            print(f"{'─' * 64}")
            report = run_single_model(model, prompts_to_run, args, sp_name, system_prompt)
            reports.append(report)

            # Save individual report if --json specified
            if args.json:
                base, ext = args.json.rsplit(".", 1) if "." in args.json else (args.json, "json")
                path = f"{base}_{model.replace(':', '_')}.{ext}"
                save_json_report(report, path)
                print(f"  JSON saved to: {path}")

            if args.save_history:
                save_run_to_db(report)

        # Print comparison table
        print(format_comparison_table(reports))
        return 0

    # Single model mode
    print(f"🍼 baby-llm-nanny v{__version__}")
    print(f"   Model:          {args.model}")
    print(f"   Prompts:        {len(prompts_to_run)}")
    print(f"   Categories:     {', '.join(sorted(set(p.category for p in prompts_to_run)))}")
    if args.difficulty:
        print(f"   Difficulty:     {args.difficulty}")
    print(f"   System prompt:  {sp_name}")
    print(f"   Temperature:    {args.temperature}")
    print(f"   Seed:           {args.seed}")
    if args.retries > 1:
        print(f"   Retries:        {args.retries}")
    print()

    report = run_single_model(args.model, prompts_to_run, args, sp_name, system_prompt)

    # Print terminal report
    print(format_terminal_report(report, verbose=args.verbose))

    # Save outputs
    if args.json:
        path = save_json_report(report, args.json)
        print(f"JSON report saved to: {path}")
    if args.csv:
        path = save_csv_report(report, args.csv)
        print(f"CSV report saved to: {path}")
    if args.html:
        path = save_html_report(report, args.html)
        print(f"HTML report saved to: {path}")
    if args.save_history:
        run_id = save_run_to_db(report)
        print(f"History saved (run ID: {run_id})")

    return 0


if __name__ == "__main__":
    sys.exit(main())