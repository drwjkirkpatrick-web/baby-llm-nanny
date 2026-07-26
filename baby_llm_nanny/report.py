"""Report generation — terminal output, JSON/CSV/HTML export, and comparison.

Supports:
  - Colored terminal output (ANSI green/red/yellow)
  - Category + difficulty breakdowns
  - JSON export
  - CSV export (spreadsheet-friendly)
  - HTML report (self-contained file)
  - Multi-model comparison table
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ─────────────────────────────────────────────────────────────────────
# ANSI color codes
# ─────────────────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Detect if we should use colors
_USE_COLOR = True

def _c(text: str, color: str) -> str:
    """Wrap text in color codes if color is enabled."""
    if _USE_COLOR:
        return f"{color}{text}{RESET}"
    return text

def set_color_enabled(enabled: bool) -> None:
    """Enable or disable ANSI color output."""
    global _USE_COLOR
    _USE_COLOR = enabled


# ─────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class PromptResult:
    """Result of a single prompt evaluation."""
    prompt_id: str
    category: str
    difficulty: str
    prompt_text: str
    model_response: str
    expected: Any
    check: str
    passed: bool
    score: float
    detail: str
    extracted: str | None = None
    response_time_sec: float = 0.0
    tokens_per_sec: float | None = None
    eval_count: int | None = None
    total_tokens: int | None = None
    hallucination_confidence: float = 0.0
    notes: str = ""


@dataclass
class CategorySummary:
    """Aggregate results for a category."""
    category: str
    total: int = 0
    passed: int = 0
    partial: int = 0
    failed: int = 0
    avg_score: float = 0.0
    avg_time_sec: float = 0.0
    avg_tokens_per_sec: float | None = None


@dataclass
class DifficultySummary:
    """Aggregate results for a difficulty level."""
    difficulty: str
    total: int = 0
    passed: int = 0
    avg_score: float = 0.0


@dataclass
class RunReport:
    """Complete report for a single model run."""
    model: str
    timestamp: str
    system_prompt_name: str
    temperature: float
    seed: int
    results: list[PromptResult] = field(default_factory=list)
    category_summaries: dict[str, CategorySummary] = field(default_factory=dict)
    difficulty_summaries: dict[str, DifficultySummary] = field(default_factory=dict)

    @property
    def total_prompts(self) -> int:
        return len(self.results)

    @property
    def total_passed(self) -> int:
        return sum(1 for r in self.results if r.score == 1.0)

    @property
    def total_partial(self) -> int:
        return sum(1 for r in self.results if 0 < r.score < 1.0)

    @property
    def total_failed(self) -> int:
        return sum(1 for r in self.results if r.score == 0.0)

    @property
    def overall_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def overall_pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.total_passed / len(self.results)

    @property
    def total_time_sec(self) -> float:
        return sum(r.response_time_sec for r in self.results)

    @property
    def avg_tokens_per_sec(self) -> float | None:
        """Average tokens/sec across all prompts that have the metric."""
        vals = [r.tokens_per_sec for r in self.results if r.tokens_per_sec]
        if vals:
            return sum(vals) / len(vals)
        return None

    @property
    def total_tokens_used(self) -> int:
        """Total tokens consumed across all prompts."""
        return sum(r.total_tokens or 0 for r in self.results)

    @property
    def avg_hallucination_confidence(self) -> float:
        """Average hallucination confidence for hallucination-category prompts."""
        vals = [r.hallucination_confidence for r in self.results
                if r.category == "hallucination" and r.hallucination_confidence > 0]
        if vals:
            return sum(vals) / len(vals)
        return 0.0


# ─────────────────────────────────────────────────────────────────────────
# Build results
# ─────────────────────────────────────────────────────────────────────────

def build_results(
    model: str,
    prompts: list,
    responses: list,
    evaluations: list,
    system_prompt_name: str = "none",
    temperature: float = 0.0,
    seed: int = 42,
) -> RunReport:
    """Build a RunReport from prompts, responses, and evaluations."""
    from .evaluator import hallucination_confidence

    report = RunReport(
        model=model,
        timestamp=datetime.now().isoformat(),
        system_prompt_name=system_prompt_name,
        temperature=temperature,
        seed=seed,
    )

    for tp, resp, eval_res in zip(prompts, responses, evaluations):
        tps = getattr(tp, 'tokens_per_sec', None)
        r_tps = getattr(resp, 'tokens_per_sec', None)
        pr = PromptResult(
            prompt_id=tp.id,
            category=tp.category,
            difficulty=getattr(tp, 'difficulty', 'medium'),
            prompt_text=tp.prompt,
            model_response=resp.response,
            expected=tp.expected,
            check=tp.check,
            passed=eval_res.passed,
            score=eval_res.score,
            detail=eval_res.detail,
            extracted=eval_res.extracted,
            response_time_sec=resp.response_time_sec,
            tokens_per_sec=r_tps,
            eval_count=getattr(resp, 'eval_count', None),
            total_tokens=getattr(resp, 'total_tokens', None),
            hallucination_confidence=hallucination_confidence(resp.response, tp.category),
            notes=tp.notes,
        )
        report.results.append(pr)

    # Build category summaries
    cat_data: dict[str, list[PromptResult]] = defaultdict(list)
    for pr in report.results:
        cat_data[pr.category].append(pr)

    for cat, items in cat_data.items():
        total = len(items)
        passed = sum(1 for i in items if i.score == 1.0)
        partial = sum(1 for i in items if 0 < i.score < 1.0)
        failed = sum(1 for i in items if i.score == 0.0)
        avg_score = sum(i.score for i in items) / total if total else 0.0
        avg_time = sum(i.response_time_sec for i in items) / total if total else 0.0
        tps_vals = [i.tokens_per_sec for i in items if i.tokens_per_sec]
        avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else None
        report.category_summaries[cat] = CategorySummary(
            category=cat, total=total, passed=passed,
            partial=partial, failed=failed,
            avg_score=avg_score, avg_time_sec=avg_time, avg_tokens_per_sec=avg_tps,
        )

    # Build difficulty summaries
    diff_data: dict[str, list[PromptResult]] = defaultdict(list)
    for pr in report.results:
        diff_data[pr.difficulty].append(pr)

    for diff, items in diff_data.items():
        total = len(items)
        passed = sum(1 for i in items if i.score == 1.0)
        avg_score = sum(i.score for i in items) / total if total else 0.0
        report.difficulty_summaries[diff] = DifficultySummary(
            difficulty=diff, total=total,
            passed=passed, avg_score=avg_score,
        )

    return report


# ─────────────────────────────────────────────────────────────────────────
# Terminal report
# ─────────────────────────────────────────────────────────────────────────

def format_terminal_report(report: RunReport, verbose: bool = False) -> str:
    """Format a RunReport as a colored terminal string."""
    lines = []
    lines.append("")
    lines.append("═" * 64)
    lines.append(f"  🍼 baby-llm-nanny — {_c('Report for ' + report.model, BOLD)}")
    lines.append("═" * 64)
    lines.append(f"  Timestamp:     {report.timestamp}")
    lines.append(f"  System prompt: {report.system_prompt_name}")
    lines.append(f"  Temperature:   {report.temperature}")
    lines.append(f"  Seed:          {report.seed}")
    lines.append("")

    # Overall scores
    lines.append("─" * 64)
    lines.append(f"  {_c('OVERALL', BOLD)}")
    lines.append("─" * 64)
    lines.append(f"  Prompts:       {report.total_prompts}")
    pass_str = f"{report.total_passed} / {report.total_prompts} ({report.overall_pass_rate:.1%})"
    lines.append(f"  Passed:        {_c(str(pass_str), GREEN)}")
    lines.append(f"  Partial:       {report.total_partial}")
    lines.append(f"  Failed:        {_c(str(report.total_failed), RED)}")
    lines.append(f"  Avg score:     {report.overall_score:.3f}")
    lines.append(f"  Total time:    {report.total_time_sec:.1f}s")
    if report.avg_tokens_per_sec is not None:
        lines.append(f"  Avg tok/sec:   {report.avg_tokens_per_sec:.1f}")
    if report.total_tokens_used > 0:
        lines.append(f"  Total tokens:  {report.total_tokens_used}")
    if report.avg_hallucination_confidence > 0:
        hc = report.avg_hallucination_confidence
        hc_color = RED if hc > 0.5 else YELLOW if hc > 0.2 else GREEN
        lines.append(f"  Halluc conf:   {_c(f'{hc:.2f}', hc_color)}")
    lines.append("")

    # Category breakdown
    lines.append("─" * 64)
    lines.append(f"  {_c('BY CATEGORY', BOLD)}")
    lines.append("─" * 64)
    header = f"  {'Category':<16} {'Pass':<8} {'Partial':<8} {'Fail':<8} {'Score':<8} {'Time':<8} {'Tok/s':<8}"
    lines.append(header)
    lines.append("  " + "─" * 56)
    for cat in sorted(report.category_summaries.keys()):
        cs = report.category_summaries[cat]
        score_color = GREEN if cs.avg_score >= 0.8 else YELLOW if cs.avg_score >= 0.5 else RED
        tps_str = f"{cs.avg_tokens_per_sec:.1f}" if cs.avg_tokens_per_sec else "-"
        lines.append(
            f"  {cat:<16} {cs.passed:<8} {cs.partial:<8} {cs.failed:<8} "
            f"{_c(f'{cs.avg_score:.3f}', score_color):<8} {cs.avg_time_sec:<8.1f} {tps_str:<8}"
        )
    lines.append("")

    # Difficulty breakdown
    if report.difficulty_summaries:
        lines.append("─" * 64)
        lines.append(f"  {_c('BY DIFFICULTY', BOLD)}")
        lines.append("─" * 64)
        lines.append(f"  {'Difficulty':<16} {'Total':<8} {'Pass':<8} {'Score':<8}")
        lines.append("  " + "─" * 36)
        for diff in ["easy", "medium", "hard"]:
            if diff in report.difficulty_summaries:
                ds = report.difficulty_summaries[diff]
                score_color = GREEN if ds.avg_score >= 0.8 else YELLOW if ds.avg_score >= 0.5 else RED
                lines.append(
                    f"  {diff:<16} {ds.total:<8} {ds.passed:<8} "
                    f"{_c(f'{ds.avg_score:.3f}', score_color):<8}"
                )
        lines.append("")

    # Per-prompt details
    lines.append("─" * 64)
    lines.append(f"  {_c('PER-PROMPT DETAILS', BOLD)}")
    lines.append("─" * 64)
    for pr in report.results:
        if pr.score == 1.0:
            icon = _c("✅", GREEN)
        elif pr.score > 0:
            icon = _c("⚠️", YELLOW)
        else:
            icon = _c("❌", RED)
        diff_tag = f"[{pr.difficulty}]" if pr.difficulty != "medium" else ""
        lines.append(f"  {icon} [{pr.category}] {pr.prompt_id} {diff_tag}")
        if verbose or not pr.passed:
            lines.append(f"     Prompt:    {pr.prompt_text[:120]}")
            lines.append(f"     Response:  {pr.model_response[:200]}")
            lines.append(f"     Expected:  {str(pr.expected)[:120]}")
            lines.append(f"     Detail:    {pr.detail}")
            lines.append(f"     Time:      {pr.response_time_sec:.1f}s")
            if pr.tokens_per_sec:
                lines.append(f"     Tok/s:     {pr.tokens_per_sec:.1f}")
            if pr.hallucination_confidence > 0:
                lines.append(f"     Halluc:    {pr.hallucination_confidence:.1f}")
            if pr.notes:
                lines.append(f"     Notes:     {pr.notes}")
            lines.append("")

    if not verbose:
        lines.append("  (Use --verbose for full details on all prompts)")
        lines.append("")

    lines.append("═" * 64)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Multi-model comparison table
# ─────────────────────────────────────────────────────────────────────────

def format_comparison_table(reports: list[RunReport]) -> str:
    """Format a side-by-side comparison table of multiple model reports."""
    if not reports:
        return "No reports to compare."

    lines = []
    lines.append("")
    lines.append("═" * 72)
    lines.append(f"  🍼 baby-llm-nanny — {_c('Model Comparison', BOLD)}")
    lines.append("═" * 72)

    # Overall comparison
    col_w = 14
    model_names = [r.model for r in reports]
    lines.append("")
    lines.append(f"  {'Metric':<20}" + "".join(f" {name:>{col_w}}" for name in model_names))
    lines.append("  " + "─" * (20 + col_w * len(reports)))

    def fmt_val(vals, fmt="{:.1%}"):
        return "".join(f" {fmt.format(v):>{col_w}}" for v in vals)

    lines.append(f"  {'Prompts':<20}" + "".join(f" {r.total_prompts:>{col_w}}" for r in reports))
    lines.append(f"  {'Passed':<20}" + fmt_val([r.overall_pass_rate for r in reports]))
    lines.append(f"  {'Avg Score':<20}" + fmt_val([r.overall_score for r in reports], "{:.3f}"))
    lines.append(f"  {'Failed':<20}" + "".join(f" {r.total_failed:>{col_w}}" for r in reports))
    lines.append(f"  {'Total Time':<20}" + fmt_val([r.total_time_sec for r in reports], "{:.1f}s"))
    if all(r.avg_tokens_per_sec is not None for r in reports):
        lines.append(f"  {'Avg Tok/s':<20}" + fmt_val([r.avg_tokens_per_sec or 0 for r in reports], "{:.1f}"))
    if any(r.avg_hallucination_confidence > 0 for r in reports):
        lines.append(f"  {'Halluc Conf':<20}" + fmt_val([r.avg_hallucination_confidence for r in reports], "{:.2f}"))

    # Category breakdown
    all_cats = set()
    for r in reports:
        all_cats.update(r.category_summaries.keys())

    lines.append("")
    lines.append(f"  {_c('CATEGORY SCORES', BOLD)}")
    lines.append(f"  {'Category':<20}" + "".join(f" {name:>{col_w}}" for name in model_names))
    lines.append("  " + "─" * (20 + col_w * len(reports)))
    for cat in sorted(all_cats):
        vals = []
        for r in reports:
            if cat in r.category_summaries:
                vals.append(r.category_summaries[cat].avg_score)
            else:
                vals.append(0.0)
        row = f"  {cat:<20}"
        for v in vals:
            color = GREEN if v >= 0.8 else YELLOW if v >= 0.5 else RED
            row += f" {_c(f'{v:.3f}', color):>{col_w}}"
        lines.append(row)

    # Difficulty breakdown
    all_diffs = ["easy", "medium", "hard"]
    if any(any(d in r.difficulty_summaries for d in all_diffs) for r in reports):
        lines.append("")
        lines.append(f"  {_c('DIFFICULTY SCORES', BOLD)}")
        lines.append(f"  {'Difficulty':<20}" + "".join(f" {name:>{col_w}}" for name in model_names))
        lines.append("  " + "─" * (20 + col_w * len(reports)))
        for diff in all_diffs:
            vals = []
            for r in reports:
                if diff in r.difficulty_summaries:
                    vals.append(r.difficulty_summaries[diff].avg_score)
                else:
                    vals.append(0.0)
            row = f"  {diff:<20}"
            for v in vals:
                color = GREEN if v >= 0.8 else YELLOW if v >= 0.5 else RED
                row += f" {_c(f'{v:.3f}', color):>{col_w}}"
            lines.append(row)

    lines.append("")
    lines.append("═" * 72)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# JSON export
# ─────────────────────────────────────────────────────────────────────────

def save_json_report(report: RunReport, filepath: str) -> str:
    """Save report as JSON.  Returns the path written."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    data = {
        "model": report.model,
        "timestamp": report.timestamp,
        "system_prompt_name": report.system_prompt_name,
        "temperature": report.temperature,
        "seed": report.seed,
        "overall_score": report.overall_score,
        "overall_pass_rate": report.overall_pass_rate,
        "total_prompts": report.total_prompts,
        "total_passed": report.total_passed,
        "total_partial": report.total_partial,
        "total_failed": report.total_failed,
        "total_time_sec": report.total_time_sec,
        "avg_tokens_per_sec": report.avg_tokens_per_sec,
        "total_tokens_used": report.total_tokens_used,
        "avg_hallucination_confidence": report.avg_hallucination_confidence,
        "category_summaries": {
            cat: {
                "total": cs.total, "passed": cs.passed, "partial": cs.partial,
                "failed": cs.failed, "avg_score": cs.avg_score,
                "avg_time_sec": cs.avg_time_sec,
                "avg_tokens_per_sec": cs.avg_tokens_per_sec,
            }
            for cat, cs in report.category_summaries.items()
        },
        "difficulty_summaries": {
            diff: {"total": ds.total, "passed": ds.passed, "avg_score": ds.avg_score}
            for diff, ds in report.difficulty_summaries.items()
        },
        "results": [
            {
                "prompt_id": pr.prompt_id,
                "category": pr.category,
                "difficulty": pr.difficulty,
                "prompt_text": pr.prompt_text,
                "model_response": pr.model_response,
                "expected": pr.expected if not isinstance(pr.expected, (dict, list)) else pr.expected,
                "check": pr.check,
                "passed": pr.passed,
                "score": pr.score,
                "detail": pr.detail,
                "extracted": pr.extracted,
                "response_time_sec": pr.response_time_sec,
                "tokens_per_sec": pr.tokens_per_sec,
                "eval_count": pr.eval_count,
                "total_tokens": pr.total_tokens,
                "hallucination_confidence": pr.hallucination_confidence,
                "notes": pr.notes,
            }
            for pr in report.results
        ],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return filepath


# ─────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────

def save_csv_report(report: RunReport, filepath: str) -> str:
    """Save report as CSV (one row per prompt).  Returns the path written."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt_id", "category", "difficulty", "check", "passed",
            "score", "response_time_sec", "tokens_per_sec", "eval_count",
            "hallucination_confidence", "model_response", "expected",
            "detail", "notes",
        ])
        for pr in report.results:
            writer.writerow([
                pr.prompt_id, pr.category, pr.difficulty, pr.check,
                pr.passed, pr.score, f"{pr.response_time_sec:.2f}",
                f"{pr.tokens_per_sec:.1f}" if pr.tokens_per_sec else "",
                pr.eval_count or "",
                f"{pr.hallucination_confidence:.2f}" if pr.hallucination_confidence else "0.00",
                pr.model_response[:500],
                str(pr.expected)[:200],
                pr.detail[:300],
                pr.notes[:200],
            ])
    return filepath


# ─────────────────────────────────────────────────────────────────────────
# HTML export
# ─────────────────────────────────────────────────────────────────────────

def save_html_report(report: RunReport, filepath: str) -> str:
    """Save a self-contained HTML report.  Returns the path written."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    def score_color(v: float) -> str:
        if v >= 0.8:
            return "#28a745"
        elif v >= 0.5:
            return "#ffc107"
        else:
            return "#dc3545"

    def pass_icon(passed: bool, score: float) -> str:
        if score == 1.0:
            return "✅"
        elif score > 0:
            return "⚠️"
        return "❌"

    cat_rows = ""
    for cat in sorted(report.category_summaries.keys()):
        cs = report.category_summaries[cat]
        color = score_color(cs.avg_score)
        cat_rows += f"""
        <tr>
          <td>{cat}</td>
          <td>{cs.total}</td>
          <td>{cs.passed}</td>
          <td>{cs.partial}</td>
          <td>{cs.failed}</td>
          <td style="color:{color}; font-weight:bold">{cs.avg_score:.1%}</td>
          <td>{cs.avg_time_sec:.1f}s</td>
          <td>{f'{cs.avg_tokens_per_sec:.1f}' if cs.avg_tokens_per_sec else '-'}</td>
        </tr>"""

    diff_rows = ""
    for diff in ["easy", "medium", "hard"]:
        if diff in report.difficulty_summaries:
            ds = report.difficulty_summaries[diff]
            color = score_color(ds.avg_score)
            diff_rows += f"""
        <tr>
          <td>{diff}</td>
          <td>{ds.total}</td>
          <td>{ds.passed}</td>
          <td style="color:{color}; font-weight:bold">{ds.avg_score:.1%}</td>
        </tr>"""

    prompt_rows = ""
    for pr in report.results:
        icon = pass_icon(pr.passed, pr.score)
        color = score_color(pr.score)
        response_escaped = pr.model_response.replace("<", "&lt;").replace(">", "&gt;")[:300]
        prompt_escaped = pr.prompt_text.replace("<", "&lt;").replace(">", "&gt;")[:150]
        prompt_rows += f"""
        <tr>
          <td>{icon}</td>
          <td>{pr.category}</td>
          <td>{pr.difficulty}</td>
          <td>{pr.prompt_id}</td>
          <td style="color:{color}; font-weight:bold">{pr.score:.0%}</td>
          <td>{pr.response_time_sec:.1f}s</td>
          <td>{f'{pr.tokens_per_sec:.1f}' if pr.tokens_per_sec else '-'}</td>
          <td title="{prompt_escaped}">{prompt_escaped}</td>
          <td title="{response_escaped}">{response_escaped[:100]}</td>
          <td>{pr.detail[:100]}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>baby-llm-nanny Report — {report.model}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          margin: 2rem; background: #f8f9fa; color: #212529; }}
  h1, h2 {{ color: #495057; }}
  .card {{ background: #fff; border-radius: 8px; padding: 1.5rem;
           margin-bottom: 1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
  th, td {{ padding: 0.5rem 0.75rem; text-align: left;
           border-bottom: 1px solid #dee2e6; }}
  th {{ background: #e9ecef; font-weight: 600; }}
  .metric {{ display: inline-block; min-width: 8rem; }}
  .big {{ font-size: 1.5rem; font-weight: bold; }}
  .header {{ display: flex; justify-content: space-between; align-items: center; }}
</style>
</head>
<body>

<div class="header">
  <h1>🍼 baby-llm-nanny</h1>
  <span style="color:#6c757d">{report.timestamp}</span>
</div>

<div class="card">
  <h2>Overview — <code>{report.model}</code></h2>
  <p>
    <span class="metric"><b>System prompt:</b> {report.system_prompt_name}</span>
    <span class="metric"><b>Temperature:</b> {report.temperature}</span>
    <span class="metric"><b>Seed:</b> {report.seed}</span>
  </p>
  <p>
    <span class="metric big" style="color:{score_color(report.overall_pass_rate)}">
      {report.overall_pass_rate:.1%} pass</span>
    <span class="metric big" style="color:{score_color(report.overall_score)}">
      {report.overall_score:.3f} score</span>
    <span class="metric"><b>Prompts:</b> {report.total_prompts}</span>
    <span class="metric"><b>Failed:</b> {report.total_failed}</span>
    <span class="metric"><b>Time:</b> {report.total_time_sec:.1f}s</span>
    <span class="metric"><b>Tokens:</b> {report.total_tokens_used or '-'}</span>
    <span class="metric"><b>Tok/s:</b> {f'{report.avg_tokens_per_sec:.1f}' if report.avg_tokens_per_sec else '-'}</span>
  </p>
</div>

<div class="card">
  <h2>By Category</h2>
  <table>
    <thead><tr>
      <th>Category</th><th>Total</th><th>Pass</th><th>Partial</th><th>Fail</th>
      <th>Score</th><th>Time</th><th>Tok/s</th>
    </tr></thead>
    <tbody>{cat_rows}
    </tbody>
  </table>
</div>

<div class="card">
  <h2>By Difficulty</h2>
  <table>
    <thead><tr><th>Difficulty</th><th>Total</th><th>Pass</th><th>Score</th></tr></thead>
    <tbody>{diff_rows}
    </tbody>
  </table>
</div>

<div class="card">
  <h2>Per-Prompt Details</h2>
  <table>
    <thead><tr>
      <th></th><th>Category</th><th>Diff</th><th>ID</th><th>Score</th>
      <th>Time</th><th>Tok/s</th><th>Prompt</th><th>Response</th><th>Detail</th>
    </tr></thead>
    <tbody>{prompt_rows}
    </tbody>
  </table>
</div>

</body>
</html>"""

    with open(filepath, "w") as f:
        f.write(html)
    return filepath