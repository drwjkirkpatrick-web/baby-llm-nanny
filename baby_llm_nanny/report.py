"""Report generation — terminal output and JSON export for evaluation results."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PromptResult:
    """Result of a single prompt evaluation."""
    prompt_id: str
    category: str
    prompt_text: str
    model_response: str
    expected: Any
    check: str
    passed: bool
    score: float
    detail: str
    extracted: str | None = None
    response_time_sec: float = 0.0
    notes: str = ""


@dataclass
class CategorySummary:
    """Aggregate results for a category."""
    category: str
    total: int = 0
    passed: int = 0
    partial: int = 0  # score > 0 but < 1
    failed: int = 0
    avg_score: float = 0.0
    avg_time_sec: float = 0.0


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

    @property
    def total_prompts(self) -> int:
        return len(self.results)

    @property
    def total_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

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
    report = RunReport(
        model=model,
        timestamp=datetime.now().isoformat(),
        system_prompt_name=system_prompt_name,
        temperature=temperature,
        seed=seed,
    )

    for tp, resp, eval_res in zip(prompts, responses, evaluations):
        pr = PromptResult(
            prompt_id=tp.id,
            category=tp.category,
            prompt_text=tp.prompt,
            model_response=resp.response,
            expected=tp.expected,
            check=tp.check,
            passed=eval_res.passed,
            score=eval_res.score,
            detail=eval_res.detail,
            extracted=eval_res.extracted,
            response_time_sec=resp.response_time_sec,
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
        report.category_summaries[cat] = CategorySummary(
            category=cat,
            total=total,
            passed=passed,
            partial=partial,
            failed=failed,
            avg_score=avg_score,
            avg_time_sec=avg_time,
        )

    return report


def format_terminal_report(report: RunReport, verbose: bool = False) -> str:
    """Format a RunReport as a colored terminal string."""
    lines = []
    lines.append("")
    lines.append("═" * 64)
    lines.append(f"  🍼 baby-llm-nanny — Report for {report.model}")
    lines.append("═" * 64)
    lines.append(f"  Timestamp:     {report.timestamp}")
    lines.append(f"  System prompt: {report.system_prompt_name}")
    lines.append(f"  Temperature:   {report.temperature}")
    lines.append(f"  Seed:          {report.seed}")
    lines.append("")

    # Overall scores
    lines.append("─" * 64)
    lines.append("  OVERALL")
    lines.append("─" * 64)
    lines.append(f"  Prompts:       {report.total_prompts}")
    lines.append(f"  Passed:        {report.total_passed} / {report.total_prompts} "
                 f"({report.overall_pass_rate:.1%})")
    lines.append(f"  Partial:       {report.total_partial}")
    lines.append(f"  Failed:        {report.total_failed}")
    lines.append(f"  Avg score:     {report.overall_score:.3f}")
    lines.append(f"  Total time:    {report.total_time_sec:.1f}s")
    lines.append("")

    # Category breakdown
    lines.append("─" * 64)
    lines.append("  BY CATEGORY")
    lines.append("─" * 64)
    lines.append(f"  {'Category':<16} {'Pass':<8} {'Partial':<8} {'Fail':<8} {'AvgScore':<10} {'AvgTime':<10}")
    lines.append("  " + "─" * 54)
    for cat in sorted(report.category_summaries.keys()):
        cs = report.category_summaries[cat]
        lines.append(
            f"  {cat:<16} {cs.passed:<8} {cs.partial:<8} {cs.failed:<8} "
            f"{cs.avg_score:<10.3f} {cs.avg_time_sec:<10.1f}"
        )
    lines.append("")

    # Per-prompt details
    lines.append("─" * 64)
    lines.append("  PER-PROMPT DETAILS")
    lines.append("─" * 64)
    for pr in report.results:
        icon = "✅" if pr.passed else ("⚠️" if pr.score > 0 else "❌")
        lines.append(f"  {icon} [{pr.category}] {pr.prompt_id}")
        if verbose or not pr.passed:
            lines.append(f"     Prompt:    {pr.prompt_text[:120]}")
            lines.append(f"     Response:  {pr.model_response[:200]}")
            lines.append(f"     Expected:  {str(pr.expected)[:120]}")
            lines.append(f"     Detail:    {pr.detail}")
            lines.append(f"     Time:      {pr.response_time_sec:.1f}s")
            if pr.notes:
                lines.append(f"     Notes:     {pr.notes}")
            lines.append("")

    if not verbose:
        lines.append("  (Use --verbose for full details on all prompts)")
        lines.append("")

    lines.append("═" * 64)
    return "\n".join(lines)


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
        "category_summaries": {
            cat: {
                "total": cs.total,
                "passed": cs.passed,
                "partial": cs.partial,
                "failed": cs.failed,
                "avg_score": cs.avg_score,
                "avg_time_sec": cs.avg_time_sec,
            }
            for cat, cs in report.category_summaries.items()
        },
        "results": [
            {
                "prompt_id": pr.prompt_id,
                "category": pr.category,
                "prompt_text": pr.prompt_text,
                "model_response": pr.model_response,
                "expected": pr.expected if not isinstance(pr.expected, (dict, list)) else pr.expected,
                "check": pr.check,
                "passed": pr.passed,
                "score": pr.score,
                "detail": pr.detail,
                "extracted": pr.extracted,
                "response_time_sec": pr.response_time_sec,
                "notes": pr.notes,
            }
            for pr in report.results
        ],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return filepath