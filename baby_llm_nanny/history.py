"""Historical trend tracking — SQLite database for storing and comparing runs over time.

Stores each run's results in a lightweight SQLite database so you can track
model performance regressions over time, compare runs, and see trends.
"""

from __future__ import annotations

import json
import sqlite3
import os
from datetime import datetime
from typing import Optional
from .report import RunReport


DEFAULT_DB_PATH = os.path.expanduser("~/.local/share/baby-llm-nanny/history.db")


def _get_db_path(db_path: str | None = None) -> str:
    """Return the DB path, creating the directory if needed."""
    path = db_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def init_db(db_path: str | None = None) -> str:
    """Initialize the SQLite database.  Returns the path."""
    path = _get_db_path(db_path)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model TEXT NOT NULL,
            system_prompt TEXT NOT NULL,
            temperature REAL NOT NULL,
            seed INTEGER NOT NULL,
            total_prompts INTEGER NOT NULL,
            total_passed INTEGER NOT NULL,
            total_failed INTEGER NOT NULL,
            overall_score REAL NOT NULL,
            overall_pass_rate REAL NOT NULL,
            total_time_sec REAL NOT NULL,
            avg_tokens_per_sec REAL,
            total_tokens_used INTEGER,
            avg_hallucination_confidence REAL
        );

        CREATE TABLE IF NOT EXISTS prompt_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            prompt_id TEXT NOT NULL,
            category TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            passed INTEGER NOT NULL,
            score REAL NOT NULL,
            response_time_sec REAL NOT NULL,
            tokens_per_sec REAL,
            eval_count INTEGER,
            hallucination_confidence REAL DEFAULT 0,
            model_response TEXT,
            detail TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        );

        CREATE TABLE IF NOT EXISTS retry_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            prompt_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            passed INTEGER NOT NULL,
            score REAL NOT NULL,
            response TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        );

        CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model);
        CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp);
        CREATE INDEX IF NOT EXISTS idx_prompt_results_run ON prompt_results(run_id);
    """)
    conn.commit()
    conn.close()
    return path


def save_run_to_db(report: RunReport, db_path: str | None = None) -> int:
    """Save a RunReport to the database.  Returns the run ID."""
    path = init_db(db_path)
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO runs (timestamp, model, system_prompt, temperature, seed,
                         total_prompts, total_passed, total_failed,
                         overall_score, overall_pass_rate, total_time_sec,
                         avg_tokens_per_sec, total_tokens_used,
                         avg_hallucination_confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        report.timestamp, report.model, report.system_prompt_name,
        report.temperature, report.seed,
        report.total_prompts, report.total_passed, report.total_failed,
        report.overall_score, report.overall_pass_rate, report.total_time_sec,
        report.avg_tokens_per_sec, report.total_tokens_used,
        report.avg_hallucination_confidence,
    ))
    run_id = cur.lastrowid

    for pr in report.results:
        cur.execute("""
            INSERT INTO prompt_results (run_id, prompt_id, category, difficulty,
                                       passed, score, response_time_sec,
                                       tokens_per_sec, eval_count,
                                       hallucination_confidence, model_response, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, pr.prompt_id, pr.category, pr.difficulty,
            int(pr.passed), pr.score, pr.response_time_sec,
            pr.tokens_per_sec, pr.eval_count,
            pr.hallucination_confidence, pr.model_response[:500], pr.detail[:500],
        ))

    conn.commit()
    conn.close()
    return run_id


def list_runs(db_path: str | None = None, model: str | None = None,
              limit: int = 20) -> list[dict]:
    """List recent runs from the database."""
    path = _get_db_path(db_path)
    if not os.path.exists(path):
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if model:
        cur.execute("""
            SELECT * FROM runs WHERE model = ? ORDER BY timestamp DESC LIMIT ?
        """, (model, limit))
    else:
        cur.execute("SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?", (limit,))

    runs = [dict(row) for row in cur.fetchall()]
    conn.close()
    return runs


def format_trend_report(db_path: str | None = None, model: str | None = None,
                        limit: int = 10) -> str:
    """Format a trend report showing recent runs and regressions."""
    from .report import GREEN, RED, YELLOW, BOLD, RESET, _c

    runs = list_runs(db_path, model, limit)
    if not runs:
        return "No historical runs found. Run baby-llm-nanny with --save-history to start tracking."

    lines = []
    lines.append("")
    lines.append("═" * 64)
    lines.append(f"  🍼 {_c('Historical Trends', BOLD)}")
    lines.append("═" * 64)

    if model:
        lines.append(f"  Model: {model}")
    lines.append("")

    # Show recent runs
    lines.append(f"  {'Timestamp':<24} {'Model':<16} {'Pass':<8} {'Score':<8} {'Time':<8}")
    lines.append("  " + "─" * 60)

    for run in runs:
        ts = run["timestamp"][:19]
        m = run["model"][:14]
        pr = run["overall_pass_rate"]
        sc = run["overall_score"]
        tm = run["total_time_sec"]
        pr_color = GREEN if pr >= 0.8 else YELLOW if pr >= 0.5 else RED
        lines.append(f"  {ts:<24} {m:<16} {_c(f'{pr:.1%}', pr_color):<8} {sc:<8.3f} {tm:<8.1f}s")

    # Detect regressions
    if len(runs) >= 2:
        latest = runs[0]
        prev = runs[1]
        if latest["model"] == prev["model"]:
            lines.append("")
            lines.append("  " + "─" * 60)
            lines.append(f"  {_c('Change vs Previous Run', BOLD)}")
            score_diff = latest["overall_score"] - prev["overall_score"]
            pass_diff = latest["overall_pass_rate"] - prev["overall_pass_rate"]

            sc_color = GREEN if score_diff >= 0 else RED
            pr_color = GREEN if pass_diff >= 0 else RED

            lines.append(f"  Score:     {_c(f'{score_diff:+.3f}', sc_color)} "
                        f"({prev['overall_score']:.3f} → {latest['overall_score']:.3f})")
            lines.append(f"  Pass rate: {_c(f'{pass_diff:+.1%}', pr_color)} "
                        f"({prev['overall_pass_rate']:.1%} → {latest['overall_pass_rate']:.1%})")

    lines.append("")
    lines.append("═" * 64)
    return "\n".join(lines)


def save_retry_results(run_id: int, prompt_id: str, results: list[dict],
                       db_path: str | None = None) -> None:
    """Save retry/consistency results for a single prompt.

    Each result in the list should have:
      - attempt: int (1-indexed)
      - passed: bool
      - score: float
      - response: str
    """
    path = _get_db_path(db_path)
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    for r in results:
        cur.execute("""
            INSERT INTO retry_results (run_id, prompt_id, attempt, passed, score, response)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (run_id, prompt_id, r["attempt"], int(r["passed"]), r["score"], r["response"][:500]))

    conn.commit()
    conn.close()


def get_retry_stats(db_path: str | None = None, run_id: int | None = None) -> list[dict]:
    """Get retry statistics (consistency) per prompt for a given run."""
    path = _get_db_path(db_path)
    if not os.path.exists(path):
        return []
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if run_id:
        cur.execute("""
            SELECT prompt_id,
                   COUNT(*) as attempts,
                   SUM(CASE WHEN passed=1 THEN 1 ELSE 0 END) as passes,
                   AVG(score) as avg_score,
                   MIN(score) as min_score,
                   MAX(score) as max_score
            FROM retry_results
            WHERE run_id = ?
            GROUP BY prompt_id
            ORDER BY prompt_id
        """, (run_id,))
    else:
        cur.execute("""
            SELECT prompt_id,
                   COUNT(*) as attempts,
                   SUM(CASE WHEN passed=1 THEN 1 ELSE 0 END) as passes,
                   AVG(score) as avg_score,
                   MIN(score) as min_score,
                   MAX(score) as max_score
            FROM retry_results
            GROUP BY prompt_id
            ORDER BY prompt_id
        """)

    stats = [dict(row) for row in cur.fetchall()]
    conn.close()
    return stats