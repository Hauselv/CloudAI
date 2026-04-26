from __future__ import annotations

import pandas as pd


def summarize_runs(run_summaries: list[dict[str, object]]) -> pd.DataFrame:
    if not run_summaries:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for run in run_summaries:
        metrics = run.get("metrics", {})
        row = {"run_id": run.get("run_id"), "kind": run.get("kind")}
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)
