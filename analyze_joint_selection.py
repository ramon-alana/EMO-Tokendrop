"""Audit paired TokenDrop results and the learned selector's decisions.

Use only after all requested shards have finished. The script refuses partial,
misaligned or mixed-protocol inputs; it never selects a checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter
from pathlib import Path


def read_method(root: Path, method: str, expected: int) -> dict[str, dict]:
    paths = sorted(glob.glob(str(root / f"{method}_shard*.jsonl")))
    if not paths:
        raise ValueError(f"Missing {method} shards")
    rows = {}
    for path in paths:
        for text in Path(path).read_text(encoding="utf-8").splitlines():
            if not text:
                continue
            row = json.loads(text)
            sample_id = str(row["id"])
            if sample_id in rows or row["method"] != method:
                raise ValueError(f"Duplicate ID or method mismatch: {path} {sample_id}")
            rows[sample_id] = row
    if len(rows) != expected:
        raise ValueError(f"{method}: only {len(rows)}/{expected} unique images")
    return rows


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    dx = [x - xbar for x in xs]
    dy = [y - ybar for y in ys]
    denominator = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denominator if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--count", type=int, choices=(200, 1000, 2000), required=True)
    parser.add_argument(
        "--methods", nargs="+", default=["full", "random_matched", "foe_matched", "learned"]
    )
    args = parser.parse_args()
    results = {method: read_method(args.root, method, args.count) for method in args.methods}
    methods = list(results)
    ids = set(results[methods[0]])
    for method in methods[1:]:
        if set(results[method]) != ids:
            raise ValueError(f"{method}: test IDs differ")
    protocol_keys = ("answer", "test_file", "checkpoint", "joint_step", "prefill_step1", "budget_decode")
    for sample_id in ids:
        reference = results[methods[0]][sample_id]
        for method in methods[1:]:
            row = results[method][sample_id]
            if any(row[key] != reference[key] for key in protocol_keys):
                raise ValueError(f"{sample_id}: mixed protocol in {method}")
        drop_rows = [results[m][sample_id] for m in methods if m != "full"]
        if drop_rows and any(row["visual_tokens_after"] != drop_rows[0]["visual_tokens_after"] for row in drop_rows):
            raise ValueError(f"{sample_id}: unequal token budgets")
    summary = {"count": args.count, "methods": {}, "paired": {}, "selector": {}}
    ordered = sorted(ids)
    for method in methods:
        rows = [results[method][sample_id] for sample_id in ordered]
        summary["methods"][method] = {
            "correct": sum(bool(row["correct"]) for row in rows),
            "strict_set": sum(bool(row["strict_set"]) for row in rows),
            "truncated": sum(bool(row["hit_length_cap"]) for row in rows),
            "drop_mean": mean(rows, "ratio_actual"),
            "visual_tokens_before_mean": mean(rows, "visual_tokens_before"),
            "visual_tokens_after_mean": mean(rows, "visual_tokens_after"),
            **{f"{name}_seconds_mean": mean(rows, f"{name}_seconds")
               for name in ("probe", "vision", "generation", "pipeline")},
        }
    if "learned" in results:
        learned = results["learned"]
        for method in methods:
            if method == "learned":
                continue
            other = results[method]
            summary["paired"][f"learned_vs_{method}"] = {
                "learned_only_correct": sum(learned[i]["correct"] and not other[i]["correct"] for i in ordered),
                "other_only_correct": sum(other[i]["correct"] and not learned[i]["correct"] for i in ordered),
            }
        rows = [learned[i] for i in ordered]
        budgets = [float(row["budget_target"]) for row in rows]
        confidence = [float(row["sepm_confidence"]["confidence"]) for row in rows]
        summary["selector"] = {
            "budget_counts": dict(sorted(Counter(f"{x:.2f}" for x in budgets).items())),
            "budget_entropy_mean": mean(rows, "budget_entropy"),
            "budget_range": [min(budgets), max(budgets)],
            "actual_drop_range": [min(row["ratio_actual"] for row in rows), max(row["ratio_actual"] for row in rows)],
            "confidence_budget_pearson": pearson(confidence, budgets),
            "budget_probabilities_mean": [
                sum(float(row["budget_probabilities"][j]) for row in rows) / len(rows)
                for j in range(len(rows[0]["budget_probabilities"]))
            ],
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
