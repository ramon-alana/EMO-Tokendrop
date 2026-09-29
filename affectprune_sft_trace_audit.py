"""Audit train-only pseudo-SET traces before random-drop SFT."""

from __future__ import annotations

import glob
import os
import json
from collections import Counter
from pathlib import Path


root = Path(os.environ.get("EMOR3_PROJECT_ROOT", Path(__file__).resolve().parent))
train_path = root / "data/EmoSet2k_full/train.jsonl"
trace_glob = str(root / "outputs/affectprune_sft_teacher_train2000/teacher_shard*/full.jsonl")
train = {
    str(row["id"]): row
    for line in train_path.read_text(encoding="utf-8").splitlines()
    if line
    for row in [json.loads(line)]
}
paths = sorted(glob.glob(trace_glob))
seen = set()
selected = set()
per_shard = {}
classes = Counter()
for path in paths:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    accepted = 0
    for row in rows:
        sample_id = str(row["id"])
        if sample_id in seen or sample_id not in train or not sample_id.startswith("train_"):
            raise ValueError(f"Invalid or duplicate training trace: {sample_id}")
        seen.add(sample_id)
        if row.get("answer") != train[sample_id]["answer"]:
            raise ValueError(f"Mismatched training label: {sample_id}")
        if row.get("correct") and row.get("strict_set") and not row.get("hit_length_cap"):
            selected.add(sample_id)
            classes[str(row["answer"])] += 1
            accepted += 1
    per_shard[Path(path).parent.name] = {"rows": len(rows), "accepted": accepted}
print(json.dumps({
    "train_images": len(train),
    "trace_files": len(paths),
    "unique_traces": len(seen),
    "accepted": len(selected),
    "missing": len(train) - len(seen),
    "per_shard": per_shard,
    "accepted_classes": dict(sorted(classes.items())),
}, ensure_ascii=False))
