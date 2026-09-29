"""Freeze a TRAIN-only 64-image development split for free-ratio RL.

The evaluation code accepts a 200-row dataset and evaluates only its first
64 rows. The remaining 136 rows are unused fillers from TRAIN, never TEST.
"""

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("EMOR3_PROJECT_ROOT", Path(__file__).resolve().parent))
source = ROOT / "outputs/affectprune_emor3start_sft_validation_20260929/stepadapter_step2400_ratio0_prefix0_n8.jsonl"
target = ROOT / "data/EmoSet2k_free_train_dev200_20260929.jsonl"
train = [json.loads(line) for line in (ROOT / "data/EmoSet2k_full/train.jsonl").read_text(encoding="utf-8").splitlines() if line]
dev = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
by_id = {str(row["id"]): row for row in train}
ids = [str(row["id"]) for row in dev]
if len(train) != 2000 or len(by_id) != 2000 or len(ids) != 64 or len(set(ids)) != 64:
    raise RuntimeError("TRAIN or prior TRAIN-internal development IDs are incomplete")
if not set(ids) <= set(by_id):
    raise RuntimeError("Development ID absent from TRAIN")
if any(row["answer"] != by_id[str(row["id"])]["answer"] for row in dev):
    raise RuntimeError("Development labels differ from TRAIN")
fillers = [row for row in train if str(row["id"]) not in set(ids)][:136]
rows = [by_id[sample_id] for sample_id in ids] + fillers
if len(rows) != 200 or len({str(row["id"]) for row in rows}) != 200:
    raise RuntimeError("Derived 200-row file is invalid")
if target.exists():
    existing = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines() if line]
    if existing != rows:
        raise RuntimeError("Refusing to overwrite a different development split")
else:
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
print(json.dumps({"path": str(target), "dev_count": 64, "total_rows": 200, "first_id": ids[0], "last_id": ids[-1]}))
