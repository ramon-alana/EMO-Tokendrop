# Dataset layout (data files are not included)

Place your licensed EmoSet parquet source in `data/EmoSet2k/data/` with one
`train-*.parquet` and one `test-*.parquet`. Then run
`python convert_emoset_full.py` from the repository root. The converter
creates `data/EmoSet2k_full/train.jsonl`,
`data/EmoSet2k_full/test.jsonl`, and image files. It refuses to overwrite
an existing JSONL file.

Each JSONL row has `id`, `problem`, `images` (one absolute image path),
and `answer`. The recorded experiment uses 2,000 distinct TRAIN IDs and
2,000 distinct TEST IDs. TRAIN and TEST must remain disjoint. Neither image
data nor derived teacher answers are redistributed in this archive.
