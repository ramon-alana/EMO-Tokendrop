"""Materialize the 2,000+2,000 EmoSet parquet rows for EMO-R3 training."""

import io
import json
import os
from pathlib import Path

import pyarrow.parquet as parquet
from PIL import Image


root = Path(os.environ.get("EMOR3_PROJECT_ROOT", Path(__file__).resolve().parent))
source_dir = root / 'data/EmoSet2k/data'
target_dir = root / 'data/EmoSet2k_full'
image_dir = target_dir / 'images'
image_dir.mkdir(parents=True, exist_ok=True)

for split in ('train', 'test'):
    source = next(source_dir.glob(f'{split}-*.parquet'))
    dest = target_dir / f'{split}.jsonl'
    if dest.exists():
        raise FileExistsError(f'Refusing to replace existing dataset: {dest}')
    staging = target_dir / f'{split}.jsonl.incomplete'
    if staging.exists():
        raise FileExistsError(f'Remove or inspect incomplete conversion first: {staging}')

    count = 0
    with staging.open('x', encoding='utf-8') as stream:
        for batch in parquet.ParquetFile(source).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                if len(row['images']) != 1:
                    raise ValueError(f'{split}_{count}: expected one image')
                raw = row['images'][0]['bytes']
                if not raw:
                    raise ValueError(f'{split}_{count}: missing image bytes')
                with Image.open(io.BytesIO(raw)) as image:
                    extension = {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp'}[image.format]
                    image.verify()
                image_path = image_dir / f'{split}_{count}.{extension}'
                if image_path.exists():
                    if image_path.read_bytes() != raw:
                        raise ValueError(f'Existing image differs: {image_path}')
                else:
                    image_path.write_bytes(raw)

                problem = row['problem'].replace('<image>', '<image>\n', 1)
                record = {
                    'id': f'{split}_{count}',
                    'problem': problem,
                    'images': [str(image_path)],
                    'answer': row['answer'],
                }
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
    staging.rename(dest)
    print(f'{split}: {count} records -> {dest}', flush=True)
