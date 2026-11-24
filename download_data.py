# download_data.py - save local copies of Enron (AESLC) and deepset into data/raw
from pathlib import Path
from datasets import load_dataset   # pip install datasets pyarrow

# Enron emails (AESLC): train / validation / test
out = Path("data/raw/aeslc"); out.mkdir(parents=True, exist_ok=True)
for name in ("Yale-LILY/aeslc", "aeslc"):          # the dataset has moved names before
    try:
        ds = load_dataset(name); break
    except Exception as e:
        print(f"{name} failed: {e}")
for split in ("train", "validation", "test"):
    ds[split].to_pandas().to_json(out / f"{split}.jsonl", orient="records", lines=True, force_ascii=False)
    print("saved", out / f"{split}.jsonl", len(ds[split]), "emails")

# deepset prompt-injections: train / test
out = Path("data/raw/deepset"); out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("deepset/prompt-injections")
for split in ("train", "test"):
    ds[split].to_pandas().to_parquet(out / f"{split}.parquet")
    print("saved", out / f"{split}.parquet", len(ds[split]), "rows")