"""Stage 6a: zero-shot sentiment probabilities from a pretrained classifier, one model per run.

Run with the GPU environment (from the repo root):
    <gpu-env>\\Scripts\\python scripts\\run_stage6a_sentiment_features.py --model finbert
    <gpu-env>\\Scripts\\python scripts\\run_stage6a_sentiment_features.py --model sst2
    ... --benchmark          # throughput and fp32-vs-fp16 agreement on a small sample, writes nothing else

Safety: only train and validation rows are read. The sealed test split is refused unless
--allow-test is passed explicitly (do that only after the Stage 7 freeze).
Output: data/processed/sent_<tag>.parquet (keys + probabilities, git-ignored) and
reports/text_models/sent_<tag>_manifest.json. Label names come from the model's own config,
never from an assumed order.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "reviews_clean.parquet"
OUT_DIR = ROOT / "reports" / "text_models"
KEY = ["user_id", "parent_asin", "timestamp_ms"]
MODELS = {"finbert": "ProsusAI/finbert", "sst2": "distilbert-base-uncased-finetuned-sst-2-english"}

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=sorted(MODELS), required=True)
ap.add_argument("--splits", nargs="+", default=["train", "validation"])
ap.add_argument("--allow-test", action="store_true", help="only after the Stage 7 freeze")
ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
ap.add_argument("--batch-size", type=int, default=128)
ap.add_argument("--max-length", type=int, default=128)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--benchmark", action="store_true")
args = ap.parse_args()
if "test" in args.splits and not args.allow_test:
    sys.exit("refusing to read the sealed test split without --allow-test")
if "stream" in args.splits:
    sys.exit("stream is not scored here")

name = MODELS[args.model]
df = pd.read_parquet(CLEAN, columns=KEY + ["title", "text", "split"], filters=[("split", "in", args.splits)])
if args.limit or args.benchmark:
    df = df.sample(n=min(args.limit or 2000, len(df)), random_state=42)
df = df.reset_index(drop=True)
texts = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()

tok = AutoTokenizer.from_pretrained(name, cache_dir=str(ROOT / "hf_cache"))
model = AutoModelForSequenceClassification.from_pretrained(name, cache_dir=str(ROOT / "hf_cache")).to("cuda").eval()
id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
labels = [id2label[i] for i in range(len(id2label))]
print("device:", torch.cuda.get_device_name(0), "| labels from config:", labels, flush=True)


def score(dtype: str) -> tuple[np.ndarray, float]:
    order = np.argsort(texts.str.len().to_numpy(), kind="stable")  # similar lengths -> less padding
    out = np.zeros((len(texts), len(labels)), dtype=np.float32)
    t = time.time()
    with torch.no_grad():
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            batch = tok(texts.iloc[idx].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to("cuda")
            if dtype == "fp16":
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**batch).logits
            else:
                logits = model(**batch).logits
            out[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            if (start // args.batch_size) % 100 == 0:
                print(f"  {start}/{len(order)} rows, {time.time() - t:.0f}s", flush=True)
    return out, time.time() - t


if args.benchmark:
    p32, s32 = score("fp32")
    p16, s16 = score("fp16")
    res = {"model": name, "rows": len(df), "fp32_rows_per_s": round(len(df) / s32, 1), "fp16_rows_per_s": round(len(df) / s16, 1),
           "max_abs_prob_diff_fp16_vs_fp32": float(np.abs(p32 - p16).max()),
           "label_agreement_fp16_vs_fp32": float((p32.argmax(1) == p16.argmax(1)).mean())}
    print(json.dumps(res, indent=2))
    sys.exit(0)

probs, seconds = score(args.dtype)
tag = args.model
out = df[KEY].copy()
for i, lab in enumerate(labels):
    out[f"{tag}_{lab}"] = probs[:, i]
out_path = ROOT / "data" / "processed" / f"sent_{tag}.parquet"
out.to_parquet(out_path, index=False, compression="zstd")
OUT_DIR.mkdir(parents=True, exist_ok=True)
manifest = {
    "model": name, "commit_hash": getattr(model.config, "_commit_hash", None), "id2label_from_config": id2label,
    "dtype": args.dtype, "max_length": args.max_length, "batch_size": args.batch_size, "splits": args.splits,
    "rows": int(len(out)), "seconds": round(seconds, 1), "rows_per_second": round(len(out) / seconds, 1),
    "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "output": str(out_path.relative_to(ROOT)),
    "input_text": "title + ' ' + text, truncated to max_length tokens",
}
(OUT_DIR / f"sent_{tag}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
