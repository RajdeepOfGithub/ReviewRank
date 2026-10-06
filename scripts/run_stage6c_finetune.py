"""Stage 6c: fine-tune a transformer as a regressor on log1p(votes). Train split only; test never loaded.

Run with the GPU environment (from the repo root):
    <gpu-env>\\Scripts\\python scripts\\run_stage6c_finetune.py --model distilbert --max-train-rows 60000 --epochs 2
    <gpu-env>\\Scripts\\python scripts\\run_stage6c_finetune.py --model finbert    --max-train-rows 60000 --epochs 2
    (The reported runs used 60,000 rows, as recorded in reports/text_models/ft_*_manifest.json; the argparse
    default below is 150,000, so always pass --max-train-rows 60000 to reproduce them.)
    ... --max-train-rows 2000 --epochs 1 --tag bench        # throughput check

Design: fixed epochs, fixed hyperparameters, NO validation-based early stopping (fewer degrees of
freedom). Validation rows are only SCORED after training. The sealed test is refused unless --allow-test.
Output: data/processed/ft_<tag>_val.parquet (keys + score), reports/text_models/ft_<tag>_manifest.json
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "reviews_clean.parquet"
OUT_DIR = ROOT / "reports" / "text_models"
KEY = ["user_id", "parent_asin", "timestamp_ms"]
BACKBONES = {"distilbert": "distilbert-base-uncased", "finbert": "ProsusAI/finbert"}

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=sorted(BACKBONES), required=True)
ap.add_argument("--max-train-rows", type=int, default=150_000)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--max-length", type=int, default=128)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--tag", default=None)
ap.add_argument("--splits-score", nargs="+", default=["validation"])
ap.add_argument("--allow-test", action="store_true", help="only after the Stage 7 freeze")
args = ap.parse_args()
if "test" in args.splits_score and not args.allow_test:
    sys.exit("refusing to score the sealed test split without --allow-test")
if sys.platform == "win32":
    # Ask Windows not to sleep while this process runs (process-level; no system setting is changed and
    # the request ends when the process exits). Without this, an idle laptop slept mid-run and the job hung.
    import ctypes

    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
tag = args.tag or args.model
name = BACKBONES[args.model]
torch.manual_seed(args.seed)
np.random.seed(args.seed)

df = pd.read_parquet(CLEAN, columns=KEY + ["title", "text", "helpful_vote", "split"], filters=[("split", "in", ["train"] + args.splits_score)])
train = df[df["split"] == "train"]
train = train.sample(n=min(args.max_train_rows, len(train)), random_state=args.seed).reset_index(drop=True)
score_df = df[df["split"].isin(args.splits_score)].reset_index(drop=True)


def text_of(frame):
    return (frame["title"].fillna("") + " " + frame["text"].fillna("")).str.strip().tolist()


tok = AutoTokenizer.from_pretrained(name, cache_dir=str(ROOT / "hf_cache"))
model = AutoModelForSequenceClassification.from_pretrained(
    name, num_labels=1, problem_type="regression", ignore_mismatched_sizes=True, cache_dir=str(ROOT / "hf_cache")
).to("cuda")
y = torch.tensor(np.log1p(train["helpful_vote"].to_numpy(dtype=np.float32)))
texts = text_of(train)

steps_per_epoch = (len(texts) + args.batch_size - 1) // args.batch_size
total_steps = steps_per_epoch * args.epochs
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
scaler = torch.amp.GradScaler("cuda")
print(f"{name}: train rows {len(texts)}, steps {total_steps}", flush=True)

history, t0 = [], time.time()
model.train()
for epoch in range(args.epochs):
    perm = np.random.default_rng(args.seed + epoch).permutation(len(texts))
    running = []
    for s in range(steps_per_epoch):
        idx = perm[s * args.batch_size : (s + 1) * args.batch_size]
        batch = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to("cuda")
        target = y[idx].to("cuda")
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(**batch).logits.squeeze(-1)
        loss = torch.nn.functional.mse_loss(pred.float(), target)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        running.append(float(loss.detach()))
        if s % 200 == 0:
            print(f"  epoch {epoch + 1} step {s}/{steps_per_epoch} loss {np.mean(running[-200:]):.4f} {time.time() - t0:.0f}s", flush=True)
    history.append({"epoch": epoch + 1, "mean_train_loss": float(np.mean(running)), "elapsed_s": round(time.time() - t0, 1)})
train_seconds = time.time() - t0

# ---- score (validation by default); no further training or selection
model.eval()
stext = text_of(score_df)
order = np.argsort([len(t) for t in stext], kind="stable")
scores = np.zeros(len(stext), dtype=np.float32)
t1 = time.time()
with torch.no_grad():
    for s in range(0, len(order), 128):
        idx = order[s : s + 128]
        batch = tok([stext[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=args.max_length).to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            scores[idx] = model(**batch).logits.squeeze(-1).float().cpu().numpy()
score_seconds = time.time() - t1

out = score_df[KEY + ["split"]].copy()
out[f"ft_{tag}"] = scores
out_path = ROOT / "data" / "processed" / f"ft_{tag}_val.parquet"
out.to_parquet(out_path, index=False, compression="zstd")
OUT_DIR.mkdir(parents=True, exist_ok=True)
manifest = {
    "backbone": name, "commit_hash": getattr(model.config, "_commit_hash", None), "task": "regression on log1p(helpful_vote)",
    "train_rows": int(len(texts)), "train_sampling": f"random sample of the train split, seed {args.seed}",
    "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "max_length": args.max_length,
    "precision": "fp16 autocast + GradScaler", "scheduler": "linear, 6% warmup", "early_stopping": "none (fixed epochs)",
    "history": history, "train_seconds": round(train_seconds, 1), "train_rows_per_second": round(args.epochs * len(texts) / train_seconds, 1),
    "scored_rows": int(len(out)), "scored_splits": args.splits_score, "score_seconds": round(score_seconds, 1),
    "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "output": str(out_path.relative_to(ROOT)),
}
(OUT_DIR / f"ft_{tag}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
