"""GPU smoke test. Run with the SEPARATE GPU environment (short path, because the ROCm SDK files exceed the
Windows long-path limit under this repo's long path):  <gpu-env>\\Scripts\\python scripts\\gpu_smoke_test.py

Checks, in order (each result is recorded even if a later check fails):
1. PyTorch sees the GPU (ROCm build on Windows) and reports version info.
2. A matrix multiply on the GPU matches the CPU.
3. A small transformer encoder trains one step: forward, loss.backward(), optimizer step,
   gradients finite. Backward is the step that has crashed on some ROCm setups.
4. (needs `transformers`) ProsusAI/finbert loads, its config label order is read, a forward pass
   runs on the GPU, and one backward step on a dummy label works.
Writes reports/gpu/smoke_test.json. No project data is read.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "gpu"
result: dict = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None), "checks": {}}


def check(name):
    def wrap(fn):
        t = time.time()
        try:
            out = fn()
            result["checks"][name] = {"ok": True, "seconds": round(time.time() - t, 2), **(out or {})}
        except Exception as exc:  # record and continue
            result["checks"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-800:]}
        print(name, "->", {k: v for k, v in result["checks"][name].items() if k != "trace"}, flush=True)
        return fn
    return wrap


@check("1_gpu_visible")
def _():
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    return {"device_name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@check("2_matmul_matches_cpu")
def _():
    torch.manual_seed(0)
    a, b = torch.randn(512, 512), torch.randn(512, 512)
    gpu = (a.to(DEVICE) @ b.to(DEVICE)).cpu()
    cpu = a @ b
    err = float((gpu - cpu).abs().max())
    assert err < 1e-2, f"max abs error {err}"
    return {"max_abs_error": err}


@check("3_transformer_backward_step")
def _():
    torch.manual_seed(0)
    layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, batch_first=True)
    model = torch.nn.Sequential(layer, torch.nn.Flatten(), torch.nn.LazyLinear(4)).to(DEVICE)
    x = torch.randn(16, 32, 128, device=DEVICE)
    y = torch.randint(0, 4, (16,), device=DEVICE)
    model(x)  # materialize lazy layer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(5):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    grads_finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert grads_finite, "non-finite gradients"
    return {"losses": [round(l, 4) for l in losses], "loss_decreased": losses[-1] < losses[0]}


@check("4_finbert_forward_backward_and_labels")
def _():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(name, cache_dir=str(ROOT / "hf_cache"))
    model = AutoModelForSequenceClassification.from_pretrained(name, cache_dir=str(ROOT / "hf_cache")).to(DEVICE)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    texts = ["This product is great and works perfectly.", "Terrible, it broke after two days.", "It is a bottle of vitamins."]
    batch = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(**batch).logits, dim=-1).cpu()
    model.train()
    loss = model(**batch, labels=torch.tensor([0, 1, 2], device=DEVICE)).loss
    loss.backward()
    grads_finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert grads_finite
    return {
        "id2label": id2label,
        "probs": [[round(float(v), 3) for v in row] for row in probs],
        "predicted_labels": [id2label[int(i)] for i in probs.argmax(dim=-1)],
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "backward_loss": round(float(loss), 4),
    }


OUT.mkdir(parents=True, exist_ok=True)
(OUT / "smoke_test.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
print("all checks ok:", all(c["ok"] for c in result["checks"].values()))
sys.exit(0 if all(c["ok"] for c in result["checks"].values()) else 1)
