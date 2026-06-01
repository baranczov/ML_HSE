"""
Deployment-oriented benchmarking utilities.

Measures: model size, inference latency, throughput, peak memory,
and quality metrics — then assembles a comparison DataFrame analogous
to Tables 1-2 in the reference distillation/quantization report.
"""
import io
import time
import tracemalloc

import numpy as np
import pandas as pd
import psutil
import torch

from src.utils import rmse, mae, acc_at_k


def measure_model_size_mb(model) -> float:
    """Size of the model's state_dict serialized to a buffer (MB)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024 / 1024


def measure_latency(model, example_inputs, device=None, n_warmup=10, n_iter=100):
    """
    Measure per-batch inference latency and throughput.

    Args:
        model:          PyTorch model in eval mode.
        example_inputs: torch.Tensor of shape (B, C, H, W).
        device:         device to run on (defaults to CPU for quantized models).
        n_warmup:       warm-up iterations (not counted).
        n_iter:         measured iterations.

    Returns:
        dict with keys: latency_ms, throughput_img_per_s
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()
    x = example_inputs.to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iter):
            model(x)
    # Flush async ops before stopping the clock
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    batch_size = x.size(0)
    latency_ms = (elapsed / n_iter) * 1000
    throughput  = (batch_size * n_iter) / elapsed

    return {"latency_ms": latency_ms, "throughput_img_per_s": throughput}


def measure_peak_memory_mb(model, example_inputs, device=None) -> float:
    """
    Measure peak memory consumption during a single forward pass (MB).

    Uses tracemalloc for CPU; torch.cuda.max_memory_allocated for CUDA.
    Returns 0.0 on MPS (no reliable API).
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()
    x = example_inputs.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(x)
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024

    elif device.type == "cpu":
        # psutil RSS delta: captures PyTorch tensor allocs (tracemalloc misses them)
        proc = psutil.Process()
        rss_before = proc.memory_info().rss
        with torch.no_grad():
            model(x)
        rss_after = proc.memory_info().rss
        return max(0.0, (rss_after - rss_before) / 1024 / 1024)

    return 0.0


def evaluate_quality(model, loader, device=None):
    """
    Run full val-set inference and return quality metrics.

    Returns dict: rmse, mae, acc@5, acc@10
    """
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    preds_list, trues_list = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).squeeze(1).cpu().numpy()
            pred = np.clip(pred, 0, 116)
            preds_list.append(pred)
            trues_list.append(y.numpy())

    preds = np.concatenate(preds_list)
    trues = np.concatenate(trues_list)

    return {
        "rmse":   rmse(trues, preds),
        "mae":    mae(trues, preds),
        "acc@5":  acc_at_k(trues, preds, k=5),
        "acc@10": acc_at_k(trues, preds, k=10),
    }


def compare_models(models_dict, val_loader, example_inputs,
                   bench_device=None, quality_device=None):
    """
    Build a comparison DataFrame for a dict of {name: model}.

    Columns: model_size_mb, latency_ms, throughput_img_per_s,
             peak_memory_mb, rmse, mae, acc@5, acc@10

    Args:
        models_dict:    OrderedDict / dict of {str: nn.Module}
        val_loader:     DataLoader for quality evaluation
        example_inputs: example batch tensor (B, C, H, W)
        bench_device:   device for latency/memory (default: CPU)
        quality_device: device for quality eval (default: CPU)
    """
    if bench_device is None:
        bench_device = torch.device("cpu")
    if quality_device is None:
        quality_device = torch.device("cpu")

    rows = []

    for name, model in models_dict.items():
        print(f"  benchmarking: {name} ...")
        model.eval()

        size_mb = measure_model_size_mb(model)
        lat     = measure_latency(model, example_inputs, device=bench_device)
        mem_mb  = measure_peak_memory_mb(model, example_inputs, device=bench_device)
        quality = evaluate_quality(model, val_loader, device=quality_device)

        row = {"model": name, "model_size_mb": round(size_mb, 2)}
        row.update({k: round(v, 4) for k, v in lat.items()})
        row["peak_memory_mb"] = round(mem_mb, 2)
        row.update({k: round(v, 4) for k, v in quality.items()})
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    return df
