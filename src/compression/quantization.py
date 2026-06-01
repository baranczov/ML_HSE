"""
Model quantization for age-prediction ResNet.

Dynamic quantization — fast, no calibration needed, works on any device.
PTQ (Post-Training Quantization) via torch.ao.quantization FX pipeline —
  significant size reduction (~4x); benchmark on CPU only
  (torch quantized ops don't run on MPS/CUDA).
"""
import copy
import torch
import torch.nn as nn


def quantize_dynamic(model, backend="qnnpack"):
    """
    Apply dynamic quantization to Linear layers.
    Returns a new quantized model; original is unchanged.
    Works on CPU only at inference time.

    Note: Conv2d is excluded — torch dynamic quant only supports Linear in practice.
    """
    # Must set the quantized engine before creating any quantized ops
    torch.backends.quantized.engine = backend

    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()
    q_model = torch.ao.quantization.quantize_dynamic(
        model_cpu,
        qconfig_spec={nn.Linear},
        dtype=torch.qint8,
    )
    return q_model


def prepare_backbone_for_ptq(model, example_inputs, backend="qnnpack"):
    """
    Prepare model for Post-Training Quantization using FX graph mode.

    Args:
        model:          trained float model (will be deep-copied and moved to CPU).
        example_inputs: tuple of example inputs, e.g. (torch.randn(1, 3, 224, 224),).
        backend:        "qnnpack" (ARM/Mac) or "fbgemm" (x86).

    Returns:
        prepared model ready for calibration.
    """
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx

    model_cpu = copy.deepcopy(model).cpu()
    model_cpu.eval()

    torch.backends.quantized.engine = backend
    qconfig_mapping = get_default_qconfig_mapping(backend)

    prepared = prepare_fx(model_cpu, qconfig_mapping, example_inputs)
    return prepared


def calibrate(prepared_model, loader, calibration_samples=200, device="cpu"):
    """
    Run calibration data through the prepared model to collect activation stats.

    Args:
        prepared_model:      model returned by prepare_backbone_for_ptq.
        loader:              DataLoader (val or train split).
        calibration_samples: max number of images to use for calibration.
        device:              must be CPU for PTQ.
    """
    prepared_model.eval()
    n_seen = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            prepared_model(x)
            n_seen += x.size(0)
            if n_seen >= calibration_samples:
                break

    return prepared_model


def convert_prepared_quantized_model(prepared_model):
    """
    Convert a calibrated prepared model to the final quantized form.
    Returns the quantized model (CPU-only inference).
    """
    from torch.ao.quantization.quantize_fx import convert_fx
    return convert_fx(prepared_model)
