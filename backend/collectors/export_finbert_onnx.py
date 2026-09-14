"""Export FinBERT to ONNX INT8 for CPU inference.

Run once to create the quantized model:
    python -m collectors.export_finbert_onnx

Output: ~/.cache/finbert-int8/{model.onnx,config.json,tokenizer files}
"""

import logging
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("export_finbert_onnx")

MODEL_ID = "ProsusAI/finbert"
CACHE_DIR = Path.home() / ".cache" / "finbert-int8"


def _strip_and_quantize(onnx_fp32: str, onnx_int8: str) -> bool:
    """Strip shape info and quantize - avoids shape inference conflicts."""
    model = onnx.load(onnx_fp32)
    model.graph.ClearField("value_info")
    cleaned = onnx_fp32.replace(".onnx", "_cleaned.onnx")
    onnx.save(model, cleaned)
    try:
        quantize_dynamic(
            cleaned,
            onnx_int8,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
        )
        return True
    except Exception as e:
        logger.warning("INT8 quantize attempt failed: %s", e)
        return False
    finally:
        Path(cleaned).unlink(missing_ok=True)


def export_and_quantize():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading FinBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(str(CACHE_DIR))

    logger.info("Loading FinBERT model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()

    class WrappedModel(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapped = WrappedModel(model)

    dummy = tokenizer("test", return_tensors="pt")
    onnx_fp32 = str(CACHE_DIR / "model.onnx")
    logger.info("Exporting to ONNX FP32...")
    torch.onnx.export(
        wrapped,
        (dummy["input_ids"], dummy["attention_mask"]),
        onnx_fp32,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    logger.info("Quantizing to INT8...")
    onnx_int8 = str(CACHE_DIR / "model_int8.onnx")
    if _strip_and_quantize(onnx_fp32, onnx_int8):
        Path(onnx_fp32).unlink(missing_ok=True)
        size_mb = Path(onnx_int8).stat().st_size / 1e6
        logger.info("INT8 model: %.1f MB", size_mb)
        return onnx_int8
    else:
        logger.warning("Falling back to FP32 ONNX model")
        return onnx_fp32


def verify():
    import onnxruntime as ort

    int8_path = CACHE_DIR / "model_int8.onnx"
    fp32_path = CACHE_DIR / "model.onnx"
    onnx_path = int8_path if int8_path.exists() else fp32_path

    if not onnx_path.exists():
        logger.warning("Model not found, run export first")
        return

    tokenizer = AutoTokenizer.from_pretrained(str(CACHE_DIR))
    session = ort.InferenceSession(str(onnx_path))
    model_type = "INT8" if int8_path.exists() else "FP32"

    def score(text: str) -> float:
        inp = tokenizer(text, return_tensors="np", truncation=True, max_length=128)
        logits = session.run(
            None,
            {
                "input_ids": inp["input_ids"],
                "attention_mask": inp["attention_mask"],
            },
        )[0][0]
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        return float(probs[2] - probs[0])

    logger.info("Verifying %s ONNX model:", model_type)
    tests = [
        "BFK CW MW low float",
        "Blue gem T1 pattern",
        "This skin is sleeping",
        "Pump and dump incoming",
        "Sold my collection",
        "Redline is so undervalued right now",
        "Great price on this AK",
    ]
    all_ok = True
    for text in tests:
        s = score(text)
        logger.info("  %-45s → %+.4f", text, s)

    logger.info("Verification passed (%s, %.1f MB)", model_type, onnx_path.stat().st_size / 1e6)
    return all_ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    export_and_quantize()
    verify()
