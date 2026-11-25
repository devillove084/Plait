"""Reproduce the Qwen3.5-9B W4A16 checkpoint used by the NVIDIA Ampere tier.

Run in a transformers 5.17 / llmcompressor 0.14 environment, not in the vLLM
runtime environment. GPTQ is calibration-based; replace the local README
calibration corpus with a representative text corpus for production quality.
"""

import os

from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL = os.environ.get("QWEN35_SOURCE", "/mnt/nas1/hf/Qwen3.5-9B")
OUTPUT = os.environ.get("QWEN35_W4A16", "/mnt/nas1/hf/Qwen3.5-9B-W4A16")

texts = []
for name in ("README.md", "chat_template.jinja"):
    path = os.path.join(MODEL, name)
    if os.path.exists(path):
        text = open(path, encoding="utf-8", errors="ignore").read()
        texts.extend(text[i : i + 3000] for i in range(0, len(text), 3000))
texts = (texts * 512)[:256]
dataset = Dataset.from_dict({"text": texts})

oneshot(
    model=MODEL,
    dataset=dataset,
    recipe=GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=["lm_head", "re:.*embed.*"],
    ),
    num_calibration_samples=128,
    max_seq_length=2048,
    trust_remote_code_model=True,
    save_compressed=True,
    output_dir=OUTPUT,
)
print("wrote", OUTPUT)
