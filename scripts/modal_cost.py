#!/usr/bin/env python3
"""Per-GPU slot count and cost estimate for scanning an export on Modal.

    python scripts/modal_cost.py qwen3:32b export.zip

Model size and KV cache per token come from the GGUF header in the Ollama
registry, GPU prices from modal.com/pricing, and the export's shape from
looselips' own chunker. Needs `pip install gguf` (not a looselips dependency).

The time model is crude: prefill is compute-bound (2 FLOPs/param/token at
40% of peak), decode is bandwidth-bound (stream the weights once per token
step, shared across the batch). Trust the ranking, not the absolute numbers.
"""

import json
import re
import sys
import tempfile
import urllib.request

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFReader

NUM_CTX = 16384
OUT_TOKENS = 150        # reasoning + verdict JSON per call
CHARS_PER_TOKEN = 3.5
PROMPT_OVERHEAD = 700   # scanner prompt + matcher instructions, tokens
MAX_SLOTS = 16
HEADER_BYTES = 32 << 20  # GGUF metadata incl. tokenizer vocab (~15 MB for Qwen)
# Fraction of VRAM for weights + KV cache; the rest is compute buffers.
# Same value and meaning as vLLM's default --gpu-memory-utilization.
GPU_MEMORY_UTILIZATION = 0.92

# first word of the name on modal.com/pricing -> (VRAM GB, TB/s, bf16 TFLOPS)
GPUS = {
    "T4": (16, 0.32, 65), "L4": (24, 0.30, 121), "A10": (24, 0.60, 125),
    "L40S": (48, 0.86, 362), "RTX": (96, 1.8, 500), "H100": (80, 3.35, 989),
    "H200": (141, 4.8, 989), "B200": (192, 8.0, 2250), "B300": (288, 8.0, 2250),
}


def fetch(url: str, first_bytes: int = 0) -> bytes:
    headers = {"User-Agent": "myapp/1.0"}
    if first_bytes:
        headers["Range"] = f"bytes=0-{first_bytes - 1}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


class HeaderReader(GGUFReader):
    """GGUFReader over a truncated file: sum tensor descriptors, skip data."""

    def _build_tensors(self, start_offs, fields):  # type: ignore[no-untyped-def]
        self.n_params = self.n_bytes = 0
        for f in fields:
            n = int(np.prod(np.array(f.parts[3], dtype=np.uint64)))
            block, size = GGML_QUANT_SIZES[GGMLQuantizationType(f.parts[4][0])]
            self.n_params += n
            self.n_bytes += n // block * size
        self.tensors = []

    def get(self, key: str, default: int | None = None) -> int:
        f = self.fields.get(key)
        if f is None:
            if default is None:
                raise KeyError(key)
            return default
        return int(f.parts[f.data[0]][0])


def model_size(tag: str) -> tuple[int, int, int]:
    """(weight bytes, params, fp16 KV cache bytes per token) from the GGUF header."""
    name, _, version = tag.partition(":")
    registry = f"https://registry.ollama.ai/v2/library/{name}"
    manifest = json.loads(fetch(f"{registry}/manifests/{version or 'latest'}"))
    digest = next(x["digest"] for x in manifest["layers"]
                  if x["mediaType"].endswith(".model"))
    with tempfile.NamedTemporaryFile(suffix=".gguf") as tmp:
        tmp.write(fetch(f"{registry}/blobs/{digest}", HEADER_BYTES))
        tmp.flush()
        r = HeaderReader(tmp.name)
        arch = r.fields["general.architecture"].parts[-1].tobytes().decode()
        a = f"{arch}.attention."
        # key_length is optional in GGUF (Llama files omit it); the standard
        # default is embedding_length / head_count.
        k = r.get(a + "key_length",
                  r.get(f"{arch}.embedding_length") // r.get(a + "head_count"))
        v = r.get(a + "value_length", k)
        layers, kv_heads = r.get(f"{arch}.block_count"), r.get(a + "head_count_kv")
        kv_per_token = layers * kv_heads * (k + v) * 2
    return r.n_bytes, r.n_params, kv_per_token


def modal_prices() -> dict[str, tuple[str, float]]:
    """First word of GPU name -> (full name, $/hour)."""
    text = re.sub(r"<[^>]+>", " ", fetch("https://modal.com/pricing").decode())
    found = re.findall(r"Nvidia ([A-Z0-9 ]+?)\s+\$([0-9.]+)\s*/\s*sec", text)
    if not found:
        sys.exit("could not find GPU prices on modal.com/pricing")
    return {name.split()[0]: (name, float(p) * 3600) for name, p in found}


def export_shape(path: str) -> tuple[int, float]:
    """(LLM calls, total prompt tokens) for one LLM matcher over the export."""
    from looselips.input import load_conversations
    from looselips.scanner import _chunk_conversation
    chunks = [len(c) for conv in load_conversations(path) if conv.messages
              for c in _chunk_conversation(conv)]
    return len(chunks), sum(chunks) / CHARS_PER_TOKEN + PROMPT_OVERHEAD * len(chunks)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    tag, export = sys.argv[1:]
    weights, params, kv_per_token = model_size(tag)
    calls, prompt_tokens = export_shape(export)
    kv_slot = kv_per_token * NUM_CTX
    print(f"{tag}: {weights / 1e9:.1f} GB weights, {params / 1e9:.1f}B params, "
          f"{kv_slot / 1e9:.2f} GB KV cache per {NUM_CTX}-token slot")
    print(f"{export}: {calls} LLM calls, {prompt_tokens / 1e6:.2f}M prompt tokens "
          f"(per LLM matcher)\n")

    rows, unfit = [], []
    for key, (name, price) in modal_prices().items():
        if key not in GPUS:
            continue
        vram, tbs, tflops = GPUS[key]
        usable = vram * 1e9 * GPU_MEMORY_UTILIZATION - weights
        slots = min(MAX_SLOTS, int(usable / kv_slot))
        if slots < 1:
            unfit.append(name)
            continue
        prefill = prompt_tokens * 2 * params / (tflops * 1e12 * 0.4)
        decode = calls * OUT_TOKENS / slots * weights / (tbs * 1e12)
        hours = (prefill + decode) / 3600
        rows.append((hours * price, name, price, slots, hours))
    print(f"{'GPU':14} {'$/hr':>6} {'slots':>5} {'hours':>6} {'cost':>7}  flags")
    for cost, name, price, slots, hours in sorted(rows):
        print(f"{name:14} {price:6.2f} {slots:5d} {hours:6.1f} {cost:7.2f}  "
              f"--num-parallel {slots}")
    if unfit:
        print("does not fit:", ", ".join(unfit))


if __name__ == "__main__":
    main()
