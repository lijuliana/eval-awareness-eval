"""Shared pieces of the activation readers (PROPOSAL 4.2, src/READERS_CONTRACT.md, decisions D-172 to D-177).

Layer convention used everywhere in the reader scripts: **block L** = the output of decoder block L
(0-indexed), before the final norm = HF `hidden_states[L + 1]`. This is the convention of the oracle lens,
the NLA, the activation oracle, the transfer probe and the J-lens files. (src/04's stored "layer 42" is
`hidden_states[42]` = block 41, which is why the readers recompute their activations, D-172.)
"""
import json
from pathlib import Path

import torch

QWEN = "Qwen/Qwen3.6-27B"
QWEN_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"

# Blocks stored per cut point (last token before the cut). 42 is every reader's primary block (D-173);
# 16/32/48 are the activation oracle's trained layers; the rest serve the lenses' secondary layer curves.
BLOCKS = [16, 20, 24, 28, 32, 36, 40, 42, 44, 48, 52, 56, 60]
PRIMARY_BLOCK = 42
AO_BLOCK = 48           # activation oracle: trained only on blocks 16, 32, 48; 48 is nearest the others' 42
LENS_TOP_K = 20         # tokens / phrases shown to the readout judge from each single-token or phrase lens

ACT_DIR = Path("data/activations/main/readers")
OUT_DIR = Path("data/generations/main/activation_readers")
SCORE_FILES = {"dev": Path("data/labels/main/activation_readers_dev.csv"),
               "locked": Path("data/labels/main/activation_readers_locked_test.csv")}
READER_CSVS = {"dev": Path("data/labels/main/llm_readers_dev.csv"),
               "locked": Path("data/labels/main/llm_readers_locked_test.csv")}


def target_keys(part):
    """(trace_id, offset) of every cut point the LLM text readers score in this part (dev = val,
    locked = test_id), so the activation readers are scored on exactly the same items (paired)."""
    import csv
    with READER_CSVS[part].open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["reader"] == "llm_reader"]
    return sorted({(r["trace_id"], int(r["offset"])) for r in rows})


def decoder_blocks(model):
    """The decoder's block list, for a plain CausalLM or a PEFT-wrapped one."""
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and hasattr(module, "norm"):
            return layers
    raise ValueError("no decoder layers found")


def load_qwen(path=QWEN, revision=QWEN_REVISION, device="cuda"):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, revision=revision, dtype=torch.bfloat16,
                                                 device_map={"": device})
    model.requires_grad_(False)
    return model.eval()


def load_activations(part):
    """{(trace_id, offset): {block: float32 tensor [5120]}} from src/32's file for this part."""
    import numpy as np
    with np.load(ACT_DIR / f"{part}.npz") as data:
        blocks = [int(b) for b in data["blocks"]]
        acts = torch.from_numpy(data["last"])          # [N, len(blocks), 5120]
        keys = list(zip([str(t) for t in data["trace_id"]], [int(o) for o in data["offset"]]))
    return {key: {b: acts[i, j] for j, b in enumerate(blocks)} for i, key in enumerate(keys)}


def add_norm_matched_hook(model, positions, vectors, block=1):
    """Activation-oracle / NLA injection (Karvonen et al. 2025, eq. 1): at each marked position p of the
    prompt, h_p <- h_p + ||h_p|| * v / ||v|| on the output of decoder block `block`. Only the full-prompt
    forward is changed (decode steps have sequence length 1). Returns the hook handle."""
    def hook(_module, _inputs, output):
        resid = output[0] if isinstance(output, tuple) else output
        if resid.shape[1] < 2:
            return output
        resid = resid.clone()
        for b, (pos, vec) in enumerate(zip(positions, vectors)):
            h = resid[b, pos]
            v = vec.to(resid.device, torch.float32)
            resid[b, pos] = (h.float() + h.float().norm() * v / v.norm().clamp_min(1e-9)).to(resid.dtype)
        return (resid, *output[1:]) if isinstance(output, tuple) else resid
    return decoder_blocks(model)[block].register_forward_hook(hook)


def write_outputs(reader, part, rows, provenance):
    """Raw outputs (READERS_CONTRACT 'What to hand back' 1) plus a provenance note."""
    folder = OUT_DIR / reader / part
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "outputs.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=1))


def write_scores(part, reader, scores):
    """Merge one reader's scores into the part's CSV (columns trace_id, offset, reader, status, score),
    replacing that reader's earlier rows."""
    import csv
    path = SCORE_FILES[part]
    rows = []
    if path.exists():
        with path.open() as handle:
            rows = [r for r in csv.DictReader(handle) if r["reader"] != reader]
    rows += [{"trace_id": t, "offset": o, "reader": reader, "status": s, "score": "" if v is None else v}
             for (t, o), (s, v) in sorted(scores.items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trace_id", "offset", "reader", "status", "score"])
        writer.writeheader()
        writer.writerows(rows)


def jacobian(blob, block):
    """J for one source block from a J-lens / J++ file: either the official dict {block: J} (jlens
    JacobianLens.save) or one stacked tensor whose rows follow blob["source_layers"]."""
    J = blob["J"]
    if isinstance(J, dict):
        return J[block].float()
    return J[list(blob["source_layers"]).index(block)].float()
