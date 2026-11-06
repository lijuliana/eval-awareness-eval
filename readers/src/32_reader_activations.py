"""Recompute the activations the activation readers read (READERS_CONTRACT "Which activation to read").

For every cut point the LLM text readers score (dev = val, locked = test_id), one teacher-forced pass of
base Qwen3.6-27B over prompt + the first `offset` generated tokens, storing the residual of the **last
token before the cut** at the blocks in readers_common.BLOCKS (block L = hidden_states[L + 1]).

Check before anything is read (dev only, test lock): the model's log-probability of each prefix token
must match what vLLM reported when the trace was generated (median absolute difference <= 0.1 nats, as
gate G2). This confirms the right model, tokens and positions without the activations stored by src/04.

    python src/32_reader_activations.py --part dev
    python src/32_reader_activations.py --part locked     # test_id: outputs only, nothing printed about them
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import common
import readers_common as rc

G2_MAX_MEDIAN = 0.1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["dev", "locked"], required=True)
    parser.add_argument("--gen-dir", default="data/generations/main/base")
    parser.add_argument("--limit", type=int, default=None, help="first N cut points only (smoke test)")
    args = parser.parse_args()

    keys = rc.target_keys(args.part)[: args.limit]
    wanted = {t for t, _ in keys}
    traces = {r["trace_id"]: r for r in common.read_records(args.gen_dir, "traces") if r["trace_id"] in wanted}
    missing = wanted - traces.keys()
    if missing:
        raise SystemExit(f"{len(missing)} target traces not found, e.g. {sorted(missing)[:3]}")

    model = rc.load_qwen()
    started, last, g2 = time.time(), [], []
    for n, (trace_id, offset) in enumerate(keys):
        trace = traces[trace_id]
        prompt, generated = trace["prompt_ids"], trace["generated_ids"][:offset]
        ids = torch.tensor([prompt + generated], device="cuda")
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        pos = len(prompt) + offset - 1          # the last token before the cut
        last.append(torch.stack([out.hidden_states[b + 1][0, pos] for b in rc.BLOCKS]).float().cpu().numpy())
        if args.part == "dev":
            logits = out.logits[0, len(prompt) - 1: len(prompt) + offset - 1].float()
            lp = torch.log_softmax(logits, -1)[torch.arange(offset), torch.tensor(generated, device="cuda")]
            diff = np.abs(lp.cpu().numpy() - np.array(trace["generated_logprobs"][:offset]))
            g2.append(float(np.median(diff)))
        if (n + 1) % 100 == 0:
            print(f"{n + 1}/{len(keys)} cut points, {time.time() - started:.0f}s", flush=True)

    rc.ACT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(rc.ACT_DIR / f"{args.part}.npz", trace_id=np.array([k[0] for k in keys]),
             offset=np.array([k[1] for k in keys]), blocks=np.array(rc.BLOCKS), last=np.stack(last))
    meta = common.provenance({"model": rc.QWEN, "revision": rc.QWEN_REVISION, "blocks": rc.BLOCKS,
                              "position": "last token before the cut", "part": args.part,
                              "n": len(keys), "args": vars(args)}, "32_reader_activations")
    meta["seconds"] = time.time() - started
    if args.part == "dev":
        meta["g2"] = {"median_of_medians": float(np.median(g2)), "max_median": float(np.max(g2)),
                      "share_over_limit": float(np.mean(np.array(g2) > G2_MAX_MEDIAN)),
                      "passed": bool(np.max(g2) <= G2_MAX_MEDIAN)}
        print("G2 check:", json.dumps(meta["g2"]))
    (rc.ACT_DIR / f"{args.part}_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {len(keys)} cut points x {len(rc.BLOCKS)} blocks to {rc.ACT_DIR / (args.part + '.npz')}")


if __name__ == "__main__":
    main()
