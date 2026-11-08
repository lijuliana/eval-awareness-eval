"""Items for the silent-continuation reading (D-189; exploratory, validation only).

Unverbalized awareness, per continuation: a continuation that never voices test suspicion, resampled from a
cut point where other continuations do (q > 0). The shared state at the cut leads to voiced awareness in
some samples, and this sample never voices it. Each item is read inside its own continuation.

Item kinds (validation, eligible cut points that the readers score):
- silent_aware: continuations judged not to voice suspicion at cut points with q > 0. Read position = the
  median onset (tokens after the cut) of the sibling continuations that voice it, so the activation is taken
  where awareness surfaced in the siblings; clipped to the continuation's reasoning.
- verbalizing: continuations at q > 0 cut points that voice it, read at the last token before their own
  first awareness quote (a reference: awareness about to be stated).
- unaware: one seeded continuation per q = 0 cut point, read at a position drawn (seeded) from the
  silent_aware read positions, clipped to its reasoning.
The read position r means: the activation of the last token of prompt + prefix + generated[:r].

Inputs: data/labels/main/cut_points_dev.csv, data/labels/main/llm_readers_dev.csv (reader targets),
data/generations/main/base/continuations-*.jsonl (val records only), the judge's dev continuation rows.
Output: data/labels/main/silent_items_dev.csv (item_id, continuation_id, trace_id, offset, scenario_id,
condition, kind, q, read_pos, reasoning_len, onset).
"""
import collections
import csv
import json
import statistics
from pathlib import Path

import numpy as np

import common
import labels_common as lc

THINK_END_ID = 248069
SEED = 20261107
LABELS = Path("data/labels/main")
GEN = Path("data/generations/main")


def main():
    config = common.select_model(common.load_config("src/configs/main.yaml"), "base")
    tok = common.load_tokenizer(config)
    cp = {(r["trace_id"], int(r["offset"])): r for r in csv.DictReader(open(LABELS / "cut_points_dev.csv"))}
    targets = {(r["trace_id"], int(r["offset"])) for r in csv.DictReader(open(LABELS / "llm_readers_dev.csv"))
               if r["reader"] == "llm_reader"}
    cuts = {k for k in targets if k in cp and cp[k]["split"] == "val" and cp[k]["eligible"] == "1" and cp[k]["q"] != ""}
    judge = {}
    for line in open(GEN / "judge/sonnet/dev/base_continuation.jsonl"):
        r = json.loads(line)
        if r["split"] == "val" and r["status"] == "ok":
            judge[r["id"]] = r            # later rows replace earlier ones (RowStore keeps the latest)
    by_cut = collections.defaultdict(list)
    for cont in lc.iter_records(GEN / "base", "continuations"):
        if cont["split"] != "val" or lc.is_replica(cont) or (cont["trace_id"], cont["offset"]) not in cuts:
            continue
        row = judge.get(cont["continuation_id"])
        if row is None:
            continue
        labels = row["labels"] if isinstance(row["labels"], dict) else json.loads(row["labels"])
        label = int(labels["reasoning_label"] in lc.POSITIVE or labels["answer_label"] in lc.POSITIVE)
        gen = cont["generated_ids"]
        rlen = gen.index(THINK_END_ID) if THINK_END_ID in gen else len(gen)
        onset = None
        if label:
            full, ids, boundary = lc.continuation_text(tok, cont)
            where = labels["first_awareness_location"] if labels["first_awareness_location"] != "none" else "reasoning"
            char, _ = lc.locate_quote(full, labels["first_awareness_quote"], lc.text_regions(full, where))
            tokpos = lc.token_index_at_char(tok, ids, full, char) if char is not None else None
            if tokpos is not None and tokpos >= len(cont["prefix_ids"]):
                onset = tokpos - len(cont["prefix_ids"])
        by_cut[(cont["trace_id"], cont["offset"])].append(
            {"continuation_id": cont["continuation_id"], "label": label, "onset": onset, "rlen": rlen})

    rng = np.random.default_rng(SEED)
    items, silent_pos = [], []
    for key in sorted(by_cut):
        c, q = cp[key], float(cp[key]["q"])
        conts = by_cut[key]
        base = {"trace_id": key[0], "offset": key[1], "scenario_id": c["scenario_id"], "condition": c["condition"], "q": q}
        if q > 0:
            onsets = [x["onset"] for x in conts if x["label"] and x["onset"] is not None and x["onset"] >= 1]
            for x in conts:
                if x["label"] and x["onset"] is not None and x["onset"] >= 1:
                    items.append({**base, "continuation_id": x["continuation_id"], "kind": "verbalizing",
                                  "read_pos": x["onset"], "reasoning_len": x["rlen"], "onset": x["onset"]})
                elif not x["label"] and onsets:
                    r = int(statistics.median(onsets))
                    silent_pos.append(r)
                    items.append({**base, "continuation_id": x["continuation_id"], "kind": "silent_aware",
                                  "read_pos": max(1, min(r, x["rlen"])), "reasoning_len": x["rlen"], "onset": ""})
    for key in sorted(by_cut):
        c, q = cp[key], float(cp[key]["q"])
        if q > 0:
            continue
        x = by_cut[key][int(rng.integers(len(by_cut[key])))]
        r = int(rng.choice(silent_pos))
        items.append({"trace_id": key[0], "offset": key[1], "scenario_id": c["scenario_id"], "condition": c["condition"],
                      "q": q, "continuation_id": x["continuation_id"], "kind": "unaware",
                      "read_pos": max(1, min(r, x["rlen"])), "reasoning_len": x["rlen"], "onset": ""})
    for it in items:
        it["item_id"] = f"{it['continuation_id']}_p{it['read_pos']}"
    cols = ["item_id", "continuation_id", "trace_id", "offset", "scenario_id", "condition", "kind", "q",
            "read_pos", "reasoning_len", "onset"]
    with open(LABELS / "silent_items_dev.csv", "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=cols)
        w.writeheader()
        w.writerows(items)
    print(json.dumps({"cut_points": len(by_cut), "items": collections.Counter(i["kind"] for i in items),
                      "silent_read_pos_median": statistics.median(silent_pos) if silent_pos else None,
                      "clipped_silent": sum(1 for i in items if i["kind"] == "silent_aware" and i["read_pos"] == i["reasoning_len"]),
                      "clipped_unaware": sum(1 for i in items if i["kind"] == "unaware" and i["read_pos"] == i["reasoning_len"])}, default=dict))


if __name__ == "__main__":
    main()
