"""Readout judge (PROPOSAL 4.2; D-163(5)): one Claude call scores each word-producing reader's output for
"the model is considering that it is being tested", 0 to 1. The judge sees only the readout, never the
prompt or the trace (prompts/readout_judge_v1.md). Same model for every reader (Sonnet 5.5), so H4 compares
readers, not scoring methods (B9).

Readers scored: logit_lens, jlens, jpp (top-20 tokens with probabilities), olens (its 4 readouts in one
call), nla (its explanation). One call per (reader, cut point).

Batch, dry-run, resume and locked-output design as src/26: dry run by default (counts + cost); `--submit`
spends; `--collect` stores finished batches; a rerun never re-sends an item with a stored row. `--part
locked` sends test_id readouts once and logs the access; their scores go only to the locked CSV.

    python src/35_readout_judge.py --part dev                  # dry run
    python src/35_readout_judge.py --part dev --submit
"""
import argparse
import json
import time
from pathlib import Path

import common
import labels_common as lc
import readers_common as rc

JUDGE_MODEL = "claude-sonnet-5-5"
PROMPT = Path("prompts/readout_judge_v1.md")
WORD_READERS = ["logit_lens", "jlens", "jpp", "olens", "nla"]
SCHEMA = {"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"],
          "additionalProperties": False}
ROOT = Path("data/generations/main/readout_judge")


def readout_text(reader, raw):
    if reader in ("logit_lens", "jlens", "jpp"):
        return "Top tokens (probability): " + ", ".join(f"{t!r} ({p:.3f})" for t, p in raw["top_tokens"])
    if reader == "olens":
        return "\n\n".join(f"Readout {i + 1}:\n{r.strip()}" for i, r in enumerate(raw["readouts"]))
    return raw["explanation"].strip()


def parse_body(body):
    try:
        value = json.loads(body)["score"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["dev", "locked"], required=True)
    parser.add_argument("--readers", nargs="+", default=WORD_READERS)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2000)
    args = parser.parse_args()
    system = PROMPT.read_text()
    split = "val" if args.part == "dev" else "test_id"

    items = []
    for reader in args.readers:
        path = rc.OUT_DIR / reader / args.part / "outputs.jsonl"
        if not path.exists():
            print(f"skip {reader}: {path} missing")
            continue
        for row in common.read_jsonl(path):
            items.append({"custom_id": f"{reader}-{row['trace_id']}_t{row['offset']}"[:64], "reader": reader,
                          "trace_id": row["trace_id"], "offset": row["offset"], "split": split,
                          "user": f"<readout>\n{readout_text(reader, row['raw'])}\n</readout>"})
    items = common.seeded_shuffle(items, "readout_judge", args.part)
    store, ledger = lc.RowStore(ROOT), lc.Ledger(ROOT)
    pending = ledger.pending()
    todo = [it for it in items if not store.is_done(it["custom_id"]) and it["custom_id"] not in pending][: args.limit]
    print(json.dumps({"part": args.part, "items": len(items), "stored": len(store.latest), "to_send": len(todo),
                      "prompt_sha256_16": lc.sha16(system)}))
    if todo:
        print("cost estimate:", json.dumps(lc.estimate_cost(JUDGE_MODEL, len(todo), len(system),
                                                             sum(len(it["user"]) for it in todo), 40, 1)))
    if not (args.submit or args.collect):
        print("dry run: nothing sent. Add --submit to send (costs money).")
        return
    if args.part == "locked" and args.submit and todo:
        common.log_test_access("35_readout_judge", "readout-judge scores of activation readers on test cut points", vars(args))
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "runs.jsonl").open("a") as handle:
        handle.write(json.dumps({**common.provenance({"judge_model": JUDGE_MODEL, "prompt": str(PROMPT),
                                                      "prompt_sha256_16": lc.sha16(system)}, "35_readout_judge"),
                                 "argv": vars(args), "n_to_send": len(todo)}) + "\n")
    client = lc.make_client()

    def store_result(custom_id, result, batch_id, route):
        fields = lc.parse_batch_result(lc.as_dict(result), parse_body)
        if "labels" in fields:
            fields["score"] = fields.pop("labels")
        store.append({**route, "custom_id": custom_id, "batch_id": batch_id, "judge_model": JUDGE_MODEL,
                      "stored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields})
        return fields["status"]

    lc.collect(client, ledger, store_result, wait=not args.no_wait)
    if args.submit and todo:
        requests = [{"custom_id": it["custom_id"],
                     "params": {"model": JUDGE_MODEL, "max_tokens": args.max_tokens,
                                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                                "messages": [{"role": "user", "content": it["user"]}],
                                "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}}}
                    for it in todo]
        routes = {it["custom_id"]: {"id": f"{it['trace_id']}_t{it['offset']}", "kind": it["reader"], "model": "base",
                                    "split": it["split"], "trace_id": it["trace_id"], "offset": it["offset"]}
                  for it in todo}
        lc.submit(client, ledger, requests, routes)
        lc.collect(client, ledger, store_result, wait=not args.no_wait)

    for reader in args.readers:
        rows = {(r["trace_id"], r["offset"]): r for r in store.latest.values()
                if r.get("kind") == reader and r.get("split") == split}
        if rows:
            rc.write_scores(args.part, reader, {k: (r["status"], r.get("score") if r["status"] == "ok" else None)
                                                 for k, r in rows.items()})


if __name__ == "__main__":
    main()
