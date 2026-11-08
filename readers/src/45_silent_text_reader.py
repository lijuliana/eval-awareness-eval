"""Text baseline for the silent-continuation reading (D-189): src/26's frozen llm_reader (Opus 5.5, effort
high, same template and few-shot examples) given the request and the reasoning up to each item's read
position (prefix + generated[:read_pos]), i.e. the same text the activation readers' token closes.

    python src/45_silent_text_reader.py            # dry run: counts and cost
    python src/45_silent_text_reader.py --submit   # send (Batch API), wait, write the scores
Output: data/labels/main/llm_reader_silent_dev.csv (item_id, status, q_pred). Validation only.
"""
import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import common
import labels_common as lc

spec = importlib.util.spec_from_file_location("r26", "src/26_llm_readers.py")
r26 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r26)
ROOT = Path("data/generations/main/readers_silent")
OUT = Path("data/labels/main/llm_reader_silent_dev.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = common.select_model(common.load_config("src/configs/main.yaml"), "base")
    tok = common.load_tokenizer(config)
    prompts = lc.load_prompts("data/main/prompts.json")
    examples = json.loads(Path("data/labels/main/fewshot_ids.json").read_text())["examples"]
    ex_texts = r26.prefix_texts("data/generations/main", [(e["trace_id"], e["offset"]) for e in examples], tok, prompts)
    template = Path(r26.READERS["llm_reader"]).read_text()
    system = template.replace("{examples}", r26.example_block("llm_reader", examples, ex_texts))

    items = list(csv.DictReader(open("data/labels/main/silent_items_dev.csv")))
    wanted = {it["continuation_id"] for it in items}
    conts = {c["continuation_id"]: c for c in lc.iter_records(Path("data/generations/main/base"), "continuations")
             if c["split"] == "val" and c["continuation_id"] in wanted}
    reqs = []
    for it in items:
        c = conts[it["continuation_id"]]
        text = tok.decode(c["prefix_ids"] + c["generated_ids"][: int(it["read_pos"])], skip_special_tokens=True)
        prompt = prompts[(c["scenario_id"], c["version"], c["condition"])]
        reqs.append({"custom_id": f"llm_reader-{it['item_id']}"[:64], "kind": "llm_reader", "id": it["item_id"],
                     "split": "val", "condition": c["condition"], "model": "base",
                     "user": f"<request>\n{prompt}\n</request>\n\n<reasoning_so_far>\n{text}\n</reasoning_so_far>"})
    assert len({r["custom_id"] for r in reqs}) == len(reqs), "custom_id collision"
    store, ledger = lc.RowStore(ROOT), lc.Ledger(ROOT)
    todo = [r for r in reqs if not store.is_done(r["custom_id"]) and r["custom_id"] not in ledger.pending()][: args.limit]
    print(json.dumps({"items": len(reqs), "to_send": len(todo), "prompt_sha256_16": lc.sha16(system),
                      "template_sha256_16": lc.sha16(template)}))
    if todo:
        print("cost estimate:", json.dumps(lc.estimate_cost(r26.READER_MODEL, len(todo), len(system),
                                                             sum(len(r["user"]) for r in todo), r26.EST_OUTPUT_TOKENS, 1)))
    if args.submit:
        client = lc.make_client()

        def store_result(custom_id, result, batch_id, route):
            fields = lc.parse_batch_result(lc.as_dict(result), r26.parse_reader_body)
            if "labels" in fields:
                fields["prediction"] = fields.pop("labels")
            store.append({**route, "custom_id": custom_id, "batch_id": batch_id, "reader_model": r26.READER_MODEL,
                          "prompt_sha256_16": lc.sha16(system),
                          "stored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields})
            return fields["status"]

        ROOT.mkdir(parents=True, exist_ok=True)
        lc.collect(client, ledger, store_result)
        if todo:
            requests = [r26.reader_request(r, system, "high", 16000) for r in todo]
            routes = {r["custom_id"]: {k: r[k] for k in ("id", "kind", "model", "split", "condition")} for r in todo}
            lc.submit(client, ledger, requests, routes)
            lc.collect(client, ledger, store_result)
    rows = []
    for r in reqs:
        row = store.latest.get(r["custom_id"])
        status = r26.STATUS[row["status"]] if row else "missing"
        rows.append({"item_id": r["id"], "status": status, "q_pred": row.get("prediction", "") if row and status == "ok" else ""})
    with OUT.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["item_id", "status", "q_pred"])
        w.writeheader()
        w.writerows(rows)
    print("written", OUT, json.dumps({s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}}))


if __name__ == "__main__":
    main()
