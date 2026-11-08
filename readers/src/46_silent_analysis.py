"""Silent-continuation reading (D-189; exploratory, validation only): do readers separate continuations that
never voice awareness but come from an aware state (silent_aware, cut point q > 0) from continuations of
unaware states (unaware, q = 0), read at matched positions inside the continuation?

Per reader: AUROC silent_aware vs unaware, and flag rates (D-186 thresholds) for silent_aware, unaware and
verbalizing (read just before the continuation's own first awareness quote). Text baselines: the frozen
llm_reader given the same text (src/45) and the prompt-only reader (one score per prompt). Paired AUROC
difference of each activation reader minus the text reader. Then the part the text does not show: AUROC on
items whose text-reader score is below its flag threshold. 95% intervals from the near-duplicate cluster
bootstrap (src/analysis.py).

    python src/46_silent_analysis.py      -> results/silent_reading.json
"""
import collections
import csv
import json

import numpy as np

import analysis

WORD = ["olens", "nla", "jlens", "jpp", "logit_lens"]
NUM = ["template_lens", "ao", "transfer_probe"]
FROZEN = json.load(open("results/frozen/42_reader_analysis_selection.json"))["thresholds"]


def main():
    items = list(csv.DictReader(open("data/labels/main/silent_items_dev.csv")))
    scores = collections.defaultdict(dict)
    for r in csv.DictReader(open("data/labels/main/activation_readers_silent_dev.csv")):
        if r["status"] == "ok" and r["score"] != "":
            scores[r["reader"]][r["trace_id"]] = float(r["score"])       # trace_id column holds item_id
    for r in csv.DictReader(open("data/labels/main/llm_reader_silent_dev.csv")):
        if r["status"] == "ok" and r["q_pred"] != "":
            scores["llm_reader"][r["item_id"]] = float(r["q_pred"])
    po = {(r["trace_id"], int(r["offset"])): float(r["q_pred"]) for r in csv.DictReader(open("data/labels/main/llm_readers_dev.csv"))
          if r["reader"] == "prompt_only_llm" and r["status"] == "ok" and r["q_pred"] != ""}
    for it in items:
        k = (it["trace_id"], int(it["offset"]))
        if k in po:
            scores["prompt_only_llm"][it["item_id"]] = po[k]
    clusters = analysis.load_clusters()
    readers = ["llm_reader", "prompt_only_llm"] + WORD + NUM
    kinds = collections.Counter(it["kind"] for it in items)
    out = {"n_items": dict(kinds), "n_cut_points": len({(it["trace_id"], it["offset"]) for it in items}),
           "flag_rates": {}, "auroc": {}, "auroc_minus_text": {}, "auroc_text_quiet": {}}

    for rd in readers:
        thr = FROZEN[rd]
        out["flag_rates"][rd] = {}
        for kind in ["silent_aware", "unaware", "verbalizing"]:
            v = [scores[rd][it["item_id"]] >= thr for it in items if it["kind"] == kind and it["item_id"] in scores[rd]]
            out["flag_rates"][rd][kind] = {"rate": float(np.mean(v)) if v else None, "n": len(v)}

    pair = [it for it in items if it["kind"] in ("silent_aware", "unaware")]

    def table(rows, names):
        ks = [it for it in rows if all(it["item_id"] in scores[n] for n in names)]
        y = np.array([it["kind"] == "silent_aware" for it in ks], int)
        cl = [clusters[int(it["scenario_id"])] for it in ks]
        cols = {n: np.array([scores[n][it["item_id"]] for it in ks]) for n in names}
        return y, cl, cols

    for rd in readers:
        y, cl, cols = table(pair, [rd])
        out["auroc"][rd] = analysis.estimate(lambda i: {"auroc": analysis.auroc(y[i], cols[rd][i])}, cl)["auroc"]
    for rd in WORD + NUM:
        y, cl, cols = table(pair, [rd, "llm_reader"])

        def diff(i):
            a, b = analysis.auroc(y[i], cols[rd][i]), analysis.auroc(y[i], cols["llm_reader"][i])
            return {"d": None if a is None or b is None else a - b}
        out["auroc_minus_text"][rd] = analysis.estimate(diff, cl)["d"]
    quiet = [it for it in pair if it["item_id"] in scores["llm_reader"] and scores["llm_reader"][it["item_id"]] < FROZEN["llm_reader"]]
    out["n_text_quiet"] = dict(collections.Counter(it["kind"] for it in quiet))
    for rd in WORD + NUM + ["prompt_only_llm"]:
        y, cl, cols = table(quiet, [rd])
        out["auroc_text_quiet"][rd] = analysis.estimate(lambda i: {"auroc": analysis.auroc(y[i], cols[rd][i])}, cl)["auroc"]
    analysis.write_report("results/silent_reading.json", out)
    print(json.dumps(out["n_items"]), "text-quiet", json.dumps(out["n_text_quiet"]))
    for rd in readers:
        f = out["flag_rates"][rd]
        a = out["auroc"][rd]
        line = (f"{rd:15s} AUROC {a['value']} {a['ci95']}  flags silent {f['silent_aware']['rate']:.3f} "
                f"unaware {f['unaware']['rate']:.3f} verbalizing {f['verbalizing']['rate']:.3f}")
        if rd in out["auroc_minus_text"]:
            d, q = out["auroc_minus_text"][rd], out["auroc_text_quiet"][rd]
            line += f"  minus text {d['value']} {d['ci95']}  text-quiet AUROC {q['value']} {q['ci95']}"
        print(line)


if __name__ == "__main__":
    main()
