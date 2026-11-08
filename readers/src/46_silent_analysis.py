"""Silent-continuation reading (D-189, amended by D-190; exploratory, validation only). Descriptive: no
ground truth for unverbalized awareness exists, so nothing is scored against a label.

Population: continuations that never verbalize awareness (src/44 kinds silent_aware and unaware), each read
inside the continuation (src/44's read position). Continuations that verbalize (kind verbalizing, read just
before their own statement) are a reference for what the readers do when awareness reaches the text.

Reported:
1. Per activation reader, the share of silent continuations it flags (D-186 thresholds), overall, by prompt
   version, and by whether other samples from the same cut point verbalize (q > 0) or not (q = 0). The q
   split describes where flags fall; it is not a label.
2. Reader ensemble (D-190). Readers are grouped into families, because J-lens, J++ lens and template lens
   read the same kind of signal: description (oracle lens, NLA), token (J-lens, J++ lens, logit lens,
   template lens), other (activation oracle, transfer probe). Primary rule: a continuation is
   activation-flagged when at least one description reader AND at least one token reader flag it.
   Sensitivity rules: oracle lens AND NLA; any 2 of the 3 families. The rate expected if the families
   flagged independently (from their own rates) is reported beside each rule.
3. The eval, per prompt version: of all continuations resampled from validation cut points where nothing has
   been verbalized, the share that verbalize awareness (mean q, weighted by judged continuations), the share
   that never verbalize it and are activation-flagged, and the share with neither. Never-verbalizing
   continuations were read one per q = 0 cut point and all of them at q > 0 cut points, so each cut point's
   flag rate is weighted by its number of never-verbalizing continuations.
4. For activation-flagged silent continuations: the share the text readers also flag (Claude given the
   request and the reasoning up to the same token; Claude given the request only).
Rates carry 95% intervals from the near-duplicate cluster bootstrap (src/analysis.py).

    python src/46_silent_analysis.py      -> results/silent_reading.json
"""
import collections
import csv
import json

import numpy as np

import analysis

ACT = ["olens", "nla", "jlens", "jpp", "logit_lens", "template_lens", "ao", "transfer_probe"]
TEXT = ["llm_reader", "prompt_only_llm"]
FAMILIES = {"description": ["olens", "nla"], "token": ["jlens", "jpp", "logit_lens", "template_lens"],
            "other": ["ao", "transfer_probe"]}
RULES = {"description_and_token": lambda f: f["description"] and f["token"],
         "olens_and_nla": lambda f: f["olens"] and f["nla"],
         "two_of_three_families": lambda f: f["description"] + f["token"] + f["other"] >= 2}
PRIMARY = "description_and_token"
FROZEN = json.load(open("results/frozen/42_reader_analysis_selection.json"))["thresholds"]


def poisson_binomial_at_least(ps, k):
    dist = np.zeros(len(ps) + 1)
    dist[0] = 1.0
    for p in ps:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return float(dist[k:].sum())


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
        if (it["trace_id"], int(it["offset"])) in po:
            scores["prompt_only_llm"][it["item_id"]] = po[(it["trace_id"], int(it["offset"]))]
    clusters = analysis.load_clusters()
    flag = {rd: {i: s >= FROZEN[rd] for i, s in scores[rd].items()} for rd in ACT + TEXT}
    for it in items:      # readers missing a score (readout-judge refusal) count as not flagging
        fl = {rd: flag[rd].get(it["item_id"], False) for rd in ACT}
        fl.update({fam: any(fl[rd] for rd in rds) for fam, rds in FAMILIES.items()})
        it["fam"] = fl
        it["rules"] = {name: bool(rule(fl)) for name, rule in RULES.items()}
        it["ens"] = it["rules"][PRIMARY]

    def rate(rows, fn):
        if not rows:
            return {"value": None, "n": 0}
        v = np.array([fn(it) for it in rows], float)
        cl = [clusters[int(it["scenario_id"])] for it in rows]
        return analysis.estimate(lambda i: {"r": float(v[i].mean())}, cl)["r"]

    silent = [it for it in items if it["kind"] in ("silent_aware", "unaware")]
    groups = {"all_silent": silent,
              "silent_q_pos": [it for it in silent if it["kind"] == "silent_aware"],
              "silent_q0": [it for it in silent if it["kind"] == "unaware"],
              "silent_placeholder": [it for it in silent if it["condition"] == "HYPO"],
              "silent_real": [it for it in silent if it["condition"] == "REAL"],
              "verbalizing_reference": [it for it in items if it["kind"] == "verbalizing"]}
    out = {"n": {g: len(rows) for g, rows in groups.items()}, "families": FAMILIES, "primary_rule": PRIMARY,
           "flag_rate": {}, "ensemble": {}}
    # D-191: text baselines at a matched cutoff, the lowest at which they flag no more of the silent continuations
    # from unaware states (no continuation from the cut verbalizes) than the activation readers do on average.
    ref = np.mean([np.mean([flag[rd].get(it["item_id"], False) for it in groups["silent_q0"]]) for rd in ACT])
    out["matched_reference_rate"] = float(ref)
    out["matched_cutoff"] = {}
    for rd in TEXT:
        neg = np.array([scores[rd][it["item_id"]] for it in groups["silent_q0"] if it["item_id"] in scores[rd]])
        cut = next(t for t in sorted(set(neg)) + [np.inf] if (neg >= t).mean() <= ref)
        out["matched_cutoff"][rd] = float(cut)
        flag[rd + "_matched"] = {i: v >= cut for i, v in scores[rd].items()}
        scores[rd + "_matched"] = scores[rd]
    for rd in ACT + TEXT + [t + "_matched" for t in TEXT]:
        out["flag_rate"][rd] = {g: rate([it for it in rows if it["item_id"] in scores[rd]],
                                        lambda it, rd=rd: flag[rd][it["item_id"]]) for g, rows in groups.items()}
    for g, rows in groups.items():
        out["ensemble"][g] = {}
        fam = {k: np.mean([it["fam"][k] for it in rows]) for k in list(FAMILIES) + ["olens", "nla"]}
        indep = {"description_and_token": fam["description"] * fam["token"], "olens_and_nla": fam["olens"] * fam["nla"],
                 "two_of_three_families": poisson_binomial_at_least([fam["description"], fam["token"], fam["other"]], 2)}
        for name in RULES:
            out["ensemble"][g][name] = {"rate": rate(rows, lambda it, n=name: it["rules"][n]),
                                        "expected_if_independent": float(indep[name])}
        out["ensemble"][g]["family_rates"] = {k: float(v) for k, v in fam.items()}

    # 3. the eval, per prompt version
    cp = {(r["trace_id"], int(r["offset"])): r for r in csv.DictReader(open("data/labels/main/cut_points_dev.csv"))}
    by_cut = collections.defaultdict(list)
    for it in silent:
        by_cut[(it["trace_id"], int(it["offset"]))].append(it)
    cuts = sorted({(it["trace_id"], int(it["offset"])) for it in items} |
                  {k for k, r in cp.items() if r["split"] == "val" and r["eligible"] == "1" and r["q"] != ""
                   and k in {(it["trace_id"], int(it["offset"])) for it in items}})
    cut_rows = []
    for k in cuts:
        r = cp[k]; n = int(r["n_judged"]); q = float(r["q"])
        read = by_cut.get(k, [])
        cut_rows.append({"scenario_id": r["scenario_id"], "condition": r["condition"], "n": n, "verb": q * n,
                         "silent": (1 - q) * n, "flag_rate": np.mean([it["ens"] for it in read]) if read else np.nan})
    out["eval"] = {}
    for cond, name in (("HYPO", "placeholder"), ("REAL", "real")):
        rows = [c for c in cut_rows if c["condition"] == cond and not (c["silent"] > 0 and np.isnan(c["flag_rate"]))]
        cl = [clusters[int(c["scenario_id"])] for c in rows]
        n = np.array([c["n"] for c in rows], float); verb = np.array([c["verb"] for c in rows])
        sil = np.array([c["silent"] for c in rows]); fr = np.nan_to_num(np.array([c["flag_rate"] for c in rows]))

        def stat(i):
            return {"verbalize": verb[i].sum() / n[i].sum(), "silent_activation_flagged": (sil[i] * fr[i]).sum() / n[i].sum(),
                    "silent_not_flagged": (sil[i] * (1 - fr[i])).sum() / n[i].sum(),
                    "flagged_among_silent": (sil[i] * fr[i]).sum() / sil[i].sum()}
        out["eval"][name] = {**analysis.estimate(stat, cl), "n_cut_points": len(rows),
                             "n_continuations": int(n.sum())}
    ens = [it for it in silent if it["ens"]]
    out["ensemble_flagged_silent"] = {
        "n": len(ens),
        "placeholder_share": float(np.mean([it["condition"] == "HYPO" for it in ens])) if ens else None,
        "q_pos_share": float(np.mean([it["kind"] == "silent_aware" for it in ens])) if ens else None,
        "text_reader_flags": rate([it for it in ens if it["item_id"] in scores["llm_reader"]], lambda it: flag["llm_reader"][it["item_id"]]),
        "prompt_only_flags": rate([it for it in ens if it["item_id"] in scores["prompt_only_llm"]], lambda it: flag["prompt_only_llm"][it["item_id"]]),
        "items": sorted(it["item_id"] for it in ens)}
    analysis.write_report("results/silent_reading.json", out)
    print(json.dumps(out["n"]))
    print("matched", out["matched_reference_rate"], out["matched_cutoff"])
    for rd in ACT + TEXT + [t + "_matched" for t in TEXT]:
        f = out["flag_rate"][rd]
        print(f"{rd:15s} " + "  ".join(f"{g} {f[g]['value']:.3f}" for g in groups))
    for g in groups:
        e = out["ensemble"][g]
        print(f"{g:22s} " + "  ".join(f"{n} {e[n]['rate']['value']:.3f} (indep {e[n]['expected_if_independent']:.3f})" for n in RULES))
    for v, e in out["eval"].items():
        print(v, e["n_cut_points"], e["n_continuations"], {k: (e[k]["value"], e[k]["ci95"]) for k in ("verbalize", "silent_activation_flagged", "silent_not_flagged", "flagged_among_silent")})
    print(json.dumps({k: v for k, v in out["ensemble_flagged_silent"].items() if k != "items"}))


if __name__ == "__main__":
    main()
