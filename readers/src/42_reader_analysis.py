"""Activation-reader analysis (H3, H4, level 3; D-182, D-185, D-186), with the src/analysis.py test gate.

Population: eligible cut points with a q, in the readers' targets (src/26 `target_cut_points`, D-172):
validation in development mode, test_id with --test. Offsets 128 and 256 pooled; offset 256 also reported.

1. Every reader against q: Spearman rho and AUROC (q > 0), 95% CI from the near-duplicate cluster bootstrap.
2. H3 (descriptive, OPEN_DECISIONS A3): Spearman matrix among all readers and q, on cut points every reader
   scored. The trained q-probe joins when src/25's per-cut-point predictions exist for the split.
3. H4: rho(single-token lens, q) - rho(verbalizing reader, q) for J-lens and logit lens against NLA, oracle
   lens and activation oracle (6 paired comparisons on shared cut points); CI and two-sided p from the cluster
   bootstrap; Holm over the 6.
4. Level 3 (D-182 as amended by D-185): cut points with q = 0 that the prompt+reasoning text reader does not
   flag. Flag rate on placeholder (HYPO) minus real (REAL) prompts; CI from resampling scenarios, p from
   10,000 within-scenario swaps of the condition labels, Holm over the five word readers. Also the two text
   readers as controls, all q = 0 cuts, and cuts whose reasoning before the cut does not remark that the
   setting looks like a test (D-185).

Flag thresholds: word readers at a readout-judge score of 0.25 or more; number readers (activation oracle,
transfer probe, template lens) and the text readers at the value that flags 10% of validation q = 0 cut
points. Thresholds are computed on validation, frozen by --freeze, and only read back by --test (D-186).

Usage (repo root):
    python src/42_reader_analysis.py            # development: validation only, nothing from test read
    python src/42_reader_analysis.py --freeze   # save thresholds and settings to results/frozen/
    python src/42_reader_analysis.py --test     # once; logged in results/TEST_ACCESS_LOG.md
Outputs: results/reader_analysis.json (development) or results/reader_analysis_test.json.
"""
import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

import analysis

SCRIPT = "42_reader_analysis"
LABELS = Path("data/labels/main")
WORD = ["olens", "nla", "jlens", "jpp", "logit_lens"]
NUM = ["template_lens", "ao", "transfer_probe"]
TEXT = ["llm_reader", "prompt_only_llm"]
H4_LENSES, H4_DESCRIBERS = ["jlens", "logit_lens"], ["nla", "olens", "ao"]
WORD_CUTOFF = 0.25
L3_SEED, L3_BOOT, L3_PERM = 20261106, 2000, 10000
REMARK = (r"(a|or|be|likely|probably|maybe|just|is|this) (a )?test\b|testing (me|the|my|whether|how)|being tested"
          r"|evaluat(ing|ion) (me|of me|the (ai|model|assistant))|red[- ]team")


def load_scores(test):
    """reader -> {(trace_id, offset): score}, status ok only (failed rows are excluded, D-186)."""
    suffix = "locked_test" if test else "dev"
    scores = collections.defaultdict(dict)
    for name, col in [(f"activation_readers_{suffix}.csv", "score"), (f"llm_readers_{suffix}.csv", "q_pred")]:
        for r in analysis.read_table(LABELS / name):
            if r["status"] == "ok" and r[col] is not None:
                scores[r["reader"]][(r["trace_id"], int(r["offset"]))] = float(r[col])
    return scores


def population(cut_points, scores, split):
    cp = {(r["trace_id"], int(r["offset"])): r for r in cut_points}
    keys = sorted(k for k in scores["llm_reader"].keys() | scores["transfer_probe"].keys()
                  if k in cp and cp[k]["split"] == split and cp[k]["eligible"] == 1 and cp[k]["q"] is not None)
    return cp, keys


def thresholds(cp, keys, scores):
    """Number and text readers: the threshold that flags 10% of q = 0 cut points (+1e-12, flag at or above)."""
    out = {rd: WORD_CUTOFF for rd in WORD}
    for rd in NUM + TEXT:
        neg = np.sort([scores[rd][k] for k in keys if k in scores[rd] and cp[k]["q"] == 0])
        out[rd] = float(neg[int(np.ceil(0.9 * len(neg))) - 1]) + 1e-12
    return out


def bootstrap_draws(stat, clusters, resamples=analysis.RESAMPLES, seed=analysis.SEED):
    """Same cluster resampling as analysis.cluster_bootstrap, returning the draws (for a two-sided p)."""
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    members = [np.flatnonzero(clusters == c) for c in uniq]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        idx = np.concatenate([members[c] for c in rng.integers(0, len(uniq), len(uniq))])
        v = stat(idx)
        if v is not None and np.isfinite(v):
            draws.append(v)
    return np.array(draws)


def holm(pvals):
    order = sorted(pvals, key=pvals.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvals[k]))
        out[k] = running
    return out


def vs_q(cp, keys, scores, clusters, offset=None):
    out = {}
    for rd in WORD + NUM + TEXT:
        ks = [k for k in keys if k in scores[rd] and (offset is None or k[1] == offset)]
        s = np.array([scores[rd][k] for k in ks]); q = np.array([cp[k]["q"] for k in ks], float)
        cl = [clusters[int(cp[k]["scenario_id"])] for k in ks]
        out[rd] = analysis.estimate(lambda i: {"rho": analysis.spearman(s[i], q[i]),
                                               "auroc": analysis.auroc((q[i] > 0).astype(int), s[i])}, cl)
        out[rd]["n_q_positive"] = int((q > 0).sum())
    return out


def h3_matrix(cp, keys, scores, probe):
    names = WORD + NUM + TEXT + (["q_probe"] if probe else [])
    table = dict(scores, q_probe=probe or {})
    ks = [k for k in keys if all(k in table[n] for n in names)]
    cols = {n: np.array([table[n][k] for k in ks]) for n in names}
    cols["q"] = np.array([cp[k]["q"] for k in ks], float)
    return {"n": len(ks), "readers": names + ["q"],
            "rho": {a: {b: analysis.spearman(cols[a], cols[b]) for b in cols} for a in cols}}


def h4(cp, keys, scores, clusters):
    out, p = {}, {}
    for lens in H4_LENSES:
        for desc in H4_DESCRIBERS:
            ks = [k for k in keys if k in scores[lens] and k in scores[desc]]
            a = np.array([scores[lens][k] for k in ks]); b = np.array([scores[desc][k] for k in ks])
            q = np.array([cp[k]["q"] for k in ks], float)
            cl = [clusters[int(cp[k]["scenario_id"])] for k in ks]

            def diff(i):
                ra, rb = analysis.spearman(a[i], q[i]), analysis.spearman(b[i], q[i])
                return None if ra is None or rb is None else ra - rb
            d = bootstrap_draws(diff, cl)
            name = f"{lens}_minus_{desc}"
            p[name] = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
            out[name] = {"value": diff(np.arange(len(ks))), "ci95": [float(np.percentile(d, 2.5)),
                         float(np.percentile(d, 97.5))], "p": p[name], "n": len(ks), "n_clusters": len(set(cl))}
    for name, ph in holm(p).items():
        out[name]["p_holm"] = ph
    return out


def level3(cp, keys, scores, thr, prefix_text):
    rng = np.random.default_rng(L3_SEED)
    base = [k for k in keys if k in scores["llm_reader"] and cp[k]["q"] == 0]
    quiet = [k for k in base if scores["llm_reader"][k] < thr["llm_reader"]]

    def analyse(ks, rd):
        ks = [k for k in ks if k in scores[rd]]
        flag = np.array([scores[rd][k] >= thr[rd] for k in ks])
        hypo = np.array([cp[k]["condition"] == "HYPO" for k in ks])
        scen = np.array([str(cp[k]["scenario_id"]) for k in ks])
        us = sorted(set(scen)); idx = {s: np.where(scen == s)[0] for s in us}
        diff = lambda f, h: f[h].mean() - f[~h].mean()  # noqa: E731
        d = diff(flag, hypo)
        boots = []
        for _ in range(L3_BOOT):
            sel = np.concatenate([idx[s] for s in rng.choice(us, len(us))])
            if hypo[sel].any() and (~hypo[sel]).any():
                boots.append(diff(flag[sel], hypo[sel]))
        null = []
        for _ in range(L3_PERM):
            sw = {s: rng.random() < .5 for s in us}
            null.append(diff(flag, np.array([hy != sw[s] for hy, s in zip(hypo, scen)])))
        return {"n_placeholder": int(hypo.sum()), "n_real": int((~hypo).sum()),
                "flag_placeholder": float(flag[hypo].mean()), "flag_real": float(flag[~hypo].mean()),
                "diff": float(d), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                "p": float((np.abs(null) >= abs(d) - 1e-12).mean())}

    out = {"n_q0": len(base), "n_q0_text_quiet": len(quiet), "primary": {}, "secondary": {}, "sensitivity_all_q0": {}}
    for rd in WORD:   # same order as readers/analysis/level3.py, so validation numbers reproduce exactly
        out["primary"][rd] = analyse(quiet, rd)
        out["sensitivity_all_q0"][rd] = analyse(base, rd)
    for rd in NUM:
        out["secondary"][rd] = analyse(quiet, rd)
        out["sensitivity_all_q0"][rd] = analyse(base, rd)
    out["controls"] = {rd: analyse(quiet, rd) for rd in TEXT}
    out["controls_all_q0"] = {rd: analyse(base, rd) for rd in TEXT}
    remark = re.compile(REMARK, re.I)
    strip = lambda t: re.sub(r"https?://\S+|`[^`]*`", " ", t)  # noqa: E731
    no_remark = [k for k in quiet if not remark.search(strip(prefix_text[k]))]
    out["n_no_test_remark"] = len(no_remark)
    out["sensitivity_no_test_remark"] = {rd: analyse(no_remark, rd) for rd in WORD + NUM}
    for rd, ph in holm({rd: out["primary"][rd]["p"] for rd in WORD}).items():
        out["primary"][rd]["p_holm"] = ph
    return out


def read_prefix_text(keys):
    keys = set(keys)
    out = {}
    for line in open(LABELS / "prefix_texts.jsonl"):
        j = json.loads(line)
        if (j["trace_id"], j["offset"]) in keys:
            out[(j["trace_id"], j["offset"])] = j["text"]
    return out


def read_probe(path, split):
    if not Path(path).exists():
        return None
    return {(r["trace_id"], int(r["offset"])): float(r["q_probe"]) for r in analysis.read_table(path)
            if r["split"] == split and r["q_probe"] is not None}


def run(cut_points, scores, split, thr, probe):
    clusters = analysis.load_clusters()
    cp, keys = population(cut_points, scores, split)
    return {"split": split, "n": len(keys), "n_scenarios": len({cp[k]["scenario_id"] for k in keys}),
            "n_q_positive": sum(cp[k]["q"] > 0 for k in keys), "thresholds": thr,
            "vs_q": vs_q(cp, keys, scores, clusters), "vs_q_offset_256": vs_q(cp, keys, scores, clusters, 256),
            "h3_matrix": h3_matrix(cp, keys, scores, probe), "h4": h4(cp, keys, scores, clusters),
            "level3": level3(cp, keys, scores, thr, read_prefix_text(keys))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    analysis.add_mode_args(parser)
    args = parser.parse_args(argv)
    if args.test:
        test_files = [LABELS / "cut_points_locked_test.csv", LABELS / "activation_readers_locked_test.csv",
                      LABELS / "llm_readers_locked_test.csv", LABELS / "prefix_texts.jsonl"]
        frozen, _ = analysis.open_test(SCRIPT, "activation-reader analysis (H3, H4, level 3) on test_id", args,
                                       needs=test_files)
        scores = load_scores(test=True)
        report = run(analysis.read_cut_points(test=True), scores, "test_id", frozen["thresholds"],
                     read_probe("results/q_probe_predictions_test.csv", "test_id"))
        report["frozen"] = frozen
        analysis.write_report("results/reader_analysis_test.json", report)
        print("wrote results/reader_analysis_test.json")
        return
    scores = load_scores(test=False)
    cut_points = analysis.read_cut_points(test=False)
    cp, keys = population(cut_points, scores, "val")
    thr = thresholds(cp, keys, scores)
    report = run(cut_points, scores, "val", thr, read_probe("results/q_probe_predictions_dev.csv", "val"))
    analysis.write_report("results/reader_analysis.json", report)
    if args.freeze:
        path = analysis.save_frozen(SCRIPT, {"thresholds": thr, "word_cutoff": WORD_CUTOFF, "word": WORD, "num": NUM,
                                             "text": TEXT, "h4_lenses": H4_LENSES, "h4_describers": H4_DESCRIBERS,
                                             "remark_regex": REMARK, "level3_seed": L3_SEED,
                                             "dev_n": report["n"], "dev_n_q_positive": report["n_q_positive"]})
        print("froze", path)
    l3 = report["level3"]["primary"]
    print(f"val n={report['n']} q>0={report['n_q_positive']} level-3 quiet={report['level3']['n_q0_text_quiet']}")
    for rd in WORD + NUM + TEXT:
        v = report["vs_q"][rd]
        print(f"  {rd:15s} rho={v['rho']['value']} auroc={v['auroc']['value']}"
              + (f"  L3 diff={l3[rd]['diff']:+.3f} p={l3[rd]['p']:.3f} holm={l3[rd]['p_holm']:.3f}" if rd in l3 else ""))
    for name, v in report["h4"].items():
        print(f"  H4 {name:25s} {v['value']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] holm={v['p_holm']:.3f}")


if __name__ == "__main__":
    main()
