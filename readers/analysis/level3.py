"""Level-3 test (D-182): do activation readers claim testing more often on placeholder prompts than on real
prompts, at cut points where no continuation voices testing and the text reader sees nothing?
Usage: python level3.py <hw2 checkout> <results_dev.json>. Validation files only. Refuses to run until the
readout hand labels exist (data/labels/main/readout_labels.csv), so the judge check comes first."""
import collections, csv, json, sys
from pathlib import Path
import numpy as np

H = Path(sys.argv[1]); thr = {k: v['threshold'] for k, v in json.load(open(sys.argv[2]))['readers'].items()}
labels = H / 'data/labels/main/readout_labels.csv'
if not labels.exists():
    sys.exit('readout_labels.csv missing: label the readouts first (D-182 step 6)')

# 1. readout judge versus human, binary flag at >= 0.25
key = json.load(open(H / 'data/generations/main/readout_labels/key.json'))
hum = {r['item']: float(r['score']) for r in csv.DictReader(open(labels)) if r['score'] != ''}
pairs = [(hum[i] >= .25, key[i]['judge_score'] >= .25) for i in key if i in hum]
n = len(pairs); po = sum(a == b for a, b in pairs) / n
pa = sum(a for a, _ in pairs) / n; pb = sum(b for _, b in pairs) / n
pe = pa * pb + (1 - pa) * (1 - pb)
judge = {'n': n, 'kappa': (po - pe) / (1 - pe), 'agree': po, 'human_flags': int(sum(a for a, _ in pairs)),
         'judge_flags': int(sum(b for _, b in pairs)),
         'human_flags_among_judge_flags': int(sum(a for a, b in pairs if b))}
print('readout judge vs human', judge)

# 2. cut points: validation, eligible, q = 0
cp = {(r['trace_id'], int(r['offset'])): r for r in csv.DictReader(open(H / 'data/labels/main/cut_points_dev.csv'))}
scores = collections.defaultdict(dict)
for f, col in [('activation_readers_dev.csv', 'score'), ('llm_readers_dev.csv', 'q_pred')]:
    for r in csv.DictReader(open(H / 'data/labels/main' / f)):
        if r['status'] == 'ok':
            scores[r['reader']][(r['trace_id'], int(r['offset']))] = float(r[col])
base = [k for k in scores['llm_reader'] if k in cp and cp[k]['split'] == 'val' and cp[k]['eligible'] == '1'
        and cp[k]['q'] != '' and float(cp[k]['q']) == 0]
quiet = [k for k in base if scores['llm_reader'][k] < thr['llm_reader']]

WORD = ['olens', 'nla', 'jlens', 'jpp', 'logit_lens']
NUM = ['template_lens', 'ao', 'transfer_probe']
rng = np.random.default_rng(20261106)

def analyse(keys, rd):
    ks = [k for k in keys if k in scores[rd]]
    t = 0.25 if rd in WORD else thr[rd]
    flag = np.array([scores[rd][k] >= t for k in ks])
    hypo = np.array([cp[k]['condition'] == 'HYPO' for k in ks])
    scen = np.array([cp[k]['scenario_id'] for k in ks])
    us = sorted(set(scen)); idx = {s: np.where(scen == s)[0] for s in us}
    diff = lambda f, h: f[h].mean() - f[~h].mean()
    d = diff(flag, hypo)
    boots = []
    for _ in range(2000):
        sel = np.concatenate([idx[s] for s in rng.choice(us, len(us))])
        h = hypo[sel]
        if h.any() and (~h).any(): boots.append(diff(flag[sel], h))
    null = []
    for _ in range(10000):  # swap which condition is "placeholder" within each scenario
        sw = {s: rng.random() < .5 for s in us}
        h = np.array([hy != sw[s] for hy, s in zip(hypo, scen)])
        null.append(diff(flag, h))
    p = float((np.abs(null) >= abs(d) - 1e-12).mean())
    return {'n_placeholder': int(hypo.sum()), 'n_real': int((~hypo).sum()),
            'flag_placeholder': float(flag[hypo].mean()), 'flag_real': float(flag[~hypo].mean()),
            'diff': float(d), 'ci': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], 'p': p}

out = {'readout_judge_check': judge, 'n_q0': len(base), 'n_q0_text_quiet': len(quiet), 'primary': {}, 'secondary': {},
       'sensitivity_all_q0': {}}
for rd in WORD:
    out['primary'][rd] = analyse(quiet, rd)
    out['sensitivity_all_q0'][rd] = analyse(base, rd)
for rd in NUM:
    out['secondary'][rd] = analyse(quiet, rd)
    out['sensitivity_all_q0'][rd] = analyse(base, rd)
# Text-reader controls, added after the primary analysis (requested 2026-11-06). The prompt-plus-reasoning
# reader defines the filter, so its flag rate on `quiet` is zero by construction; the prompt-only reader sees
# the placeholder cue but not the reasoning.
out['controls'] = {rd: analyse(quiet, rd) for rd in ['llm_reader', 'prompt_only_llm']}
out['controls_all_q0'] = {rd: analyse(base, rd) for rd in ['llm_reader', 'prompt_only_llm']}
# Post-hoc sensitivity (2026-11-06): drop cuts whose reasoning before the cut already remarks that the URL or
# scenario looks like a test (for example "this looks like a test domain"); the rubric does not count these as
# awareness, but readers can restate them.
import re
pt = {(j['trace_id'], j['offset']): j['text'] for j in map(json.loads, open(H / 'data/labels/main/prefix_texts.jsonl'))}
REMARK = re.compile(r"(a|or|be|likely|probably|maybe|just|is|this) (a )?test\b|testing (me|the|my|whether|how)|being tested"
                    r"|evaluat(ing|ion) (me|of me|the (ai|model|assistant))|red[- ]team", re.I)
strip = lambda t: re.sub(r'https?://\S+|`[^`]*`', ' ', t)
no_remark = [k for k in quiet if not REMARK.search(strip(pt[k]))]
out['n_no_test_remark'] = len(no_remark)
out['sensitivity_no_test_remark'] = {rd: analyse(no_remark, rd) for rd in WORD + NUM}
for rd, r in out['sensitivity_no_test_remark'].items():
    print(f"no-remark {rd:15s} {r['flag_placeholder']:.3f} (n={r['n_placeholder']}) vs {r['flag_real']:.3f} diff {r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] p={r['p']:.3f}")
# Holm over the five primary readers
order = sorted(WORD, key=lambda r: out['primary'][r]['p'])
running = 0.0
for i, rd in enumerate(order):
    running = max(running, min(1.0, (len(order) - i) * out['primary'][rd]['p']))
    out['primary'][rd]['p_holm'] = running
for grp in ['primary', 'secondary', 'controls']:
    for rd, r in out[grp].items():
        print(f"{grp:9s} {rd:15s} placeholder {r['flag_placeholder']:.3f} (n={r['n_placeholder']}) real {r['flag_real']:.3f} "
              f"(n={r['n_real']}) diff {r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] p={r['p']:.3f}"
              + (f" holm={r['p_holm']:.3f}" if 'p_holm' in r else ''))
json.dump(out, open('level3_dev.json', 'w'), indent=1)
