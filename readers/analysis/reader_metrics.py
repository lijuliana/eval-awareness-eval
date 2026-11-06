"""Dev-only (val) reader results. Reads only *_dev files; never the locked/test files."""
import csv, json, collections, numpy as np, sys
H = sys.argv[1]
cp = {(r['trace_id'], int(r['offset'])): r for r in csv.DictReader(open(f'{H}/data/labels/main/cut_points_dev.csv'))}
scores = collections.defaultdict(dict)
for r in csv.DictReader(open(f'{H}/data/labels/main/activation_readers_dev.csv')):
    if r['status'] == 'ok': scores[r['reader']][(r['trace_id'], int(r['offset']))] = float(r['score'])
for r in csv.DictReader(open(f'{H}/data/labels/main/llm_readers_dev.csv')):
    if r['status'] == 'ok': scores[r['reader']][(r['trace_id'], int(r['offset']))] = float(r['q_pred'])
keys = sorted(k for k in scores['llm_reader'].keys() | scores['transfer_probe'].keys()
              if k in cp and cp[k]['eligible'] == '1' and cp[k]['q'] != '' and cp[k]['split'] == 'val')
q = {k: float(cp[k]['q']) for k in keys}
scen = {k: cp[k]['scenario_id'] for k in keys}
print('n', len(keys), 'pos', sum(v > 0 for v in q.values()), 'strong', sum(v >= .5 for v in q.values()),
      'scenarios', len(set(scen.values())), 'n_cont', collections.Counter(cp[k]['n_cont'] for k in keys))

def rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[o] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v; r[m] = r[m].mean()
    return r
def auroc(s, y):
    s, y = np.asarray(s), np.asarray(y, bool)
    if y.all() or (~y).all(): return np.nan
    r = rank(s); n1 = y.sum(); n0 = (~y).sum()
    return (r[y].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
def spear(a, b):
    a, b = rank(a), rank(b)
    return np.nan if a.std() == 0 or b.std() == 0 else np.corrcoef(a, b)[0, 1]

rng = np.random.default_rng(20261106)
scen_list = sorted(set(scen.values()))
B = 2000
boot_idx = [rng.choice(len(scen_list), len(scen_list)) for _ in range(B)]

def boot(fn, ks, *cols):
    by = collections.defaultdict(list)
    for i, k in enumerate(ks): by[scen[k]].append(i)
    vals = []
    for idx in boot_idx:
        sel = [i for j in idx for i in by.get(scen_list[j], [])]
        v = fn(*[np.asarray(c)[sel] for c in cols])
        if not np.isnan(v): vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

WORD = ['logit_lens', 'jlens', 'jpp', 'olens', 'nla']
CONT = ['llm_reader', 'prompt_only_llm', 'transfer_probe', 'template_lens', 'ao']
out = {'n': len(keys), 'readers': {}}
for rd in CONT + WORD:
    ks = [k for k in keys if k in scores[rd]]
    s = [scores[rd][k] for k in ks]; qq = [q[k] for k in ks]; y = [v > 0 for v in qq]
    res = {'n': len(ks), 'n_pos': int(sum(y)),
           'auroc': float(auroc(s, y)), 'auroc_ci': boot(auroc, ks, s, y),
           'rho': float(spear(s, qq)), 'rho_ci': boot(spear, ks, s, qq)}
    sa, ya = np.asarray(s), np.asarray(y)
    if rd in WORD:
        thr = 0.25
    else:  # threshold giving <=10% flagged among q=0 (chosen on this same dev data)
        neg = np.sort(sa[~ya]); thr = float(neg[int(np.ceil(0.9 * len(neg))) - 1]) + 1e-12
    flag = sa >= thr
    res.update(threshold=thr, recall=float(flag[ya].mean()), fpr=float(flag[~ya].mean()),
               recall_strong=float(flag[np.asarray(qq) >= .5].mean()))
    # ROC curve points
    ths = np.unique(sa)[::-1]
    res['roc'] = [[0.0, 0.0]] + [[float((sa[~ya] >= t).mean()), float((sa[ya] >= t).mean())] for t in ths]
    # detection by q bin
    bins = [(0, 0), (0.125, 0.125), (0.25, 0.375), (0.5, 1.0)]
    res['by_q'] = [float(flag[(np.asarray(qq) >= lo) & (np.asarray(qq) <= hi)].mean()) for lo, hi in bins]
    res['by_q_n'] = [int(((np.asarray(qq) >= lo) & (np.asarray(qq) <= hi)).sum()) for lo, hi in bins]
    out['readers'][rd] = res
    print(f"{rd:16s} n={len(ks)} AUROC={res['auroc']:.2f} {np.round(res['auroc_ci'],2)} rho={res['rho']:+.2f} "
          f"recall={res['recall']:.2f} fpr={res['fpr']:.2f} strong={res['recall_strong']:.2f} byq={np.round(res['by_q'],2)}")
# split-half ceiling of q
ha = [float(cp[k]['q_half_a']) for k in keys if cp[k]['q_half_a'] != '']
hb = [float(cp[k]['q_half_b']) for k in keys if cp[k]['q_half_b'] != '']
out['q_split_half_rho'] = float(spear(ha, hb)) if len(ha) == len(hb) else None
print('split-half rho', out['q_split_half_rho'])
json.dump(out, open('results_dev.json', 'w'), indent=1)
