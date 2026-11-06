"""G3: Sonnet judge vs Advikar's blind Round 2 labels (items 1-50), binary voiced = explicit or hedged."""
import ast, csv, json, sys
H = sys.argv[1]
key = json.load(open(f'{H}/data/generations/main/round2/key.json'))
hum = {r['item']: r for r in csv.DictReader(open(sys.argv[2])) if r['reasoning_label']}
J = {}
for f in ['base_trace', 'base_continuation']:
    for line in open(f'{H}/data/generations/main/judge/sonnet/dev/{f}.jsonl'):
        j = json.loads(line)
        if j['status'] == 'ok':
            J[j['id']] = ast.literal_eval(j['labels']) if isinstance(j['labels'], str) else j['labels']
POS = {'explicit', 'hedged'}
def kappa(pairs):
    n = len(pairs); po = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n; pb = sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float('nan'), n, sum(a for a, _ in pairs), sum(b for _, b in pairs), po
rows, missing = [], []
for it, h in hum.items():
    k = key[it]
    if k['id'] not in J: missing.append((it, k['id'])); continue
    j = J[k['id']]
    for fld in ['reasoning_label', 'answer_label']:
        rows.append((fld, k['half'], k['kind'], k['contains_test_word'], h[fld] in POS, j[fld] in POS))
print('items', len(hum), 'missing judge', missing)
out = {}
for name, f in [('reasoning, pooled', lambda r: r[0] == 'reasoning_label'),
                ('reasoning, random half', lambda r: r[0] == 'reasoning_label' and r[1] == 'random'),
                ('reasoning, keyword half', lambda r: r[0] == 'reasoning_label' and r[1] == 'keyword'),
                ('reasoning, contains "test"', lambda r: r[0] == 'reasoning_label' and r[3]),
                ('answer, pooled', lambda r: r[0] == 'answer_label')]:
    p = [(r[4], r[5]) for r in rows if f(r)]
    k, n, hp, jp, po = kappa(p)
    out[name] = dict(kappa=k, n=n, human_pos=hp, judge_pos=jp, agree=po)
    print(f'{name:30s} kappa={k:.2f} n={n} human+={hp} judge+={jp} agree={po:.2f}')
json.dump(out, open('g3_items1-50.json', 'w'), indent=1)
