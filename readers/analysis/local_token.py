"""How much of a lens readout is the local word: share of validation cut points whose current word (the last
token before the cut, alphabetic tokens only) or next token appears in the lens's top 20 tokens.
Usage: python local_token.py <hw2 checkout>. Reads only validation files; writes next_token.json."""
import csv, glob, json, sys
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

H = sys.argv[1]
tok = Tokenizer.from_file(hf_hub_download('Qwen/Qwen3.6-27B', 'tokenizer.json',
                                          revision='6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'))
cp = {(r['trace_id'], int(r['offset'])): r for r in csv.DictReader(open(f'{H}/data/labels/main/cut_points_dev.csv'))}
outs = {rd: {(j['trace_id'], j['offset']): j['raw'] for j in
             map(json.loads, open(f'{H}/data/generations/main/activation_readers/{rd}/dev/outputs.jsonl'))}
        for rd in ['jlens', 'jpp', 'logit_lens']}
keys = {k for k in outs['jlens'] if cp[k]['eligible'] == '1' and cp[k]['split'] == 'val'}
need, ids = {k[0] for k in keys}, {}
for f in sorted(glob.glob(f'{H}/data/generations/main/base/traces-*.jsonl')):
    for line in open(f):
        j = json.loads(line)
        if j['trace_id'] in need:
            ids[j['trace_id']] = j['generated_ids']
norm = lambda s: s.strip().lower()
res = {}
for rd in outs:
    cur_hits = next_hits = n = 0
    for k in keys:
        g = ids[k[0]]
        cur, nxt = norm(tok.decode([g[k[1] - 1]])), norm(tok.decode([g[k[1]]]))
        if not cur.isalpha():
            continue
        top = [norm(t) for t, _ in outs[rd][k]['top_tokens']]
        n += 1; cur_hits += cur in top; next_hits += nxt in top
    res[rd] = {'n_word_tokens': n, 'current_in_top20': cur_hits / n, 'next_in_top20': next_hits / n}
    print(rd, res[rd])
json.dump(res, open('next_token.json', 'w'), indent=1)
