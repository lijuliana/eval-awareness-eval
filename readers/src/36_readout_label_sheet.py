"""Build the readout hand-labeling sheet (PROPOSAL Round 2 "20 readouts"; D-182).

40 validation readouts of the word readers, shown exactly as the readout judge sees them, with the judge's
question and scale. The labeler sees no judge score, condition, prompt or q. Selection uses only the judge
score and the reader (never the condition): all 4 readouts the judge scored 0.5, 22 of the 86 scored 0.25
(olens 6, nla 6, jlens 4, jpp 3, logit_lens 3), and 14 scored 0 (olens 4, nla 4, jlens 2, jpp 2,
logit_lens 2). Seeded; one fixed draw. Validation split only.

Outputs (data/generations/main/readout_labels/):
- label_tool.html   the labeling page; downloads readout_labels.csv (save as data/labels/main/readout_labels.csv)
- key.json          item -> trace_id, offset, reader, judge score (do not open before labeling)
"""
import csv, html, importlib.util, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/generations/main/readout_labels"
spec = importlib.util.spec_from_file_location("rj", ROOT / "src/35_readout_judge.py")
rj = importlib.util.module_from_spec(spec); spec.loader.exec_module(rj)
readout_text = rj.readout_text

READERS = ["olens", "nla", "jlens", "jpp", "logit_lens"]
TAKE_025 = {"olens": 6, "nla": 6, "jlens": 4, "jpp": 3, "logit_lens": 3}
TAKE_0 = {"olens": 4, "nla": 4, "jlens": 2, "jpp": 2, "logit_lens": 2}
SEED = 20261106

cp = {(r["trace_id"], int(r["offset"])): r for r in csv.DictReader(open(ROOT / "data/labels/main/cut_points_dev.csv"))}
ok = lambda k: k in cp and cp[k]["split"] == "val" and cp[k]["eligible"] == "1" and cp[k]["q"] != ""
score = {}
for r in csv.DictReader(open(ROOT / "data/labels/main/activation_readers_dev.csv")):
    k = (r["trace_id"], int(r["offset"]))
    if r["reader"] in READERS and r["status"] == "ok" and ok(k):
        score[(r["reader"],) + k] = float(r["score"])
raw = {}
for rd in READERS:
    for line in open(ROOT / f"data/generations/main/activation_readers/{rd}/dev/outputs.jsonl"):
        j = json.loads(line)
        raw[(rd, j["trace_id"], j["offset"])] = j["raw"]

rng = random.Random(SEED)
pick = sorted(k for k, s in score.items() if s == 0.5)
for rd in READERS:
    for level, take in [(0.25, TAKE_025), (0.0, TAKE_0)]:
        pool = sorted(k for k, s in score.items() if k[0] == rd and s == level)
        pick += rng.sample(pool, take[rd])
assert len(pick) == 40 and len(set(pick)) == 40
rng.shuffle(pick)

OUT.mkdir(parents=True, exist_ok=True)
key = {str(i + 1): {"reader": k[0], "trace_id": k[1], "offset": k[2], "judge_score": score[k]} for i, k in enumerate(pick)}
json.dump(key, open(OUT / "key.json", "w"), indent=1)

question = Path(ROOT / rj.PROMPT).read_text().split("Reply with JSON only")[0]
items = "\n".join(f"""<section class=item data-i="{i + 1}"><h2>Item {i + 1} of 40</h2>
<pre>{html.escape(readout_text(k[0], raw[k]))}</pre>
<div class=opts>{''.join(f'<label><input type=radio name=s{i + 1} value="{v}"> {v}</label>' for v in ['0', '0.25', '0.5', '0.75', '1'])}</div>
<input class=note name=n{i + 1} placeholder="notes (optional)"></section>""" for i, k in enumerate(pick))

page = f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Readout labels</title>
<style>body{{font:16px/1.5 Georgia,serif;max-width:760px;margin:24px auto;padding:0 16px;background:#fbfcfd;color:#1d2430}}
pre{{white-space:pre-wrap;background:#eef2f6;padding:10px 12px;border-radius:6px;font:13px/1.45 Menlo,monospace}}
.item{{border-top:1px solid #d3d8e0;padding:14px 0}} h2{{font-size:1rem;margin:0 0 6px}}
.opts label{{margin-right:16px}} .note{{width:100%;margin-top:6px;font:14px Georgia,serif}}
.rules{{background:#fff;border:1px solid #d3d8e0;padding:12px 16px;border-radius:6px;font-size:.92rem}}
button{{font:15px Georgia,serif;padding:8px 14px;margin:16px 8px 40px 0}} #st{{color:#5f6876}}</style>
<h1>Readout labels (40 items)</h1>
<p>Score each readout with the same question the readout judge answers. You see only the readout. Answers save in this browser as you go; press <b>Download CSV</b> at the end and save it as <code>data/labels/main/readout_labels.csv</code>. Do not open <code>key.json</code> before you finish.</p>
<details open class=rules><summary>Question and scale (the judge's instructions, verbatim)</summary><pre style="background:none;padding:0">{html.escape(question)}</pre></details>
{items}
<button id=dl>Download CSV</button><span id=st></span>
<script>
const K='readout_labels_v1';let S={{}};try{{S=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}
document.querySelectorAll('input').forEach(el=>{{const v=S[el.name];if(v!==undefined){{if(el.type==='radio')el.checked=(el.value===v);else el.value=v}}
 el.addEventListener('input',()=>{{S[el.name]=el.value;try{{localStorage.setItem(K,JSON.stringify(S))}}catch(e){{}};count()}})}});
function count(){{const n=[...Array(40)].filter((_,i)=>S['s'+(i+1)]!==undefined).length;document.getElementById('st').textContent=n+' of 40 scored'}}count();
document.getElementById('dl').onclick=()=>{{const q=s=>'"'+String(s||'').replace(/"/g,'""')+'"';
 const rows=['item,score,notes'].concat([...Array(40)].map((_,i)=>[i+1,S['s'+(i+1)]??'',q(S['n'+(i+1)])].join(',')));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')+'\\n'],{{type:'text/csv'}}));a.download='readout_labels.csv';a.click()}};
</script>"""
(OUT / "label_tool.html").write_text(page)
print("wrote", OUT / "label_tool.html")
