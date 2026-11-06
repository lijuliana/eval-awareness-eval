"""Dev-only (val): within-prompt Spearman and LLM reader + activation reader combinations.
Usage: python text_vs_activation.py <hw2 checkout>. Reads only *_dev label files."""
import csv, collections, numpy as np, sys, json
H=sys.argv[1]
cp={(r['trace_id'],int(r['offset'])):r for r in csv.DictReader(open(f'{H}/data/labels/main/cut_points_dev.csv'))}
S=collections.defaultdict(dict)
for r in csv.DictReader(open(f'{H}/data/labels/main/activation_readers_dev.csv')):
    if r['status']=='ok': S[r['reader']][(r['trace_id'],int(r['offset']))]=float(r['score'])
for r in csv.DictReader(open(f'{H}/data/labels/main/llm_readers_dev.csv')):
    if r['status']=='ok': S[r['reader']][(r['trace_id'],int(r['offset']))]=float(r['q_pred'])
keys=sorted(k for k in S['llm_reader'] if k in cp and cp[k]['eligible']=='1' and cp[k]['q']!='' and cp[k]['split']=='val')
q=np.array([float(cp[k]['q']) for k in keys]); y=q>0
scen=[cp[k]['scenario_id'] for k in keys]
prompt=[k[0].rsplit('_',1)[0]+f"@{k[1]}" for k in keys]
def rank(x):
    x=np.asarray(x,float); o=np.argsort(x,kind='mergesort'); r=np.empty(len(x)); r[o]=np.arange(len(x))
    for v in np.unique(x): m=x==v; r[m]=r[m].mean()
    return r
def auroc(s,y):
    r=rank(s); n1=y.sum(); n0=(~y).sum(); return (r[y].sum()-n1*(n1-1)/2)/(n1*n0)
def spear(a,b): a,b=rank(a),rank(b); return np.corrcoef(a,b)[0,1] if a.std()>0 and b.std()>0 else np.nan
def z(x): r=rank(x); return (r-r.mean())/r.std()
def within(s):
    g=collections.defaultdict(list)
    for i,p in enumerate(prompt): g[p].append(i)
    a=[];b=[]
    for idx in g.values():
        if len(idx)<2 or np.std(q[idx])==0: continue
        rs=rank(np.asarray(s)[idx]); rq=rank(q[idx]); a+=list(rs-rs.mean()); b+=list(rq-rq.mean())
    a,b=np.array(a),np.array(b); return (a@b)/np.sqrt((a@a)*(b@b)) if (a@a)>0 else np.nan, len(a)
sl=sorted(set(scen)); by=collections.defaultdict(list)
for i,s_ in enumerate(scen): by[s_].append(i)
rng=np.random.default_rng(7); boots=[[i for j in rng.choice(len(sl),len(sl)) for i in by[sl[j]]] for _ in range(2000)]
llm=np.array([S['llm_reader'][k] for k in keys])
print('LLM reader q_pred: mean at q=0 %.3f, at q>0 %.3f; share of q=0 with q_pred>=0.25: %.2f; mean q %.3f'%(llm[~y].mean(),llm[y].mean(),(llm[~y]>=0.25).mean(),q.mean()))
out={'within_prompt':{}, 'paired_with_llm':{}}
def within_idx(s, idx_set):
    g=collections.defaultdict(list)
    for i in idx_set: g[prompt[i]].append(i)
    a=[];b=[]
    for idx in g.values():
        if len(idx)<2 or np.std(q[idx])==0: continue
        rs=rank(np.asarray(s)[idx]); rq=rank(q[idx]); a+=list(rs-rs.mean()); b+=list(rq-rq.mean())
    a,b=np.array(a),np.array(b); return (a@b)/np.sqrt((a@a)*(b@b)) if len(a) and (a@a)>0 else np.nan
print('within-prompt rho (same prompt+offset, different samples):')
for rd in ['llm_reader','prompt_only_llm','template_lens','ao','transfer_probe','jlens','olens','nla','jpp','logit_lens']:
    s=[S[rd].get(k,np.nan) for k in keys]; valid=[i for i in range(len(keys)) if not np.isnan(s[i])]; vs=set(valid)
    w=within_idx(s,valid)
    n=len([i for i in valid if sum(1 for j in valid if prompt[j]==prompt[i])>=2 and np.std([q[j] for j in valid if prompt[j]==prompt[i]])>0])
    bs=[within_idx(s,[i for i in b if i in vs]) for b in boots]; bs=[v for v in bs if not np.isnan(v)]
    ci=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))] if bs else None
    out['within_prompt'][rd]={'rho':None if np.isnan(w) else float(w),'n':n,'ci':ci}
    print(f'  {rd:16s} {w:+.2f} (n={n}) {ci}')
print('LLM reader + activation reader (sum of standardized ranks): AUROC gain over LLM alone [95% CI]')
base=auroc(llm,y)
for rd in ['template_lens','ao','transfer_probe','jlens','olens','nla','jpp','logit_lens','prompt_only_llm']:
    s=np.array([S[rd].get(k,0.0) for k in keys]); comb=z(llm)+z(s)
    g=[auroc(comb[b],y[b])-auroc(llm[b],y[b]) for b in boots if 0<y[b].sum()<len(b)]
    # partial spearman of reader with q controlling for llm
    rl,rs,rq=rank(llm),rank(s),rank(q)
    res=lambda a: a-np.polyval(np.polyfit(rl,a,1),rl)
    ps=np.corrcoef(res(rs),res(rq))[0,1]
    out['paired_with_llm'][rd]={'auroc_combined':float(auroc(comb,y)),'auroc_llm':float(base),'gain_ci':[float(np.percentile(g,2.5)),float(np.percentile(g,97.5))],'partial_rho':float(ps)}
    print(f'  {rd:16s} combined {auroc(comb,y):.3f} vs {base:.3f}: gain {auroc(comb,y)-base:+.3f} [{np.percentile(g,2.5):+.3f}, {np.percentile(g,97.5):+.3f}]  partial rho|LLM {ps:+.2f}')

json.dump(out,open('results_dev_extra.json','w'),indent=1)
