"""Main figure for the README: pooled AUROC and within-prompt Spearman, validation split."""
import json, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
INK, MUTED = '#1d2430', '#6b7280'
C = ['#0072B2', '#E69F00', '#009E73']
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True, 'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'pdf.fonttype': 42})
d = json.load(open('results_dev.json'))['readers']; w = json.load(open('results_dev_extra.json'))['within_prompt']
ORDER = [('llm_reader', 'LLM reader (text before cut)', 0), ('prompt_only_llm', 'LLM reader (prompt only)', 0),
         ('template_lens', 'Template lens', 1), ('ao', 'Activation oracle', 1), ('transfer_probe', 'Transfer probe', 1),
         ('jlens', 'J-lens', 2), ('olens', 'Oracle lens', 2), ('nla', 'NLA', 2), ('jpp', 'J++ lens', 2), ('logit_lens', 'Logit lens', 2)]
MK = ['o', 's', '^']
ys, y = [], 0
for i, (_, _, g) in enumerate(ORDER):
    if i and ORDER[i - 1][2] != g: y += 0.6
    ys.append(-y); y += 1
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.75, 3.3), sharey=True, gridspec_kw={'wspace': 0.08})
a1.axvline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 3)), zorder=0)
a2.axvline(0.0, color=MUTED, lw=0.7, ls=(0, (3, 3)), zorder=0)
for (k, lab, g), yy in zip(ORDER, ys):
    r = d[k]
    a1.plot(r['auroc_ci'], [yy, yy], color=C[g], lw=1.4, solid_capstyle='round')
    a1.plot(r['auroc'], yy, MK[g], color=C[g], ms=5.5, mec='none')
    v = w[k]
    if v['rho'] is None:
        a2.text(0.0, yy, 'constant within a prompt', va='center', ha='center', fontsize=7.5, color=MUTED, bbox=dict(fc='white', ec='none', pad=1.5))
    else:
        a2.plot(v['ci'], [yy, yy], color=C[g], lw=1.4, solid_capstyle='round')
        a2.plot(v['rho'], yy, MK[g], color=C[g], ms=5.5, mec='none')
a1.set_yticks(ys, [lab for _, lab, _ in ORDER])
a1.set_xlim(0.4, 0.9); a2.set_xlim(-0.65, 0.8)
a1.set_ylim(min(ys) - 0.7, 0.7)
a1.set_xlabel('AUROC, aware ($q>0$) vs. not ($q=0$)')
a2.set_xlabel(r'Spearman $\rho$ with $q$, within prompt')
a1.set_title('(a) All validation cut points', fontsize=9, loc='left')
a2.set_title('(b) Same prompt and cut offset', fontsize=9, loc='left')
fig.legend([Line2D([], [], marker=MK[g], color=C[g], ls='', ms=5.5) for g in range(3)],
           ['Reads the text', 'Activation, numeric score', 'Activation, words scored by a judge'],
           loc='lower center', ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.55, -0.13))
fig.savefig('fig_main.pdf', bbox_inches='tight', pad_inches=0.03)
fig.savefig('fig_main.png', dpi=300, bbox_inches='tight', pad_inches=0.03, facecolor='white')
print('ok')
