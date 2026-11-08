"""Silent-continuation reading (D-189): flag rates by item kind, and AUROC silent-aware vs unaware."""
import json, re, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from fig_test import style, save, SENT, REAL, NAME
D = json.load(open('silent_reading.json'))
TEXT = ['llm_reader', 'prompt_only_llm']
ACT = sorted([k for k in NAME if k not in TEXT], key=lambda k: -D['auroc'][k]['value'])
ORDER = TEXT + ACT

def fig(P, svg):
    style(P, svg)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 4.3), sharey=True, gridspec_kw={'wspace': 0.08})
    ys, y = [], 0
    for i, k in enumerate(ORDER):
        if i == len(TEXT): y += 0.8
        ys.append(-y); y += 1
    ax = axes[0]
    for k, yy in zip(ORDER, ys):
        f = D['flag_rates'][k]
        vals = [100 * f[x]['rate'] for x in ('unaware', 'silent_aware', 'verbalizing')]
        ax.plot([min(vals), max(vals)], [yy, yy], color=P['rule'], lw=1.0, zorder=0)
        ax.plot(vals[0], yy, 'o', ms=5.5, color=P['muted'], mfc='none', mew=1.2)
        ax.plot(vals[1], yy, 'o', ms=6, color=P['c1'])
        ax.plot(vals[2], yy, '^', ms=6, color=P['c2'])
    ax.set_xlim(-2, 70); ax.set_xlabel('Continuations flagged (%)')
    ax.plot([], [], 'o', color=P['muted'], mfc='none', mew=1.2, label='Unaware state')
    ax.plot([], [], 'o', color=P['c1'], label='Aware state, never verbalized')
    ax.plot([], [], '^', color=P['c2'], label='About to verbalize')
    ax.legend(frameon=False, fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), handletextpad=0.2, ncol=1)
    ax = axes[1]
    ax.axvline(0.5, color=P['muted'], lw=0.8, ls=(0, (3, 3)), zorder=0)
    for k, yy in zip(ORDER, ys):
        col = P['c2'] if k in TEXT else P['c1']
        rows = [(D['auroc'][k], 0.17, True)]
        if k != 'llm_reader': rows.append((D['auroc_text_quiet'][k], -0.17, False))
        for r, dy, filled in rows:
            lo, hi = r['ci95']
            ax.plot([lo, hi], [yy + dy] * 2, color=col, lw=1.4, solid_capstyle='round', alpha=1 if filled else 0.55)
            ax.plot(r['value'], yy + dy, 'o', ms=5.5, color=col, mfc=col if filled else 'none', mew=1.2)
    ax.plot([], [], 'o', color=P['muted'], label='All continuations')
    ax.plot([], [], 'o', color=P['muted'], mfc='none', mew=1.2, label='Text reader sees nothing')
    ax.legend(frameon=False, fontsize=8.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), handletextpad=0.2, ncol=1)
    ax.set_xlim(0.35, 0.9); ax.set_xlabel('AUROC, silent aware vs unaware')
    axes[0].set_yticks(ys); axes[0].set_yticklabels([NAME[k] for k in ORDER])
    for a in axes: a.spines['left'].set_visible(False); a.tick_params(axis='y', length=0)
    axes[0].set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
    save(fig, 'fig_silent', svg)

for P, svg in [(SENT, True), (REAL, False)]: fig(P, svg)
print('ok')
