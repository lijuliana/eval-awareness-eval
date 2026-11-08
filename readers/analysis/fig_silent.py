"""D-189/D-190 figures: (1) the eval per prompt version; (2) per-reader flag rates on never-verbalizing continuations."""
import json, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from fig_test import style, save, SENT, REAL, NAME
D = json.load(open('silent_reading.json'))

def fig_eval(P, svg):
    style(P, svg)
    fig, ax = plt.subplots(figsize=(6.3, 2.0))
    rows = [('verbalize', 'Verbalize awareness'), ('silent_activation_flagged', 'Never verbalize it,\nflagged in the activations')]
    for i, (k, lab) in enumerate(rows):
        for v, dy, col, mk, name in [('placeholder', 0.16, P['c1'], 'o', 'Test-like names'), ('real', -0.16, P['muted'], 'D', 'Real names')]:
            r = D['eval'][v][k]; lo, hi = [100 * x for x in r['ci95']]
            ax.plot([lo, hi], [-i + dy] * 2, color=col, lw=1.6, solid_capstyle='round')
            ax.plot(100 * r['value'], -i + dy, mk, color=col, ms=6.5, mec='none')
            ax.text(hi + 0.25, -i + dy, f"{name} {100 * r['value']:.1f}%", va='center', fontsize=8.5, color=P['muted'])
    ax.set_yticks([0, -1]); ax.set_yticklabels([r[1] for r in rows], fontsize=9)
    ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 14); ax.set_ylim(-1.5, 0.5); ax.set_xlabel('Percent of resampled continuations (95% interval)')
    save(fig, 'fig_eval', svg)

ORDER = ['olens', 'nla', 'jlens', 'jpp', 'logit_lens', 'template_lens', 'ao', 'transfer_probe', 'llm_reader', 'prompt_only_llm']
def fig_flags(P, svg):
    style(P, svg)
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    ys, y = [], 0
    for i, k in enumerate(ORDER):
        if k == 'llm_reader': y += 0.8
        ys.append(-y); y += 1
    F = D['flag_rate']
    for k, yy in zip(ORDER, ys):
        vals = [100 * F[k][g]['value'] for g in ('silent_q0', 'silent_q_pos', 'verbalizing_reference')]
        ax.plot([min(vals), max(vals)], [yy, yy], color=P['rule'], lw=1.0, zorder=0)
        ax.plot(vals[0], yy, 'o', ms=5.5, color=P['muted'], mfc='none', mew=1.2)
        ax.plot(vals[1], yy, 'o', ms=6, color=P['c1'])
        ax.plot(vals[2], yy, '^', ms=6, color=P['c2'])
    ax.plot([], [], 'o', color=P['muted'], mfc='none', mew=1.2, label='Never verbalize; no other sample does (289)')
    ax.plot([], [], 'o', color=P['c1'], label='Never verbalize; another sample does (260)')
    ax.plot([], [], '^', color=P['c2'], label='Read just before verbalizing (97)')
    ax.legend(frameon=False, fontsize=8.5, loc='upper center', bbox_to_anchor=(0.45, -0.17), ncol=1, handletextpad=0.2)
    ax.set_yticks(ys); ax.set_yticklabels([NAME[k] for k in ORDER]); ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False); ax.set_xlim(-2, 70); ax.set_xlabel('Continuations flagged (%)')
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
    save(fig, 'fig_silent', svg)

for P, svg in [(SENT, True), (REAL, False)]:
    fig_eval(P, svg); fig_flags(P, svg)
