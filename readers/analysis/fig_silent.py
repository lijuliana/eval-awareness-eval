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
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.9), sharey=True, gridspec_kw={'width_ratios': [1.25, 1], 'wspace': 0.12})
    ys, y = [], 0
    for k in ORDER:
        if k == 'llm_reader': y += 0.7
        ys.append(-y); y += 1
    F = D['flag_rate']; h = 0.36
    ax = axes[0]
    for k, yy in zip(ORDER, ys):
        a, b = 100 * F[k]['silent_q0']['value'], 100 * F[k]['silent_q_pos']['value']
        ax.barh(yy + h / 2, a, height=h, color=P['rule'], edgecolor='none')
        ax.barh(yy - h / 2, b, height=h, color=P['c1'], edgecolor='none')
        ax.text(b + 0.6, yy - h / 2, f"{b:.0f}", va='center', fontsize=7.5, color=P['muted'])
        ax.text(a + 0.6, yy + h / 2, f"{a:.0f}", va='center', fontsize=7.5, color=P['muted'])
    ax.set_xlim(0, 36); ax.set_xlabel('Flagged (%)')
    ax.set_title('(a) Continuations that never verbalize awareness', fontsize=9, loc='left', color=P['ink'])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=P['rule'], label='No other continuation from the cut verbalizes it (289)'),
                       Patch(color=P['c1'], label='Another continuation from the cut verbalizes it (260)')], frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.16), handlelength=1.2)
    ax = axes[1]
    for k, yy in zip(ORDER, ys):
        v = 100 * F[k]['verbalizing_reference']['value']
        ax.barh(yy, v, height=h * 1.4, color=P['c2'], edgecolor='none')
        ax.text(v + 1, yy, f"{v:.0f}", va='center', fontsize=7.5, color=P['muted'])
    ax.set_xlim(0, 80); ax.set_xlabel('Flagged (%)')
    ax.set_title('(b) Just before a continuation verbalizes it (97)', fontsize=9, loc='left', color=P['ink'])
    axes[0].set_yticks(ys); axes[0].set_yticklabels([NAME[k] for k in ORDER])
    for a in axes:
        a.tick_params(axis='y', length=0); a.spines['left'].set_visible(False)
    axes[0].set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
    save(fig, 'fig_silent', svg)

for P, svg in [(SENT, True), (REAL, False)]:
    fig_eval(P, svg); fig_flags(P, svg)
