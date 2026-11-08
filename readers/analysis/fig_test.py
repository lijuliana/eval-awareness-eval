"""Tab 2 test figure: AUROC and rho with q per reader, validation (open) vs test (filled), 95% cluster CIs."""
import json, re, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
T = json.load(open('reader_analysis_test.json'))['vs_q']; V = json.load(open('reader_analysis.json'))['vs_q']
SENT = {'ink': '#010101', 'muted': '#020202', 'rule': '#030303', 'c1': '#0a0a01', 'c2': '#0a0a02', 'c3': '#0a0a03'}
REAL = {'ink': '#1d2430', 'muted': '#5f6876', 'rule': '#cfd4dc', 'c1': '#0072B2', 'c2': '#E69F00', 'c3': '#009E73'}
NAME = {'llm_reader': 'Claude, prompt + reasoning', 'prompt_only_llm': 'Claude, prompt only', 'olens': 'Oracle lens',
        'logit_lens': 'Logit lens', 'ao': 'Activation oracle', 'jlens': 'J-lens', 'jpp': 'J++ lens', 'nla': 'NLA',
        'template_lens': 'Template lens', 'transfer_probe': 'Transfer probe'}
TEXT = ['llm_reader', 'prompt_only_llm']
ACT = sorted([k for k in NAME if k not in TEXT], key=lambda k: -T[k]['auroc']['value'])
ORDER = TEXT + ACT

def style(P, svg):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True, 'svg.fonttype': 'path', 'font.size': 10, 'text.color': P['ink'],
        'axes.labelcolor': P['ink'], 'xtick.color': P['ink'], 'ytick.color': P['ink'], 'axes.edgecolor': P['muted'],
        'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'none' if svg else 'white', 'axes.facecolor': 'none'})

def save(fig, stem, svg):
    if svg:
        fig.savefig(stem + '.svg', format='svg', bbox_inches='tight', pad_inches=0.03)
        s = open(stem + '.svg').read()
        for k, v in SENT.items(): s = s.replace(v, f'var(--{k})')
        s = re.sub(r'<\?xml.*?\?>\s*', '', s, flags=re.S); s = re.sub(r'<!DOCTYPE.*?>\s*', '', s, flags=re.S)
        s = re.sub(r'(<svg[^>]*?) width="[^"]*" height="[^"]*"', r'\1', s, count=1)
        open(stem + '.svg', 'w').write(s)
    else:
        for f in ['pdf', 'png']: fig.savefig(f'{stem}.{f}', format=f, bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close(fig)

def fig(P, svg):
    style(P, svg)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.6), sharey=True, gridspec_kw={'wspace': 0.08})
    ys = []; y = 0
    for i, k in enumerate(ORDER):
        if i == len(TEXT): y += 0.8
        ys.append(-y); y += 1
    for ax, m, ref, lab in [(axes[0], 'auroc', 0.5, 'AUROC, aware ($q>0$) vs unaware'), (axes[1], 'rho', 0.0, r'Spearman $\rho$ with $q$')]:
        ax.axvline(ref, color=P['muted'], lw=0.8, ls=(0, (3, 3)), zorder=0)
        for k, yy in zip(ORDER, ys):
            col = P['c2'] if k in TEXT else P['c1']
            for d, dy, filled in [(V, 0.17, False), (T, -0.17, True)]:
                r = d[k][m]; lo, hi = r['ci95']
                ax.plot([lo, hi], [yy + dy] * 2, color=col, lw=1.4, solid_capstyle='round', alpha=1 if filled else 0.55)
                ax.plot(r['value'], yy + dy, 'o', ms=5.5, color=col, mfc=col if filled else 'none', mew=1.2)
        ax.set_xlabel(lab)
        ax.tick_params(axis='y', length=0)
    axes[0].set_xlim(0.4, 0.9); axes[1].set_xlim(-0.15, 0.6)
    axes[0].set_yticks(ys); axes[0].set_yticklabels([NAME[k] for k in ORDER])
    for ax in axes: ax.spines['left'].set_visible(False)
    axes[1].plot([], [], 'o', color=P['muted'], mfc='none', mew=1.2, label='Validation (338)')
    axes[1].plot([], [], 'o', color=P['muted'], label='Test (443)')
    axes[1].legend(frameon=False, fontsize=8.5, loc='lower right', handletextpad=0.2)
    axes[0].set_ylim(ys[-1] - 0.6, ys[0] + 0.6)
    save(fig, 'fig_test', svg)

for P, svg in [(SENT, True), (REAL, False)]: fig(P, svg)
print('ok')
