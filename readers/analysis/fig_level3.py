"""README Figures 3 to 6: verbalization by prompt version, placeholder-minus-real flag rates with text-reader controls,
content of flagged readouts, and flag rate by how far awareness reaches the text. Run from a folder holding
level3_dev.json, gradient_counts.json and flag_content.json. Themed SVG for the artifact, PDF/PNG for the README."""
import json, re, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

L = json.load(open('level3_dev.json'))
G = json.load(open('gradient_counts.json'))
SENT = {'ink': '#010101', 'muted': '#020202', 'rule': '#030303', 'c1': '#0a0a01', 'c2': '#0a0a02', 'c3': '#0a0a03'}
REAL = {'ink': '#1d2430', 'muted': '#5f6876', 'rule': '#cfd4dc', 'c1': '#0072B2', 'c2': '#E69F00', 'c3': '#009E73'}

def style(P, svg):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True, 'svg.fonttype': 'path', 'font.size': 10, 'text.color': P['ink'],
        'axes.labelcolor': P['ink'], 'xtick.color': P['ink'], 'ytick.color': P['ink'], 'axes.edgecolor': P['muted'],
        'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'figure.facecolor': 'none' if svg else 'white', 'axes.facecolor': 'none'})

def save(fig, stem, svg):
    if svg:
        name = stem + '.svg'
        fig.savefig(name, format='svg', bbox_inches='tight', pad_inches=0.03)
        s = open(name).read()
        for k, v in SENT.items():
            s = s.replace(v, f'var(--{k})')
        s = re.sub(r'<\?xml.*?\?>\s*', '', s, flags=re.S); s = re.sub(r'<!DOCTYPE.*?>\s*', '', s, flags=re.S)
        s = re.sub(r'(<svg[^>]*?) width="[^"]*" height="[^"]*"', r'\1', s, count=1)
        open(name, 'w').write(s)
    else:
        for f in ['pdf', 'png']:
            fig.savefig(f'{stem}.{f}', format=f, bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close(fig)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h)

# 1. Verbalization by prompt version
def fig_cue(P, svg):
    style(P, svg)
    fig, ax = plt.subplots(figsize=(6.3, 1.9))
    rows = [('Development traces that verbalize\nbeing tested (1,470 per version)', (134, 1470), (41, 1470)),
            ('Validation cut points where at least\n1 of 8 continuations verbalizes it', (39, 169), (10, 169))]
    for i, (lab, h, r) in enumerate(rows):
        y = -i
        for (k, n), dy, col, mk, name in [(h, 0.17, P['c1'], 'o', 'Placeholder'), (r, -0.17, P['muted'], 'D', 'Real')]:
            p, lo, hi = wilson(k, n)
            ax.plot([lo, hi], [y + dy] * 2, color=col, lw=1.6, solid_capstyle='round')
            ax.plot(p, y + dy, mk, color=col, ms=6.5, mec='none')
            ax.text(hi + 0.8, y + dy, f'{name} {p:.1f}%', va='center', fontsize=8.5, color=col if col != P['muted'] else P['muted'])
    ax.set_yticks([0, -1]); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 40); ax.set_ylim(-1.5, 0.5)
    ax.set_xlabel('Percent (95% interval)')
    save(fig, 'fig_cue', svg)

# 2. Level-3 differences with controls
ORDER = [('llm_reader', 'LLM reader, prompt + reasoning', 0, 'controls'), ('prompt_only_llm', 'LLM reader, prompt only', 0, 'controls'),
         ('template_lens', 'Template lens', 1, 'secondary'), ('ao', 'Activation oracle', 1, 'secondary'),
         ('transfer_probe', 'Transfer probe', 1, 'secondary'),
         ('olens', 'Oracle lens', 2, 'primary'), ('nla', 'NLA', 2, 'primary'), ('jlens', 'J-lens', 2, 'primary'),
         ('logit_lens', 'Logit lens', 2, 'primary'), ('jpp', 'J++ lens', 2, 'primary')]
HEAD = {0: 'Text readers (controls)', 1: 'Activation, numeric score', 2: 'Activation, words scored by a judge'}
def fig_level3(P, svg):
    style(P, svg)
    C = [P['c1'], P['c2'], P['c3']]; MK = ['o', 's', '^']
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    ax.axvline(0, color=P['muted'], lw=0.8, ls=(0, (3, 3)), zorder=0)
    ys, y = [], 0
    for i, o in enumerate(ORDER):
        if i and ORDER[i - 1][2] != o[2]: y += 0.9
        ys.append(-y); y += 1
    for (k, lab, g, grp), yy in zip(ORDER, ys):
        r = L[grp][k]; lo, hi = [100 * v for v in r['ci']]
        if hi - lo > 1e-9:
            ax.plot([lo, hi], [yy, yy], color=C[g], lw=1.6, solid_capstyle='round')
        ax.plot(100 * r['diff'], yy, MK[g], color=C[g], ms=6.5, mec='none')
        ax.text(1.03, yy, f"{100 * r['flag_placeholder']:.1f}% vs {100 * r['flag_real']:.1f}%", va='center', ha='left',
                fontsize=8.5, color=P['muted'], transform=ax.get_yaxis_transform())
    for g, first in [(0, 0), (1, 2), (2, 5)]:
        ax.text(0.5, ys[first] + 0.75, HEAD[g], fontsize=8.5, color=P['muted'])
    ax.text(1.03, ys[0] + 0.75, 'Placeholder vs real', fontsize=8.5, color=P['muted'], transform=ax.get_yaxis_transform())
    ax.annotate('zero by construction', (0, ys[0]), xytext=(1.2, ys[0]), va='center', fontsize=8, color=P['muted'])
    ax.set_yticks(ys); ax.set_yticklabels([o[1] for o in ORDER]); ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(-8, 26); ax.set_ylim(ys[-1] - 0.6, ys[0] + 1.3)
    ax.set_xlabel('Placeholder minus real flag rate (percentage points)')
    save(fig, 'fig_level3', svg)

# 3. Flag rate by how close awareness is to being verbalized
GROUPS = [('real_quiet', 'Real\nnothing in text'), ('hypo_quiet', 'Placeholder\nnothing in text'),
          ('hypo_hint', 'Placeholder\ntext-reader hint'), ('hypo_later', 'Placeholder\nverbalized later'),
          ('real_later', 'Real\nverbalized later')]
RD = [('olens', 'Oracle lens', '^', -0.18), ('nla', 'NLA', 'v', 0.0), ('jlens', 'J-lens', 'P', 0.18)]
def fig_gradient(P, svg):
    style(P, svg)
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    ax.axvspan(-0.5, 1.5, color=P['rule'], alpha=0.35, lw=0, zorder=0)
    shade = [P['c3'], P['ink'], P['muted']]
    for j, (rd, lab, mk, dx) in enumerate(RD):
        for i, (g, _) in enumerate(GROUPS):
            k, n = G[g][rd]; p, lo, hi = wilson(k, n)
            ax.plot([i + dx] * 2, [lo, hi], color=shade[j], lw=1.2, alpha=0.8)
            ax.plot(i + dx, p, mk, color=shade[j], ms=6, mec='none', label=lab if i == 0 else None)
    n = [G[g]['olens'][1] if g != 'real_quiet' else G[g]['nla'][1] for g, _ in GROUPS]
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([f'{lab}\n(n = {m})' for (_, lab), m in zip(GROUPS, n)], fontsize=8.5)
    ax.tick_params(axis='x', length=0)
    ax.text(0.5, 47, 'never verbalized (level 3)', ha='center', fontsize=8.5, color=P['muted'])
    ax.set_ylabel('Cut points flagged (%)'); ax.set_ylim(0, 50); ax.set_xlim(-0.5, 4.5)
    ax.legend(frameon=False, fontsize=8.5, loc='upper right', ncol=3, handletextpad=0.2, columnspacing=1.0)
    save(fig, 'fig_gradient', svg)

for P, svg in [(SENT, True), (REAL, False)]:
    fig_cue(P, svg); fig_level3(P, svg); fig_gradient(P, svg)
print('ok')

# 4. What the flagged readouts say (never-verbalized cuts)
def fig_content(P, svg):
    style(P, svg)
    items = json.load(open('flag_content.json'))['items']
    cats = [('scenario', 'Test environment\nor simulation'), ('genre', 'Reasoning called an\nevaluation document'),
            ('suspicion', 'Suspicion or\nfictionality')]
    cols = {'scenario': P['c2'], 'genre': P['muted'], 'suspicion': P['rule']}
    fig, ax = plt.subplots(figsize=(6.3, 1.9))
    for y, (cond, lab, ncut) in enumerate([('HYPO', 'Placeholder', 104), ('REAL', 'Real', 159)]):
        left = 0
        for c, _ in cats:
            v = 100 * sum(1 for x in items if x[3] == cond and x[4] == c) / ncut
            if v:
                ax.barh(-y, v, left=left, height=0.55, color=cols[c], edgecolor='none')
                if cond == 'HYPO':
                    ax.text(left + v / 2, -y + 0.42, dict(cats)[c], ha='center', va='bottom', fontsize=7.5, color=P['ink'])
            left += v + 0.15
        nflag = sum(1 for x in items if x[3] == cond)
        ax.text(left + 0.4, -y, f'{nflag} readouts on {ncut} cuts', va='center', fontsize=8.5, color=P['muted'])
    ax.text(30, 0.75, 'States the model itself\nis being tested: 0 readouts', ha='right', va='bottom', fontsize=8.5, color=P['ink'])
    ax.set_yticks([0, -1]); ax.set_yticklabels(['Placeholder', 'Real']); ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 30); ax.set_ylim(-1.45, 1.25)
    ax.set_xlabel('Flagged readouts per 100 cuts (oracle lens, NLA and J-lens pooled)')
    save(fig, 'fig_content', svg)

for P, svg in [(SENT, True), (REAL, False)]:
    fig_content(P, svg)
