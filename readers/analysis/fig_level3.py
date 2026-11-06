"""Level-3 figure: placeholder minus real flag rate per activation reader, 95% CI (D-182). Themed SVG for the
artifact (sentinel colors) and PDF/PNG for the README (real colors)."""
import json, re, sys, matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
L = json.load(open('level3_dev.json'))
ORDER = [('template_lens', 'Template lens', 1), ('ao', 'Activation oracle', 1), ('transfer_probe', 'Transfer probe', 1),
         ('olens', 'Oracle lens', 2), ('nla', 'NLA', 2), ('jlens', 'J-lens', 2), ('logit_lens', 'Logit lens', 2), ('jpp', 'J++ lens', 2)]
MK = {1: 's', 2: '^'}
def draw(ink, muted, c, name, fmt):
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['cmr10'], 'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True, 'svg.fonttype': 'path', 'font.size': 10, 'text.color': ink,
        'axes.labelcolor': ink, 'xtick.color': ink, 'ytick.color': ink, 'axes.edgecolor': muted, 'axes.linewidth': 0.6,
        'ytick.major.width': 0, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
        'figure.facecolor': 'none' if fmt == 'svg' else 'white', 'axes.facecolor': 'none'})
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ax.axvline(0, color=muted, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ys, y = [], 0
    for i, (_, _, g) in enumerate(ORDER):
        if i and ORDER[i - 1][2] != g: y += 0.6
        ys.append(-y); y += 1
    for (k, lab, g), yy in zip(ORDER, ys):
        r = L['primary'].get(k) or L['secondary'][k]
        lo, hi = [100 * v for v in r['ci']]
        ax.plot([lo, hi], [yy, yy], color=c[g], lw=1.6, solid_capstyle='round')
        ax.plot(100 * r['diff'], yy, MK[g], color=c[g], ms=6.5, mec='none')
        ax.text(1.03, yy, f"{100 * r['flag_placeholder']:.1f}% vs {100 * r['flag_real']:.1f}%", va='center', ha='left',
                fontsize=8.5, color=muted, transform=ax.get_yaxis_transform())
    ax.set_yticks(ys); ax.set_yticklabels([o[1] for o in ORDER])
    ax.set_xlim(-6.5, 13); ax.set_xlabel('Placeholder minus real flag rate (percentage points)')
    ax.text(1.03, ys[0] + 0.8, 'Placeholder vs real', ha='left', transform=ax.get_yaxis_transform(), fontsize=8.5, color=muted)
    ax.text(-6.3, ys[0] + 0.8, 'Numeric score', fontsize=8.5, color=muted)
    ax.text(-6.3, ys[3] + 0.8, 'Words scored by judge', fontsize=8.5, color=muted)
    ax.set_ylim(ys[-1] - 0.6, ys[0] + 1.3)
    fig.savefig(name, format=fmt, bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.close(fig)
    if fmt == 'svg':
        s = open(name).read()
        for a, b in [(ink, 'var(--ink)'), (muted, 'var(--muted)'), (c[1], 'var(--c2)'), (c[2], 'var(--c3)')]:
            s = s.replace(a, b)
        s = re.sub(r'<\?xml.*?\?>\s*', '', s, flags=re.S); s = re.sub(r'<!DOCTYPE.*?>\s*', '', s, flags=re.S)
        s = re.sub(r'(<svg[^>]*?) width="[^"]*" height="[^"]*"', r'\1', s, count=1)
        open(name, 'w').write(s)
draw('#010101', '#020202', {1: '#0a0a02', 2: '#0a0a03'}, 'fig_level3.svg', 'svg')
for f in ['pdf', 'png']:
    draw('#1d2430', '#5f6876', {1: '#E69F00', 2: '#009E73'}, f'fig_level3.{f}', f)
print('ok')
