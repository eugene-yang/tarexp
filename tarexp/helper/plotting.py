from pathlib import Path
import argparse

import numpy as np
import pandas as pd

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    pass

from tarexp.util import readObj
from tarexp.helper.pandas_tools import createDFfromResults

_hatches = ['...', '\\\\', '', 'OO']
_colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
_legends = [
    (mpl.patches.PathPatch([[0]], facecolor=_colors[i], edgecolor='#aaa', 
                           hatch=_hatches[i], linewidth=0, linestyle='-'), n)
    for i, n in enumerate([r'First-phase / Pos: $\alpha_p Q_t$', r'First-phase / Neg: $\alpha_n (bt - Q_t)$', 
                           r'Second-phase / Pos: $\beta_p (Q - Q_t)$', r'Second-phase / Neg: $\beta_n (\rho_t - Q + Q_t)$'])
]

def _runs(runstr):
    if '=' in runstr:
        runstr = runstr.split("=")
        return runstr[0], Path(runstr[1])
    return None, Path(runstr)

def _cost_structure(cs):
    cs = cs.split('-')
    assert len(cs) == 4
    return tuple(int(u) for u in cs)

def lighten(color, amount=0.5):
    import colorsys
    try:
        c = mpl.colors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mpl.colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def cost_dynamic(runs, recall_targets, cost_structures, output=None, 
                 after_recall_color=0.5, shape=None, figsize=None, y_thousands=True, 
                 with_hatches=False, legend_col=None, max_show_round=None,
                 **kwargs):
    run_dfs = []
    for i, (name, p) in enumerate(runs):
        if p.is_dir():
            p = p / "exp_metrics.pgz"
        d = createDFfromResults(readObj(p))
        d = d.T.loc[ d.columns.get_level_values('measure').str.startswith('Count') ].T
        run_dfs.append( (name or str(i), d) )
    lighters = [ lighten(c, after_recall_color) for c in _colors ]

    if shape is None:
        ns = [ len(runs), len(recall_targets), len(cost_structures) ]
        shape = ( int(np.prod(ns)/max(ns)), max(ns) )
    if figsize is None:
        figsize = ( shape[1]*3.7 , shape[0]*2.5+0.5 )

    fig, axes = plt.subplots( *shape, figsize=figsize, tight_layout=True )
    if shape[0] == 1:
        axes = axes.reshape(1, -1)

    flat_axes = axes.flatten()
    i_ax = 0

    for name, df in run_dfs:
        for rtar in recall_targets:
            d: pd.DataFrame = df[rtar]
            assert d.shape[1] == 6
            npos = d.T.loc[:, 'Count(True)', :][0].sum()
            got_target = (d['known', 'Count(True)'] >= npos*rtar).idxmax() - 1
            if y_thousands:
                d = d / 1000
            
            d = d[[('known', 'Count(True)'), ('known', 'Count(False)'), 
                   ('unknown-above-cutoff', 'Count(True)'), ('unknown-above-cutoff', 'Count(False)')]]
            
            for cs in cost_structures:
                ax: mpl.Ax = flat_axes[i_ax]
                dd_cs = d * np.array(cs)
                cs_str = ', '.join( str(u) for u in cs )

                breakpoint = dd_cs.sum(axis=1).argmin()
                dd_cs[breakpoint:max_show_round].plot.area(ax=ax, stacked=True, linewidth=0, color=lighters)
                dd_cs[:breakpoint+1].plot.area(ax=ax, stacked=True, linewidth=0, ylim=(0, dd_cs.sum(axis=1).min()*2.5),
                                               title=fr"({i_ax+1}) Run={name} $s$=$({cs_str})$", xlabel='')
                
                if with_hatches:
                    for c, h in zip(ax.collections, _hatches*2):
                        for p in c.get_paths():
                            patch = mpl.patches.PathPatch(p, hatch=h,
                                                        edgecolor='gray', lw=0, alpha=0.5, facecolor='none')
                            ax.add_patch(patch)
                ax.legend().remove()
                ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
                if got_target > 0:
                    ax.axvline(got_target, ls='--', c='#555')
            
                i_ax += 1

    for ax in axes[:, 0]:
        ax.set_ylabel(f"Total Cost{ ' (thousands)' if y_thousands else '' }")

    if legend_col is None:
        legend_col = 2 if shape[1] < 3 else 4

    if legend_col == 2:
        axes[-1, 0].set_xlabel(' \n\n')
    else:
        axes[-1, 0].set_xlabel(' \n')

    fig.legend( *zip(*_legends), ncol=legend_col, loc='lower center', bbox_to_anchor=[0.5, 0.0], frameon=False )

    if output:
        fig.savefig(output, dpi=200)
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="plotting")
    parser.add_argument('--runs', type=_runs, nargs='+', required=True)
    parser.add_argument('--recall_targets', type=float, nargs='+', default=[0.8])
    parser.add_argument('--cost_structures', type=_cost_structure, nargs='+', default=[(1,1,1,1)])
    parser.add_argument('--output', type=Path, default=None)
    
    parser.add_argument('--after_recall_color', type=float, default=0.5)
    parser.add_argument('--shape', type=int, nargs=2, default=None)
    parser.add_argument('--figsize', type=int, nargs=2, default=None)
    parser.add_argument('--y_thousands', action='store_true', default=False)
    parser.add_argument('--max_show_round', type=int, default=None)
    parser.add_argument('--with_hatches', action='store_true', default=False)
    parser.add_argument('--legend_col', type=int, default=None)

    cost_dynamic(**vars(parser.parse_args()))