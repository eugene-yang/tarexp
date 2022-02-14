from typing import Dict
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    _hatches = ['...', '\\\\', '', 'OO']
    _colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    _legends = [
        (mpl.patches.PathPatch([[0]], facecolor=_colors[i], edgecolor='#aaa', 
                            hatch=_hatches[i], linewidth=0, linestyle='-'), n)
        for i, n in enumerate([r'First-phase / Pos: $\alpha_p Q_t$', r'First-phase / Neg: $\alpha_n (bt - Q_t)$', 
                            r'Second-phase / Pos: $\beta_p (Q - Q_t)$', r'Second-phase / Neg: $\beta_n (\rho_t - Q + Q_t)$'])
    ]
except ImportError:
    pass

from tarexp.util import readObj
from tarexp.helper.pandas_tools import createDFfromResults


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

def cost_dynamic(run_dfs: Dict[str, pd.DataFrame], recall_targets, cost_structures,
                 after_recall_color=0.5, shape=None, figsize=None, y_thousands=True, 
                 with_hatches=False, legend_col=None, max_show_round=None,
                 **kwargs):
    run_dfs = [
        (
            name, 
            d.droplevel(list(set(d.index.names)-{'round'})).T\
             .loc[ d.columns.get_level_values('measure').str.startswith('Count') ].T
        )
        for name, d in run_dfs
    ]
    lighters = [ lighten(c, after_recall_color) for c in _colors ]

    if shape is None:
        ns = [ len(run_dfs), len(recall_targets), len(cost_structures) ]
        shape = ( int(np.prod(ns)/max(ns)), max(ns) )
    if figsize is None:
        figsize = ( shape[1]*3.7 , shape[0]*2.5+0.5 )

    fig, axes = plt.subplots( *shape, figsize=figsize, tight_layout=True )
    if shape == (1, 1):
        axes = np.array([[axes]])
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
                ax.margins(x=0)
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

    if shape[1] == 1:
        legend_col, add_lines = 1, 5
    elif shape[1] < 3:
        legend_col, add_lines = 2, 2
    else:
        legend_col, add_lines = 4, 1

    axes[-1, 0].set_xlabel(' ' + '\n'*add_lines)

    fig.legend( *zip(*_legends), ncol=legend_col, loc='lower center', bbox_to_anchor=[0.5, 0.0], frameon=False )

    return fig

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

    args = parser.parse_args()

    # load runs 
    if len(args.runs) == 1 and args.runs[0] is None:
        # load all runs in the experiment directory
        df = createDFfromResults(args.runs[1])
        run_dfs = []
        for name, d in df.groupby(level='dataset'):
            if len(d.groupby('save_path')) == 1:
                run_dfs.append(name, d.droplevels(['save_path', 'dataset']))
            else:
                run_dfs += [
                    (f"{name}_{i}", dd.droplevels(['save_path', 'dataset']))
                    for i, (_, dd) in d.groupby('save_path')
                ]
    else:
        run_dfs = []
        for i, (name, p) in enumerate(args.runs):
            if p.is_dir():
                p = p / "exp_metrics.pgz"
            run_dfs.append( (name or str(i), 
                             createDFfromResults(readObj(p)).rename_axis('round', axis=0)))
    
    # do different thing if there are different plotting function implemented 
    fig = cost_dynamic(run_dfs, **vars(args))
    
    if args.output:
        fig.savefig(args.output, dpi=200)
    else:
        plt.show()