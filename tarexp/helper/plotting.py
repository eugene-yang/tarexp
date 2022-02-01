colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
lighters = [ lighten_color(c, 0.5) for c in colors ]
custom_lines = [
    (mpl.lines.Line2D([0], [0], color=colors[i], linewidth=5, linestyle='-'), n)
    for i, n in enumerate(['Training / Pos', 'Training / Neg', 'Post-training / Pos', 'Post-training / Neg'])
]

csstr = lambda x: f"{x['training', 'pos']}-{x['training', 'neg']}-{x['post-training-above', 'pos']}-{x['post-training-above', 'neg']}"
allcosts_20sub = pd.concat({
    csstr(cs): pd.concat([ data_20sub[ (-1, *p) ]*uc for p,uc in cs.items() ], axis=1).droplevel(0, axis=1)
    for i, cs in enumerate(cost_structures)
}, axis=1, names=['cs']).drop('random')


def plotCostDynamic(reuslts, **kwargs):
    hatches = ['...', '\\\\', '', 'OO']

    legends = [
        (mpl.patches.PathPatch([[0]], facecolor=colors[i], edgecolor='#aaa', 
                            hatch=hatches[i], linewidth=0, linestyle='-'), n)
        for i, n in enumerate([r'First-phase / Pos: $\alpha_p Q_t$', r'First-phase / Neg: $\alpha_n (bt - Q_t)$', 
                            r'Second-phase / Pos: $\beta_p (Q - Q_t)$', r'Second-phase / Neg: $\beta_n (\rho_t - Q + Q_t)$'])
    ]
        

    dd = allcosts_20sub.loc['relevance', 'topics_GENV', 0]
    got_target = (dd['1-1-1-1', 'training', 'pos'] >= npos_20sub['topics_GENV']*0.8).idxmax() - 1


    fig, axes = plt.subplots( 1, 3, figsize=(11,3), tight_layout=True )

    for cs, ax, tt in zip(['1-1-1-1', '10-10-1-1', '25-5-5-1'], axes.flatten(), 'abc'):
        ddd = dd[cs] / 1000
        breakpoint = ddd.sum(axis=1).argmin()
        ddd[breakpoint:31].plot.area(ax=ax, stacked=True, linewidth=0, color=lighters)
        ddd[:breakpoint+1].plot.area(
            ax=ax, stacked=True, linewidth=0, 
            # ylim=(0, ylims[iseed]), xlim=(0, xlims[iseed]),
            xlim=(0, 30),
            title=fr"({tt}) $s=({', '.join(cs.split('-'))})$", xlabel='')
        
        for c, h in zip(ax.collections, hatches*2):
            for p in c.get_paths():
                patch = mpl.patches.PathPatch(p, hatch=h,
                                            edgecolor='gray', lw=0, alpha=0.5, facecolor='none')
                ax.add_patch(patch)
        
        ax.legend().remove()
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.axvline(got_target, ls='--', c='#555')

    axes[0].set_xlabel(' \n')
    axes[0].set_ylabel('Total Cost (thousands)')

    fig.legend( *zip(*legends), ncol=4, loc='upper center', bbox_to_anchor=[0.5, 0.15], frameon=False )
