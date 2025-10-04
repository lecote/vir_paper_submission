import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def collate_and_bkgd_sub(data, value_for_pivot: str):
    """
    data: longform data as collated csv point files
    value_for_pivot: column name for creating pivot table
    """
    data['file_name'] = data['index'].str.split(':', expand=True).iloc[:,0]
    data_collate = data.pivot_table(index='file_name', columns='label', values=value_for_pivot)
    
    #background subtraction using mean of bkgd points
    data_collate['bkgd_mean'] = data_collate[['bkgd_1', 'bkgd_2','bkgd_3']].mean(axis=1)
    data_collate['vir_A_bs'] = data_collate['vir_A'] - data_collate['bkgd_mean']
    data_collate['vir_P_bs'] = data_collate['vir_P'] - data_collate['bkgd_mean']
    
    #total vir puncta intensity and percentage
    data_collate['vir_A+P'] = data_collate['vir_A_bs'] + data_collate['vir_P_bs']
    data_collate['vir_Pperc'] = data_collate['vir_P_bs']/(data_collate['vir_A_bs'] + data_collate['vir_P_bs'])
    data_collate['vir_Aperc'] = (1 - data_collate['vir_Pperc'])*100
    return data_collate


def generate_stat_table(dc, val_list, genotype_col, ctrl, expt):
    """
    iterate through data frame dc and generate normality, MannWhitU, and ttest 
    for all values in val_list for ctrl vs expt in genotype_column
    """
    control = dc[genotype_col]==ctrl
    experimental = dc[genotype_col]==expt
    stat_table = []
    for d in (val_list):
        n,np = stats.normaltest(dc[d])
        s,p = stats.mannwhitneyu(dc.loc[experimental][[d]],
                dc.loc[control][[d]])
        st,pt = stats.ttest_ind(dc.loc[experimental][[d]],
                dc.loc[control][[d]])
        stat_table.append([d,n,np,s,p,st,pt])
    stat_table = pd.DataFrame(stat_table)
    stat_table.columns = ['value',
                          'normality',
                          'normality_p',
                          'MannWhitU_stat',
                          'MannWhitU_pval',
                          'ttest_stat',
                          'ttest_pval']    
    return stat_table


#function to plot just mean as a line with width pulled from boxplot
def meanlines(data, ax, width=0.4, **vars):
    sns.boxplot(showmeans=True, 
                meanline=True, 
                meanprops={'color': 'k', 'ls': '-', 'lw': 1.5}, 
                medianprops={'visible': False},
                whiskerprops={'visible': False}, 
                showfliers=False, 
                showbox=False, 
                showcaps=False,
                zorder=10, 
                data=data, 
                **vars, 
                width=width, 
                ax=ax)

def plot_puncta(dc, vars1, vars2, order, stat_table, figsize=(3.2,3)):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.2,3))
    ax1 = sns.swarmplot(data=dc, **vars1, 
                    ax=ax1, alpha=.8, size=3,
                    order=order, hue='genotype',
                   palette=['#079885','#079885'] #teal
                   )
    meanlines(data=dc, ax=ax1, **vars1)
    sns.pointplot(data=dc, 
                  **vars1, 
                  ax=ax1,
                  color="black", 
                  linestyle='none', 
                  errorbar='sd', 
                  markers='', 
                  linewidth=0.5, 
                  order=order,
                  zorder=10)
    ax1.set_ylim(0, 100);
    ax1.set_ylabel('');
    pval = stat_table['MannWhitU_pval'].loc[stat_table['value'] == vars1['y']].reset_index(drop=True)
    ax1.set_title(vars1['y'] + "\n " + pval.name + ":\n" + str(f"{pval[0][0]:.4e}") + "\n\n", fontsize=8);

    ax2 = sns.swarmplot(data=dc, **vars2, 
                    ax=ax2, alpha=.8, size=3,
                    order=order, hue='genotype', 
                   palette=['#e36e33','#e36e33']#orange
                   )
    plt.axhline(y=50, color='grey', alpha=.5, linestyle='--')
    meanlines(data=dc, ax=ax2, **vars2)
    sns.pointplot(data=dc, **vars2, ax=ax2, 
              color="black", linestyle='none', errorbar='sd', markers='', linewidth=0.5, order=order,zorder=10)
    sns.despine()
    ax2.set_ylabel('');
    ax2.set_yticklabels('');
    ax2.set_ylim(0, 100);
    pval = stat_table['MannWhitU_pval'].loc[stat_table['value'] == vars2['y']].reset_index(drop=True)
    ax2.set_title(vars2['y'] + "\n " + pval.name + ":\n" + str(f"{pval[0][0]:.4e}") + "\n\n", fontsize=8);

    plt.tight_layout()
    plt.show()


def stacked_h_bar_graph(df, x_var, y_var, desired_geno_order, figsize=(1.0, 1.5), 
                        palette=['#b7b8b9','#757777','#db46db','#db83db'],
                        n_text_location=1.01):
    """
    From a given df, create stacked horizaontal bar graph of "y_var" values in desired_geno_order of "x_var"
    """
    #create distribution table
    distribution = pd.crosstab(df[x_var], df[y_var], normalize='index')

    # plot the cumsum, with reverse hue order
    plt.figure(figsize=figsize)

    #make stacked bargraph
    sns.barplot(data=distribution.cumsum(axis=1).stack().reset_index(name='perc'),
                y=x_var, 
                x='perc', 
                hue=y_var, 
                order = desired_geno_order,
                hue_order = distribution.columns[::-1],   # reverse hue order so that the taller bars got plotted first
                dodge=False, 
                palette=palette, #palette in reverse order
                edgecolor='none', 
                linewidth=0) 
    sns.despine()
    for i, n in enumerate(df.value_counts(x_var).reindex(desired_geno_order)):
        plt.text(n_text_location,i, str(n), va='center')

    plt.legend(
        bbox_to_anchor=(0.5, 1.12),
        loc="lower center",
        borderaxespad=0,
        frameon=False,
        ncol=4,
    )
    plt.show()


def stacked_h_bar_graph_fromdf(df,
                               figsize=(1.0, 1.5),
                               palette=['#757777','#b7b8b9','#db46db'],
                               n_text_location=1.01,
                               text_color='black',
                               text_va='center'
                              ):
    """
    From df, create stacked horizaontal bar graph of desc by genotype
    """
    df = df.set_index('genotype').astype(int)
    df = df.rename_axis('desc', axis='columns')
    
    #sum descriptions to get total animals examined
    df_p = df.div(df.sum(axis=1, numeric_only=True), axis=0)
    
    plt.figure(figsize=figsize)
    sns.barplot(data=df_p.cumsum(axis=1).stack().reset_index(name='perc'),
                y='genotype', 
                x='perc',
                hue='desc',
                hue_order = df_p.columns[::-1],   # reverse hue order so that the taller bars got plotted first
                dodge=False, 
                palette=palette, #palette in reverse order
                edgecolor='none', 
                linewidth=0
               )
    sns.despine()
    for i, n in enumerate(df.sum(axis=1, numeric_only=True)):
        plt.text(n_text_location, i, str(n), #ha='center',
                 va = text_va, color=text_color)


    plt.legend( bbox_to_anchor=(0.5, 1.12), loc="lower center", borderaxespad=1, frameon=False, ncol=4)
    plt.tight_layout()
    plt.show()
