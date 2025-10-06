import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker

def time2min(t, frame_rate, first4cellframe = 0, minFertTo4cell = 110):
    """go from Guan timepoints to min post fertilization
    I arrived at the 110min offset from the following:
    40-50 fert. From their GUI, all embryos tp180 -> 270min == 370 mpf 
    """
    m = (t-first4cellframe)*frame_rate + minFertTo4cell
    return m

def mpf2time(m, frame_rate, first4cellframe = 0, minFertTo4cell = 110):
    """
    go from min post fert to approximate Guan et al time point """
    t = (m - minFertTo4cell)/frame_rate + first4cellframe
    return t

def lineplot_breaknans(data, break_at_nan=True, break_at_inf=True, **kwargs):
    '''sns.lineplot by default doesn't break the line at nans or infs, 
        which can lead to misleading plots.
    See https://github.com/mwaskom/seaborn/issues/1552 
        and https://stackoverflow.com/questions/52098537/avoid-plotting-missing-values-on-a-line-plot
    
    This function rectifies this, and allows the user to specify 
        if it should break the line at nans, infs, or at both (default).
    
    Note: using this function means you can't use the `units` argument of sns.lineplot.'''

    cum_num_nans_infs = np.zeros(len(data))
    if break_at_nan: cum_num_nans_infs += np.cumsum(np.isnan(data[kwargs['y']]))
    if break_at_inf: cum_num_nans_infs += np.cumsum(np.isinf(data[kwargs['y']]))
    ax = sns.lineplot(data, **kwargs, units=cum_num_nans_infs, 
            estimator=None)  #estimator must be None when specifying units
    return ax

def parse_neighbor_file(file_name):
    """
    parse Guan et al file to find neighbors
    """
    embryo = file.stem.split("_")[1]
    time = int(file.stem.split("_")[2])
    with open(file, 'r') as infile: 
        contents = infile.readlines()
    for line in contents:
        elements = [int(i) for i in line.strip().split(",")]
        for target in elements[1:]:
            datadict['embryo'].append(embryo)
            datadict['time'].append(time)
            datadict['source'].append(elements[0])
            datadict['target'].append(target)

def parse_contact_file(contact_file, 
                       contact_interest,
                       div_info,
                      cat_to_use = 'contact_cat2'
                      ):
    """
    parse Guan et al file to find contact areas and keep only ones of interest
    """
    contact_list = list(pd.concat([contact_interest["contact_1"],contact_interest["contact_2"]], axis=0))
    contact_df = pd.read_csv(contact_file, sep=",", header=0)
    embryo = contact_file.stem.split('_')[1]
    contact_df['interaction'] = contact_df["cell1"]+"_"+contact_df["cell2"]
    contact_df_keep = contact_df[contact_df['interaction'].isin(contact_list)]
    contact_df_keep = contact_df_keep.drop(['cell1', 'cell2'], axis=1)
    #reformat as longform data
    contact_keep_long = contact_df_keep.melt(id_vars="interaction",
    var_name='time', value_name='contact')
    
    #populate longform data
    contact_keep_long['time'] = pd.to_numeric(contact_keep_long["time"]).astype('float64')
    
    #keep contacts of interest
    contact_keep_long2 = pd.merge(contact_keep_long, contact_interest, left_on='interaction', right_on='contact_1')

    #create cumsum of contact area
    contact_keep_sum = contact_keep_long2.groupby([cat_to_use, 'time']).agg(
        {'contact': lambda x: x.sum(min_count = 1)}).reset_index()
    
    #change to minutes
    contact_keep_sum['min'] = time2min(contact_keep_sum['time'], div_info.loc[embryo,'frame_rate'])
    contact_keep_sum['embryo'] = embryo
    
    return contact_keep_sum

def plt_contactArea_annotated(contact_keep_sum,
                              embryo,
                              div_info,
                              output_dir,
                              cat_to_use = 'contact_cat2',
                              plot_me = "gut_vir",
                              ymarkerpos = 16,
                              ymarkerposV = -3,
                              ms = 6):

    plt.figure(figsize=(2.25, 1))
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["font.size"] = 6


    #plot cell-cell contact
    ax = lineplot_breaknans(data=contact_keep_sum[contact_keep_sum[cat_to_use].str.contains(plot_me)], 
                        x="min",
                        y="contact", 
                        hue=cat_to_use, 
                        size=0.5, 
                        break_at_nan=True,
                        palette=["#e36e33", "#e39c78"], 
                        linestyle='-', 
                        hue_order = ['gut_virR','gut_virL'])


    plt.plot([time2min(div_info.loc[embryo,'ABprpapppDiv'], div_info.loc[embryo,'frame_rate'])],
                [ymarkerposV], 
             markerfacecolor="#e39c78", 
             markeredgecolor='#e39c78', 
             markersize=ms, 
             marker="|", 
             linestyle='')

    plt.plot([time2min(div_info.loc[embryo,'ABprpappppDiv'], div_info.loc[embryo,'frame_rate'])],
                [ymarkerposV], 
             markerfacecolor="#e36e33", 
             markeredgecolor='#e36e33', 
             markersize=ms, 
             marker="|", 
             linestyle='')

    plt.plot([time2min(div_info.loc[embryo,'gastrulationEnd'], div_info.loc[embryo,'frame_rate'])],
                ymarkerposV-3, 
             markerfacecolor="#e39c78", 
             markeredgewidth=0, 
             markersize=ms, 
             marker="$\\dotminus$", 
             linestyle='')

    #TODO get int8L/R - for now 3min before will suffice
    #add markers for cell divisions
    plt.plot([time2min(div_info.loc[embryo,'E16last']-3, div_info.loc[embryo,'frame_rate']),
             time2min(div_info.loc[embryo,'E16last'], div_info.loc[embryo,'frame_rate']),
             time2min(div_info.loc[embryo,'PstarR'], div_info.loc[embryo,'frame_rate']),
             time2min(div_info.loc[embryo,'PstarL'], div_info.loc[embryo,'frame_rate'])],
            np.ones(4)*ymarkerpos, 
             markerfacecolor="#009687", 
             markeredgecolor='#009687', 
             markersize=ms, 
             marker="|", 
             linestyle='')

    #add marker of gut polarization
    plt.plot([time2min(div_info.loc[embryo,'gutColumnar'], div_info.loc[embryo,'frame_rate'])],
                ymarkerpos-3, 
             markerfacecolor="#009687", 
             markeredgecolor="#009687",
             markersize=ms, 
             marker="$\\boxbar$", 
             linestyle='', 
             alpha=.4)

    #add marker of vir bilateral and wedge-shaped
    plt.plot([time2min(div_info.loc[embryo,'virWedge'], div_info.loc[embryo,'frame_rate'])],
                ymarkerposV, 
             markerfacecolor="#e36e33", 
             markeredgecolor="#e36e33", 
             markeredgewidth=0, 
             markersize=ms, 
             marker="$\curvearrowright$", 
             linestyle='')

    #add rectangle for int5 intercalation
    ax.add_patch(patches.Rectangle((time2min(div_info.loc[embryo,'int5start'], div_info.loc[embryo,'frame_rate']),
                                    ymarkerpos-1),div_info.loc[embryo,'int5len'],1.75,
                                   facecolor="#009687",alpha=.4))
    #annotate with embryo number
    plt.text(250, 5, 'embryo #'+embryo[-1])
    ax.add_patch(patches.Rectangle((245,4),63,4.5,edgecolor='black',facecolor='white'))


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_legend().remove()
    ax.set_ylim([-8,18])
    ax.set_xlim([245,485])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(60))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(30))
    plt.title(embryo)
    plt.savefig(output_dir+vir_gut_contact_v2_'+embryo+'.png', dpi=600, bbox_inches='tight')
        
    #plt.show()