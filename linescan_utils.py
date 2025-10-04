from pathlib import Path
from typing import Dict, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

worm_c_dict = {'001-m-r':7.38653,'001-m-i':5.14834,'011-l-r':4.40707,'011-l-i':5.30436,
               '013-r-r':4.80593,'013-r-i':5.11376,'013-beanYZ-i':4.99932,'013-beanYZ-r':2.00618,
               '020-mid-r':7.32993,'020-mid-i':5.81363,'021-mid-r':6.47617,'021-mid-i':4.30615,
               '024-top-r':6.66275,'024-top-i':8.46233,'024-bot-r':7.77041,'024-bot-i':7.52723,
               '025-top-r':8.46635,'025-top-i':7.18076,'026-top-r':5.54736,'026-top-i':4.8758,
               '026-bot-r':8.053,'026-bot-i':5.95275,'029-mid-r':6.30781,'029-mid-i':7.43901}

def clean_linescan_file(path: Path) -> Dict[str, Union[str, pd.DataFrame]]:
    '''
    Clean and format an individual puncta linescan file.
    Get worm location and channel from filename.
    Perform interpolation.
    
    Parameters
    ----------
    path: pathlib Path object
        The path to the csv data file to clean
    
    Returns
    -------
    Dict[str, Union[str, pd.DataFrame]]
        A dictionary of channel, loc, df, and df_interpol
        
    '''
    filename_pieces = path.stem.split('-')
    df = pd.read_csv(path)
    df = df[['Distance_(microns)','Gray_Value']].copy().dropna()
    maxval = df['Gray_Value'].max()
    worm_loc = "-".join(filename_pieces[1:4])
    channel = filename_pieces[4].split('.')[0]
    if worm_loc == '020-mid-i':
        df['Distance'] = -1*(df['Distance_(microns)'] - worm_c_dict[worm_loc]) #quantified this one linescan in this one image backwards
    else:
        df['Distance'] = df['Distance_(microns)'] - worm_c_dict[worm_loc]
    df['worm_loc'] = worm_loc
    df['worm'] = "-".join(filename_pieces[1:3])
    df['loc'] = filename_pieces[3]
    x_values = np.arange(-5,5,0.18) #step size of Distances in line scan is ~0.17
    y_interpol = np.interp(x_values, 
                           df['Distance'], 
                           df['Gray_Value'], 
                           left=np.nan, 
                           right=np.nan)
    df_interpol = pd.DataFrame({'worm_loc': worm_loc,
                                'loc': filename_pieces[3],
                                'worm': "-".join(filename_pieces[1:3]),
                                'x_values':x_values,
                                'y_interpol':y_interpol})
    if channel =='mNG':
        df['wow181'] = df['Gray_Value'] #*0.25 #/maxval
        df = df[['worm_loc','worm','loc','Distance','wow181']].copy()
        
    elif channel == 'RFP':
        df['wow118'] = df['Gray_Value']
        df = df[['worm_loc','Distance','wow118']].copy()
        df_interpol.rename(columns={'y_interpol': 'y_interpol_118'}, inplace=True)
    
    return {'channel': channel,
            'loc': filename_pieces[3],
            'df': df,
            'df_interpol': df_interpol}

def build_aggregate_linescan_df(linescan_dir: str) -> Dict[str, pd.DataFrame]:
    '''
    Clean and marge all puncta linescan files.
    
    Parameters
    ----------
    linescan_dir: str
        Directory with data files
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        A dictionary of dataframes
        
    '''
    data_files = Path(linescan_dir).glob('2023*.csv')
    x_values = np.arange(-5,5,0.18) #step size of Distances in line scan is ~0.17
    df_dict = {'df_int': pd.DataFrame(),
               'df_rect': pd.DataFrame(),
               'df_intR': pd.DataFrame(),
               'df_rectR': pd.DataFrame(),
               'df_interpolation_int': pd.DataFrame({'x_values':x_values}),
               'df_interpolation_intR': pd.DataFrame({'x_values':x_values}),
               'df_interpolation_rect': pd.DataFrame({'x_values':x_values}),
               'df_interpolation_rectR': pd.DataFrame({'x_values':x_values})}
    
    for file in data_files:
        
        cleaned_linescan_df = clean_linescan_file(file)
        if cleaned_linescan_df['loc']=='i':
            suffix = 'int'
        elif cleaned_linescan_df['loc']=='r':
            suffix = 'rect'    
        if cleaned_linescan_df['channel'] == 'RFP':
            suffix+='R'
        df_dict[f'df_{suffix}'] = pd.concat([df_dict[f'df_{suffix}'], cleaned_linescan_df['df']], 
                                            ignore_index=True)
        df_dict[f'df_interpolation_{suffix}'] = pd.concat([df_dict[f'df_interpolation_{suffix}'], cleaned_linescan_df['df_interpol']], 
                                                          ignore_index=True)
    
    df_intM = pd.merge(df_dict['df_int'],df_dict['df_intR'], 
                       on=['worm_loc','Distance'], how='left')
    df_rectM = pd.merge(df_dict['df_rect'], df_dict['df_rectR'], 
                        on=['worm_loc','Distance'], how='left')
    df_int_interpolationM = pd.merge(df_dict['df_interpolation_int'], df_dict['df_interpolation_intR'],
                                     on=['worm_loc','x_values'], how='left')
    df_rect_interpolationM = pd.merge(df_dict['df_interpolation_rect'], df_dict['df_interpolation_rectR'],
                                      on=['worm_loc','x_values'], how='left')
    
    return {'df_intM': df_intM,
            'df_rectM': df_rectM,
            'df_int_interpolationM': df_int_interpolationM,
            'df_rect_interpolationM': df_rect_interpolationM}

def plot_linescan_means(df: pd.DataFrame):
    '''
    Plot linescan means
    
    Parameters
    ----------
    df: pd.DataFrame
        An interpolation dataframe
    
    Returns
    -------
    None
    '''
    fig, ax1 = plt.subplots(figsize=(1,1.5))
    sns.lineplot(data=df, x='x_values',y='y_interpol', 
                 estimator=np.nanmean, errorbar='ci', color='#19b519', 
                 ax=ax1, alpha=.8, linewidth=1.25)
    sns.lineplot(data=df, x='x_values',y='y_interpol_118', 
                 estimator=np.nanmean, errorbar='ci', color='#db46db', 
                 ax=ax1, alpha=.8, linewidth=1.25)
    sns.despine(top=True, left=True, right=True, bottom=False)
    ax1.tick_params(top=False, right=False, left=False, bottom=True);
    plt.xlim(-5,5)
    ax1.set_ylim(0,200); # ax1.set_ylim(0,52); ax1.set_ylim(0,205)
    ax1.set(ylabel="intensity",xlabel='um')
    plt.show()
