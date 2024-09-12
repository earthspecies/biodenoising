'''
'''
import os 
import argparse 
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import numpy as np
import scipy
import confidence_intervals
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path", type=str, required=True, help="Path to biodenoising-validation"
)
parser.add_argument("--num_processes",default=1,type=int,help="number of processes for multiprocessing")

def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def results(df):
    methods = ['demucs_demucs_noisy_step0','demucs_demucs_noisereduce_step0','demucs_demucs_none_step0','demucs_demucs_time_scale_step0','noisereduce']
    df = df[df['method'].isin(methods)]
    print(df['method'].value_counts())
    median_df = df.groupby(['method','metric'])['dB'].apply(np.median).apply(lambda x: round(x,2)).reset_index()
    mad_df = df.groupby(['method','metric'])['dB'].apply(scipy.stats.median_abs_deviation).apply(lambda x: round(x,2)).reset_index()
    print(median_df)
    print(mad_df)
    latex_row = ' & '
    for method in methods: 
        for metric in ['sisdr','sisdri']:
            data = df[(df['method'] == method) & (df['metric'] == metric)]['dB'].to_numpy()
            median, (ci_low, ci_high) = confidence_intervals.evaluate_with_conf_int(data, np.median, labels=None, conditions=None, num_bootstraps=1000, alpha=5)
            #latex_row += '\SI{' +str(median_df[(median_df['method'] == method) & (median_df['metric'] == metric)]['dB'].values[0]) + '}{} (\SI{' + str(mad_df[(mad_df['method'] == method) & (mad_df['metric'] == metric)]['dB'].values[0])  + '}{}) & '
            latex_row += '\SI{' + str(np.round(median,2)) + '}{} $\\frac{'+ str(np.round(ci_low,2)) + '}{' + str(np.round(ci_high,2)) +'}$ & '
    print(latex_row)
        
def results_diff(df,methods):
    df = df[df['seed'] == 0]
    df1 = df[df['method'].isin(methods.values())].reset_index(drop=True)
    df2 = df[df['method'].isin(methods.keys())].reset_index(drop=True)
    df2['method'] = df2['method'].map(methods)
    
    # print(df1['metric'].value_counts())
    # for method in methods.values():
    #     print(method,scipy.stats.wilcoxon(df1[(df1['method'] == method) & (df1['metric'] == 'sisdri')]['dB'],df2[(df2['method'] == method) & (df2['metric'] == 'sisdri')]['dB']))
    #     print(method,scipy.stats.ttest_ind(df1[(df1['method'] == method) & (df1['metric'] == 'sisdr')]['dB'],df2[(df2['method'] == method) & (df2['metric'] == 'sisdr')]['dB'], equal_var=False))
    #     # # plt.hist(df2[df2['method'] == method]['dB'], bins=20, alpha=0.5, label=method)
    #     # # plt.show()
    #     scipy.stats.probplot(df2[(df2['method'] == method )& (df2['metric'] == 'sisdri')]['dB'], dist="norm", plot=plt)
    #     plt.title("shapiro {}".format(scipy.stats.shapiro(df2[(df2['method'] == method) & (df2['metric'] == 'sisdri')]['dB'])))   
    #     plt.show()
    #     # scipy.stats.probplot(df1[df1['method'] == method& df1['metric'] == 'sisdri']['dB'], dist="norm", plot=plt)
    #     # plt.title("shapiro {}".format(scipy.stats.shapiro(df1[df1['method'] == method & df1['metric'] == 'sisdri']['dB'])))   
    #     # plt.show()
        
    print(df1['method'].value_counts())
    print(df2['method'].value_counts())
    median_df = df2.groupby(['method','metric'])['dB'].apply(np.median).apply(lambda x: round(x,2)).reset_index()
    mad_df = df2.groupby(['method','metric'])['dB'].apply(scipy.stats.median_abs_deviation).apply(lambda x: round(x,2)).reset_index()
    print(median_df)
    print(mad_df)
    df1 = df1.set_index(['method','metric','filename'])
    df2 = df2.set_index(['method','metric','filename'])
    diff_df = df2.subtract(df1)
    # print(len(diff_df.reset_index()[diff_df.reset_index()['dB']<-0.1 & diff_df.reset_index()['metric'] == 'sisdri']),len(diff_df.reset_index()[diff_df.reset_index()['metric'] == 'sisdri']))
    # import pdb; pdb.set_trace()
    # diff_df = (df1.groupby(['method','metric','filename'])['dB'].mean() - df2.groupby(['method','metric','filename'])['dB'].mean()).reset_index()
    # mean_df = diff_df.groupby(['method','metric'])['dB'].apply(np.mean).apply(lambda x: round(x,2)).reset_index()
    mean_df = diff_df.groupby(['method','metric'])['dB'].apply(np.median).apply(lambda x: round(x,2)).reset_index()
    mad_df = diff_df.groupby(['method','metric'])['dB'].apply(scipy.stats.median_abs_deviation).apply(lambda x: round(x,2)).reset_index()
    print(mean_df)
    print(mad_df)
    diff_df = diff_df.reset_index()
    latex_row = ''
    for metric in ['sisdr','sisdri']:
        latex_row += '\\\\' + metric + ' & '
        for method in methods.values(): 
            #latex_row += '\SI{' +str(mean_df[(mean_df['method'] == method) & (mean_df['metric'] == metric)]['dB'].values[0]) + '}{} (\SI{' + str(mad_df[(mad_df['method'] == method) & (mad_df['metric'] == metric)]['dB'].values[0])  + '}{}) & '
            data = diff_df[(diff_df['method'] == method) & (diff_df['metric'] == metric)]['dB'].to_numpy()
            median, (ci_low, ci_high) = confidence_intervals.evaluate_with_conf_int(data, np.median, labels=None, conditions=None, num_bootstraps=1000, alpha=5)
            latex_row += '\SI{' + str(np.round(median,2)) + '}{} $\\frac{'+ str(np.round(ci_low,2)) + '}{' + str(np.round(ci_high,2)) +'}$ & '
    
    print(latex_row)

    
def process_file(args):
    filename, conf = args
    method = filename.split('.csv')[0]
    seed = None
    if ',seed=' in filename:
        seed = int(method.split(',seed=')[1].split(',')[0])
        suffix = ''
        if len(method.split(',seed=')[1].split(',')) > 1:
            suffix = ","+method.split(',seed=')[1].split(',')[1]
        method = method.split(',seed=')[0] + suffix

    print("Processing file {}".format(filename))
    #read a csv file into a pandas dataframe and return it 
    df = pd.read_csv(os.path.join(conf["subset_path"],filename),usecols=[1,2,3])
    df = pd.melt(df, id_vars=["filename"], var_name="metric", value_name="dB")
    df['seed'] = seed 
    df['method'] = method
    assert len(df) == 124 or len(df) == 2568 , "Data is missing from csv {} data shape {}".format(filename, len(df))
    return df.reset_index(drop=True)
    
def process_folder(arg_dic):
    files = [file for file in os.listdir(os.path.join(arg_dic["subset_path"])) if file.endswith('.csv') and not file.startswith('.')]
    files = sorted(files)
    assert len(files) > 0, "No csv files found in the results folder"
    if arg_dic["num_processes"] > 1:
        with Pool(processes=arg_dic["num_processes"]) as pool:
            mp_args = [[f,arg_dic] for f in files]
            results = tqdm(pool.map(process_file, mp_args), total=len(files))
            df = pd.concat(results)
    else:
        for i,f in enumerate(files):
            result = process_file([f,arg_dic])
            if i == 0:
                df = result
            else:  
                df = pd.concat([df,result])
    return df

if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    
    for subset in ["16000","16000_large","16000_snr_experiments"]:
        print("Processing subset {}".format(subset))
        if subset == "16000_snr_experiments":
            for snr in [-5,0,5,10]:
                print("Processing snr {}".format(snr))
                arg_dic["subset_path"] = os.path.join(arg_dic["dataset_path"],subset,str(snr),"results")
                df = process_folder(arg_dic)
                mean_df = df.groupby(['method','metric','filename'])['dB'].mean().reset_index()
                results(df)
        else:
            arg_dic["subset_path"] = os.path.join(arg_dic["dataset_path"],subset,"results")
            df = process_folder(arg_dic)
            mean_df = df.groupby(['method','metric','filename'])['dB'].mean().reset_index()
            results(df)
            if subset == "16000":
                print("Ablation small")
                methods = {'demucs_demucs_noisy_small_step0':'demucs_demucs_noisy_step0','demucs_demucs_noisereduce_small_step0':'demucs_demucs_noisereduce_step0','demucs_demucs_none_small_step0':'demucs_demucs_none_step0','demucs_demucs_time_scale_small_step0':'demucs_demucs_time_scale_step0'}
                results_diff(df, methods)
                print("Ablation random")
                methods = {'demucs_random_noisy_step0':'demucs_demucs_noisy_step0','demucs_random_noisereduce_step0':'demucs_demucs_noisereduce_step0','demucs_random_none_step0':'demucs_demucs_none_step0','demucs_random_time_scale_step0':'demucs_demucs_time_scale_step0'}
                results_diff(df, methods)
                print("Ablation cleanunet")
                methods = {'cleanunet_cleanunet_noisy_step0':'demucs_demucs_noisy_step0','cleanunet_cleanunet_noisereduce_step0':'demucs_demucs_noisereduce_step0','demucs_random_none_step0':'demucs_demucs_none_step0','cleanunet_cleanunet_demucs_time_scale_step0':'demucs_demucs_time_scale_step0'}
                results_diff(df, methods)
                print("Ablation time scale")
                methods = {'demucs_demucs_noisy_step0,timescale=0':'demucs_demucs_noisy_step0','demucs_demucs_noisereduce_step0,timescale=0':'demucs_demucs_noisereduce_step0','demucs_demucs_none_step0,timescale=0':'demucs_demucs_none_step0','demucs_demucs_time_scale_step0,timescale=0':'demucs_demucs_time_scale_step0'}
                results_diff(df, methods)
                print("Ablation small exclude")
                methods = {'demucs_demucs_noisy_excl_step0':'demucs_demucs_noisy_step0','demucs_demucs_noisereduce_excl_step0':'demucs_demucs_noisereduce_step0','demucs_demucs_none_excl_step0':'demucs_demucs_none_step0','demucs_demucs_time_scale_excl_step0':'demucs_demucs_time_scale_step0'}
                results_diff(df, methods)