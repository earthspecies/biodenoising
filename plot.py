'''
'''
import os 
import argparse 
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results_path", type=str, required=True, help="Path to results folder containing csvs with the SISDRS and filenames"
)
parser.add_argument("--num_processes",default=1,type=int,help="number of processes for multiprocessing")


def process_file(args):
    filename, conf = args
    method = filename.split('.csv')[0]
    seed = None
    if ',seed=' in filename:
        seed = int(method.split(',seed=')[1])
        method = method.split(',seed=')[0]

    print("Processing file {}".format(filename))
    #read a csv file into a pandas dataframe and return it 
    df = pd.read_csv(os.path.join(conf["results_path"],filename),usecols=[1,2,3])
    df = pd.melt(df, id_vars=["filename"], var_name="metric", value_name="dB")
    df['seed'] = seed 
    df['method'] = method
    return df.reset_index(drop=True)
    


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    
    files = [file for file in os.listdir(os.path.join(arg_dic["results_path"])) if file.endswith('.csv') and not file.startswith('.')]
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
    # import pdb; pdb.set_trace()
    fig, ax = plt.subplots()
    # sns.stripplot(data=df, x="metric", y="dB", hue="method",dodge=True, alpha=.2, legend=False, palette="dark:yellow")
    bb = sns.barplot(data=df, x="metric", y="dB", hue="method", ax=ax, palette="Blues",dodge=True,estimator=np.median)
    pp = sns.pointplot(data=df, x="metric", y="dB", hue="method", ax=ax, palette='dark:black',dodge=.64, linestyle="none", errorbar=None,estimator='mean', legend=False)
    for i in pp.containers:
        ax.bar_label(i, fmt='%.2f')
    # ax.bar_label(pp.containers[0])    
    plt.show()