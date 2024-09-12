import argparse
import json
import logging
import os
import sys
import biodenoising
import pandas as pd
import numpy as np
import math
import random
import julius
import torchaudio 

parser = argparse.ArgumentParser(
        'prepare_experiments',
        description="Generate json files with all the audios")
parser.add_argument("--data_dir", type=str, default="enhanced", required=True,
                    help="directory where the training data is containing train, valid and test subdirectories")
parser.add_argument("--step", default=0, type=int, help="step")
parser.add_argument("--method",choices=["demucs", "cleanunet"], default="demucs",help="Method that was used to denoise at step 'step'")
parser.add_argument("--approach",choices=["noisy2cleaned","noisier2noisy","noisereduce"], default="noisy2cleaned",help="Our approach vs using noisy files and noise")
parser.add_argument("--tag", default="",help="This is used to tag the models at steps>0 with the origin of training data at step 0")
parser.add_argument("--seed", default=-1, type=int, help="seed for step>0")
parser.add_argument("--transform",choices=["none", "time_scale", "all"], default="none",help="Transform input by pitch shifting or time scaling")
parser.add_argument("--dataprune",choices=["none","kmeans"], default="none",help="Data pruning method using the model's saved activations.")
parser.add_argument('--prune_ratio', type=float, default=1., help="use the prune ratio from all the files")  
parser.add_argument('--with_previous', action="store_true",help="add previous steps")
parser.add_argument('--with_lower_sr', action="store_true",help="add 16kHz data to the 48kHz experiments")
parser.add_argument('--balance', action="store_true",help="downsample the larger datasets to match the smaller ones")
parser.add_argument('--num_valid', type=int, default=0, help="number of validation files")
parser.add_argument('--use_ratio', type=float, default=1., help="use the top ratio of the files")  
parser.add_argument('--version', type=str, default='')
parser.add_argument('--exclude', nargs='+', default=[])
parser.add_argument('--overfit', action="store_true",help="oracle ablation")

test_sets = {'biodenoising': 'biodenoising_validation', 'crows':'carrion_crows_denoising','zfinches': 'zebra_finch_denoising'}  

def to_json_folder(data_dict, args):
    json_dict = {'train':[], 'valid':[]}
    for split, dirs in data_dict.items():
        for d in dirs:
            meta=biodenoising.denoiser.audio.find_audio_files(d)
            if args.approach=='noisy2cleaned' and 'valid' not in data_dict.keys() and split=='train':
                random.shuffle(meta)
                if args.num_valid > 0:
                    json_dict['valid'] += meta[:args.num_valid]
                json_dict['train'] += meta[args.num_valid:]
            else:
                json_dict[split] += meta
    return json_dict

def to_json_list(data_dict,args, json_dict={}):
    for split, filelist in data_dict.items():
        meta=biodenoising.denoiser.audio.build_meta(filelist)
        if split not in json_dict.keys():
            json_dict[split] = []
        json_dict[split].extend(meta)
    return json_dict

def write_json(json_dict, filename, args):
    for split, meta in json_dict.items():       
        exp_dirname = args.method + '_' + args.transform + args.tag + '_step' + str(args.step)
        if args.dataprune!='none':
            exp_dirname += '_'+args.dataprune + '_'+str(args.prune_ratio)
        if args.with_previous:
            exp_dirname += '_prev'
        if args.seed>=0:
            exp_dirname += ',seed='+str(args.seed)
        if args.approach=='noisier2noisy':
            exp_dirname = 'noisy'
        elif args.approach=='noisereduce':
            exp_dirname = 'noisereduce'
        out_dir = os.path.join('biodenoising','egs', os.path.basename(os.path.normpath(args.data_dir)), exp_dirname, split)
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, filename)
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=4)
            
def resample(inpath_file, outdir, to_samplerate):
    """Convert audio from a given samplerate to a target one """
    out_file = os.path.basename(inpath_file)
    if not os.path.exists(os.path.join(outdir, out_file)):
        # read wav with torchaudio 
        wav, sr = torchaudio.load(inpath_file)
        wav = julius.resample_frac(wav, sr, to_samplerate)
        # write wav with torchaudio
        torchaudio.save(os.path.join(outdir, out_file), wav, to_samplerate)
    return os.path.join(outdir, out_file)

def generate_json(args):
    if args.with_previous:
        steps = range(0, args.step+1)
    else:
        steps = [args.step]
    if args.with_lower_sr and args.data_dir.endswith('48k'):
        data_dirs = [args.data_dir, args.data_dir.replace('48k','16k')]
    else:
        data_dirs = [args.data_dir]
    if args.dataprune!='none':
        mds = [[] for i, data_dir in enumerate(data_dirs)]
        for i, data_dir in enumerate(data_dirs):
            for step in steps:
                experiment = 'clean_'+args.method+ '_' + args.transform + args.tag +'_step'+str(args.step)+args.version
                if args.seed>=0:
                    experiment += ',seed='+str(args.seed)
                md = pd.read_csv(os.path.join(args.data_dir, experiment+".csv"))
                nclusters = len(md['dataset'].unique())
                prune: Prune = biodenoising.datasets.dataprune.Prune(
                    prune_type=True,
                    ssl_type=args.dataprune,
                    number_cluster='auto',
                    random_state=42,
                    dataframe=md,
                    data_frame_clustered=True
                )
                md = prune.prune(prune_fraction=args.prune_ratio)
                mds[i].append(md)
    else:
        mds = None
                
    if args.transform!='all':
        if args.approach=='noisy2cleaned':
            clean_dirs_dict = {split:[] for split in ['train','valid']}
            clean_dirs={'train':[]}
            for i, data_dir in enumerate(data_dirs):
                for step in steps:
                    if mds is not None:
                        md = mds[i][step]
                    else:
                        experiment = 'clean_'+args.method+ '_' + args.transform + args.tag +'_step'+str(args.step)+args.version
                        if args.seed>=0:
                            experiment += ',seed='+str(args.seed)
                        md = pd.read_csv(os.path.join(data_dir, experiment+".csv"))
                    counts = md['dataset'].value_counts()
                    counts_filter = counts[counts>1000] ### for stats remove datasets with less than 1000 files
                    if len(counts_filter)==0:
                        counts = counts
                        filter_flag = False
                    else:
                        counts = counts_filter
                        filter_flag = True
                    magnitude = int(math.log10(counts.median()))+1
                    for col in md['dataset'].unique():
                        if all(not col.startswith(x) for x in args.exclude):
                            ds = md[md['dataset']==col]
                            ds.sort_values(by='metric',ascending=False,ignore_index=True)
                            filenames = ds['fn'].values.tolist()
                            col_magnitude = int(math.log10(len(filenames)))+1
                            if args.balance and filter_flag and col_magnitude>magnitude:
                                reduced = int(len(filenames)/ (10**(col_magnitude-magnitude)))
                                print("Reducing dataset size for {} from {} to {}".format(col, len(filenames), reduced))
                                filenames = filenames[:reduced]
                            filenames = [f for f in filenames if os.path.exists(f)]
                            if i==1: ### resample to 48kHz from 16kHz
                                experiment = 'clean_16_'+args.method+ '_' + args.transform + args.tag +'_step'+str(step)+args.version
                                if args.seed>=0:
                                    experiment += ',seed='+str(args.seed)
                                out_dir = os.path.join(args.data_dir,'train', experiment)
                                os.makedirs(out_dir, exist_ok=True)
                                filenames = [resample(f, out_dir, 48000) for f in filenames]
                            if len(filenames)>args.num_valid:
                                if args.num_valid>0:
                                    clean_dirs_dict['valid'].extend(filenames[:args.num_valid])
                                clean_dirs_dict['train'].extend(filenames[args.num_valid:int(len(filenames)*args.use_ratio)])
                            else:
                                clean_dirs_dict['train'].extend(filenames)

                if os.path.exists(os.path.join(args.data_dir, 'train','clean')):
                    clean_dirs={'train':[os.path.join(args.data_dir, 'train','clean',f) for f in os.listdir(os.path.join(args.data_dir, 'train','clean'))]}
        elif args.approach=='noisereduce':
            clean_dirs = {split:[os.path.join(args.data_dir,'train','noisereduce')] for split in ['train']}
        else:
            clean_dirs = {split:[os.path.join(args.data_dir,'dev',f) for f in os.listdir(os.path.join(args.data_dir,'dev'))] for split in ['train']}
        noise_dirs_dict = {split:[os.path.join(args.data_dir, split,'noise',f) for f in os.listdir(os.path.join(args.data_dir, split,'noise'))] for split in ['train']}
    else: ### use all data
        if args.approach=='noisy2cleaned':
            clean_dirs_dict = {split:[] for split in ['train','valid']}
            clean_dirs={'train':[]}
            for transforms in ['none','time_scale']:
                experiment = 'clean_'+args.method+ '_' + transforms + args.tag +'_step'+str(args.step)+args.version
                if args.seed>=0:
                    experiment += ',seed='+str(args.seed)
                md = pd.read_csv(os.path.join(args.data_dir, experiment+".csv"))
                for col in md['dataset'].unique():
                    if all(not col.startswith(x) for x in args.exclude):
                        ds = md[md['dataset']==col]
                        ds.sort_values(by='metric',ascending=False,ignore_index=True)
                        filenames = ds['fn'].values.tolist()
                        if len(filenames)>args.num_valid:
                            if args.num_valid>0:
                                clean_dirs_dict['valid'].extend(filenames[:args.num_valid])
                            clean_dirs_dict['train'].extend(filenames[args.num_valid:int(len(filenames)*args.use_ratio)])
                        else:
                            clean_dirs_dict['train'].extend(filenames)
                if os.path.exists(os.path.join(args.data_dir, 'train','clean')):
                    clean_dirs={'train':[os.path.join(args.data_dir, 'train','clean',f) for f in os.listdir(os.path.join(args.data_dir, 'train','clean'))]}
        elif args.approach=='noisereduce':
            clean_dirs = {split:[os.path.join(args.data_dir,'train','noisereduce')] for split in ['train']}
        else:
            clean_dirs = {split:[os.path.join(args.data_dir,'dev',f) for f in os.listdir(os.path.join(args.data_dir,'dev'))] for split in ['train']}              
        noise_dirs_dict = {split:[os.path.join(args.data_dir, split,'noise',f) for f in os.listdir(os.path.join(args.data_dir, split,'noise'))] for split in ['train']}
    if args.overfit:
        for td in test_sets.values():
            sample_rate = 16000 if args.data_dir.endswith('16k') else 48000
            test_dir = os.path.join(os.path.dirname(args.data_dir), td, str(sample_rate))
            if os.path.exists(test_dir):
                clean_dirs['train'].extend([os.path.join(test_dir, 'clean', f) for f in os.listdir(test_dir,'clean')])
                noise_dirs_dict['train'].extend([os.path.join(test_dir, 'noise', f) for f in os.listdir(test_dir, 'noise')])
    json_dict = to_json_folder(clean_dirs, args)
    if args.approach=='noisy2cleaned':
        json_dict = to_json_list(clean_dirs_dict, args, json_dict)
    write_json(json_dict, 'clean.json', args)
    # ### match noise diversity to the clean diversity in the validation set
    # len_valid_clean = len(json_dict['valid']) if 'valid' in json_dict.keys() else 0
    # args.num_valid = int(len_valid_clean // len(noise_dirs_dict['train'])) if len_valid_clean>0 else 0
    json_dict = to_json_folder(noise_dirs_dict, args)
    write_json(json_dict, 'noise.json', args)
    
if __name__ == "__main__":
    args = parser.parse_args()
    generate_json(args)

#### python prepare_experiments.py --data_dir /home/marius/data/biodenoising48k/
#### python prepare_experiments.py --data_dir /home/marius/data/biodenoising48k/ --step 1 --method demucs --transform none