import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import os
import sys
import pandas as pd

import torch
import torchaudio
import torchmetrics
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio
import noisereduce
import librosa
import numpy as np
import scipy
import biodenoising

logger = logging.getLogger(__name__)


def add_flags(parser):
    """
    Add the flags for the argument parser that are related to model loading and evaluation"
    """
    #biodenoising.denoiser.pretrained.add_model_flags(parser)
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--num_workers', type=int, default=1)


parser = argparse.ArgumentParser(
        'denoise',
        description="Generate denoised files")
add_flags(parser)
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--sample_rate", default=16000, type=int, help="sample_rate")
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="more logging")
parser.add_argument('--filter_silent', action="store_true",help="filter silent examples based on peaks")
parser.add_argument("--data_dir", type=str, default=None,
                    help="path to the parent directory containing subdirectories named clean, noisy, denoised (this one containing subdirs with all methods) ")

def get_peaks(wav, sample_rate, smoothing_window=6, peak_window=10):
    wav = noisereduce.reduce_noise(y=wav, sr=sample_rate)
    spec = librosa.magphase(librosa.stft(wav, n_fft=2048, hop_length=512, win_length=2048, window=np.ones, center=True))[0]
    frames2time = 512/sample_rate
    rms = librosa.feature.rms(S=spec).squeeze()
    rms =  np.nan_to_num(rms)
    if hasattr(rms, "__len__"):
        if smoothing_window>len(rms):
            smoothing_window = len(rms)//3
        if smoothing_window>1:
            rms = scipy.signal.savgol_filter(rms, smoothing_window, 2) # window size 3, polynomial order 2
        ### compute peaks in both channels
        if peak_window>len(rms):
            peak_window = len(rms)//2
        peaks, _ = scipy.signal.find_peaks(rms, height=0.01, distance=peak_window)
        allowed = int(3 * sample_rate / 512)
        peaks =  peaks * frames2time
    else:
        peaks = np.array([])
    return peaks

class EvaluationSet(torch.utils.data.Dataset):
    def __init__(self, data_dir, subdir, sample_rate=None):
        """
        """
        self.files = biodenoising.denoiser.audio.find_audio_files(os.path.join(data_dir,'clean'))
        self.sample_rate = sample_rate
        self.subdir = subdir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file, _ = self.files[index]
        clean, sr = torchaudio.load(str(file))
        assert sr == self.sample_rate, f"Expected {file} to have sample rate of {self.sample_rate}, but got {sr}"
        noisy, sr = torchaudio.load(str(file.replace('clean','noisy')))
        denoised, sr = torchaudio.load(str(file.replace('clean',"denoised"+os.sep+self.subdir)))
        assert sr == self.sample_rate, f"Expected {file} to have sample rate of {self.sample_rate}, but got {sr}"
        return clean, noisy, denoised, file
    

def _run_metrics(clean, estimate, noisy, args, sr):
    if estimate.shape[-1] < clean.shape[-1]:
        clean = clean[..., :estimate.shape[-1]]
        noisy = noisy[..., :estimate.shape[-1]]   
    sisdr = scale_invariant_signal_distortion_ratio(estimate, clean)
    sisdr_noisy = scale_invariant_signal_distortion_ratio(noisy, clean)
    sisdri = sisdr - sisdr_noisy
    return sisdri.mean().item(), sisdr.mean().item()

def _evaluate(clean_signals, noisy_signals, denoised_signals, filenames, data_dir, subdir, sample_rate, args):
    sisdri_all, sisdr_all, fnames = [], [], []
    for clean, noisy, denoised, filename in zip(clean_signals, noisy_signals, denoised_signals, filenames):
        run = True
        if args.filter_silent:
            peaks  = get_peaks(denoised.to('cpu').numpy().squeeze(), args.sample_rate)
            if len(peaks) == 0:
                run = False
        if run:
            sisdri_i, sisdr_i = _run_metrics(clean, denoised, noisy, args, sr=args.sample_rate)
            sisdri_all.append(sisdri_i)
            sisdr_all.append(sisdr_i)
            fnames.append(os.path.basename(filename))
    return sisdri_all, sisdr_all, fnames

def process(args):
    out_dir = args.data_dir
    assert os.path.exists(os.path.join(args.data_dir,'clean')), f"Directory {os.path.join(args.data_dir,'clean')} does not exist"
    subdirs = [name for name in os.listdir(os.path.join(args.data_dir,'denoised')) if os.path.isdir(os.path.join(args.data_dir,'denoised', name))]
    os.makedirs(os.path.join(args.data_dir, 'results'), exist_ok=True)
    for subdir in sorted(subdirs):
        print(f"Evaluating {subdir}")
        
        dset = EvaluationSet(args.data_dir, subdir, args.sample_rate)

        if dset is None:
            return
        loader = biodenoising.denoiser.distrib.loader(dset, batch_size=1)
        sisdri_all, sisdr_all, fns_all = [], [], []
        df = pd.DataFrame(columns=['sisdri','sisdr','filename'])
        with ProcessPoolExecutor(args.num_workers) as pool:
            iterator = biodenoising.denoiser.utils.LogProgress(logger, loader, name="Evaluating files")
            pendings = []
            for data in iterator:
                # Get batch data
                clean_signals, noisy_signals, denoised_signals, filenames = data
                clean_signals = torch.nan_to_num(clean_signals.to(args.device))
                noisy_signals = torch.nan_to_num(noisy_signals.to(args.device))
                denoised_signals = torch.nan_to_num(denoised_signals.to(args.device))
                if args.device == 'cpu' and args.num_workers > 1:
                    pendings.append(
                        pool.submit(_evaluate,
                                    clean_signals, noisy_signals, denoised_signals, filenames, out_dir, subdir, args.sample_rate, args))
                else:
                    sisdri, sisdr, fns = _evaluate(clean_signals, noisy_signals, denoised_signals, filenames, out_dir, subdir, args.sample_rate, args)
                    sisdri_all.extend(sisdri)
                    sisdr_all.extend(sisdr)
                    fns_all.extend(fns)
                    
            if pendings:
                print('Waiting for pending jobs...')
                for pending in biodenoising.denoiser.utils.LogProgress(logger, pendings, updates=5, name="Evaluating files"):
                    sisdri, sisdr, fns = pending.result()
                    sisdri_all.extend(sisdri)
                    sisdr_all.extend(sisdr)
                    fns_all.extend(fns)
        df['sisdri'] = sisdri_all
        df['sisdr'] = sisdr_all
        df['filename'] = fns_all
        df.to_csv(os.path.join(args.data_dir, 'results', subdir+'.csv'))   
        print("mean SDRi {} SDR {}".format(df['sisdri'].mean(), df['sisdr'].mean()))
        print("median SDRi {} SDR {}".format(df['sisdri'].median(), df['sisdr'].median()))           
        print(f"Done evaluating {subdir}")


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    process(args)
