import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import os
import sys
import glob
import random
import math

import scipy
import librosa
import numpy as np

import torch
import torchaudio
import noisereduce
from biodenoising.selection_table import (
    build_mask_from_events,
    find_selection_table_for,
    load_events_seconds,
)

import biodenoising

logger = logging.getLogger(__name__)


def add_flags(parser):
    """
    Add the flags for the argument parser that are related to model loading and evaluation"
    """
    biodenoising.denoiser.pretrained.add_model_flags(parser)
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--dry', type=float, default=0.1,
                        help='dry/wet knob coefficient. 0 is only denoised, 1 only input signal.')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--streaming', action="store_true",
                        help="true streaming evaluation for Demucs")


parser = argparse.ArgumentParser(
        'denoise',
        description="Generate denoised files")
add_flags(parser)
parser.add_argument("--out_dir", type=str, default="outputs",
                    help="directory putting enhanced wav files")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="more loggging")
parser.add_argument("--method",choices=["biodenoising16k_dns48","demucs", "cleanunet","demucsv4","noisereduce"], default="biodenoising16k_dns48",help="Method to use for denoising")
parser.add_argument("--transform",choices=["none", "time_scale"], default="none",help="Transform input by pitch shifting or time scaling")
parser.add_argument('--antialiasing', action="store_true",help="use an antialiasing filter when time scaling back")
parser.add_argument('--keep_original_sr', action="store_true",help="keep the original sample rate of the audio rather than the model sample rate")
parser.add_argument('--noise_reduce', action="store_true",help="use noisereduce preprocessing")
parser.add_argument('--selection_table', action="store_true", help="Enable event masking via selection tables (csv/tsv/txt) located next to audio files.")
parser.add_argument("--noisy_dir", type=str, default=None,
                    help="path to the directory with noisy wav files")
parser.add_argument("--window_size", type=int, default=0,
                    help="size of the window for continuous processing")


def normalize(wav):
    return wav / max(wav.abs().max().item(), 1)

def get_estimate(model, noisy, args):
    torch.set_num_threads(1)
    if args.method=='demucs' and args.streaming:
        streamer = biodenoising.denoiser.demucs.DemucsStreamer(model, dry=args.dry)
        with torch.no_grad():
            estimate = torch.cat([
                streamer.feed(noisy[0]),
                streamer.flush()], dim=1)[None]
    else:
        with torch.no_grad():
            if hasattr(model, 'ola_forward'):
                while noisy.ndim < 3:
                    noisy = noisy.unsqueeze(0)
                # if noisy.shape[-1] < model.window_size:
                #     noisy = torch.cat([noisy, torch.zeros((1,1,model.window_size - noisy.shape[-1])).to(args.device)], dim=-1)
                estimate = model.forward(noisy)
            else:
                estimate = model(noisy)
            estimate = (1 - args.dry) * estimate + args.dry * noisy
            #estimate = normalize(estimate)
    return estimate

def time_scaling(signal, scaling):
    output_size = int(signal.shape[-1] * scaling)
    ref = torch.arange(output_size, device=signal.device, dtype=signal.dtype).div_(scaling)

    ref1 = ref.clone().type(torch.int64)
    ref2 = torch.min(ref1 + 1, torch.full_like(ref1, signal.shape[-1] - 1, dtype=torch.int64))
    r = ref - ref1.type(ref.type())
    scaled_signal = signal[..., ref1] * (1 - r) + signal[..., ref2] * r

    return scaled_signal

def lowpass(wav, sample_rate, cutoff=20):
    [b,a] = scipy.signal.butter(4,cutoff, fs=sample_rate, btype='low')
    wav = scipy.signal.lfilter(b,a,wav)
    return wav

def save_wavs(estimates, noisy_sigs, filenames, out_dir, sr=16_000, write_noisy=False):
    # Write result
    os.makedirs(out_dir, exist_ok=True)
    for estimate, noisy, filename in zip(estimates, noisy_sigs, filenames):
        filename = os.path.join(out_dir, os.path.basename(filename).rsplit(".", 1)[0])
        if write_noisy:
            write(estimate, filename + "_enhanced.wav", sr=sr)
            write(noisy, filename +"_noisy.wav", sr=sr)
        else:
            write(estimate, filename + ".wav", sr=sr)        


def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def get_dataset(noisy_dir, sample_rate, channels, keep_original_sr):
    resample_to_sr = sample_rate if not keep_original_sr else None
    if noisy_dir:
        files = biodenoising.denoiser.audio.find_audio_files(noisy_dir)
    else:
        logger.warning(
            "Small sample set was not provided by noisy_dir. "
            "Skipping denoising.")
        return None
    return biodenoising.denoiser.audio.Audioset(files, with_path=True,
                    sample_rate=sample_rate, channels=channels, convert=True, resample_to_sr=resample_to_sr)


def _estimate_and_save(model, noisy_signals, filenames, out_dir, sample_rate, data_sample_rate, args):
    save_sr = data_sample_rate if args.keep_original_sr else sample_rate
    ### process
    if args.noise_reduce or args.method == 'noisereduce':
        noisy_signals = noisy_signals[0,0].to('cpu').numpy()  
        noisy_signals = noisereduce.reduce_noise(y=noisy_signals, sr=save_sr)
        noisy_signals = torch.from_numpy(noisy_signals[None,None,:]).to(args.device).float()
    if args.method == 'noisereduce':
        # Apply selection table mask if requested
        if args.selection_table:
            masked_signals = []
            for i, fn in enumerate(filenames):
                table = find_selection_table_for(fn)
                events = load_events_seconds(table)
                length_frames = noisy_signals.shape[-1]
                mask_1d = build_mask_from_events(length_frames, save_sr, events, noisy_signals.device)
                mask = mask_1d.view(1, 1, -1)
                masked_signals.append(noisy_signals[i:i+1] * mask)
            noisy_signals = torch.cat(masked_signals, dim=0) if masked_signals else noisy_signals
        save_wavs(noisy_signals, noisy_signals, filenames, os.path.join(out_dir,args.method), sr=save_sr)
    else:
        ### Forward
        estimate = get_estimate(model, noisy_signals, args)
        if not args.model_path and args.method == 'demucsv4':
            estimate = (estimate[:,1,...]+estimate[:,3,...]).sum(axis=1)[None,...]
        
        experiment = args.tag if args.model_path else args.method + '_pretrained'
        if args.noise_reduce:
            experiment += '_nr'
        if args.transform == 'none':
            if not args.model_path:
                experiment += '_none'
            # Apply selection table mask if requested
            if args.selection_table:
                masked_estimates = []
                for i, fn in enumerate(filenames):
                    table = find_selection_table_for(fn)
                    events = load_events_seconds(table)
                    length_frames = estimate.shape[-1]
                    mask_1d = build_mask_from_events(length_frames, save_sr, events, estimate.device)
                    mask = mask_1d.view(1, 1, -1)
                    masked_estimates.append(estimate[i:i+1] * mask)
                estimate = torch.cat(masked_estimates, dim=0) if masked_estimates else estimate
            save_wavs(estimate, noisy_signals, filenames, os.path.join(out_dir,experiment), sr=save_sr)
        else:
            estimate_sum = estimate
            #noisy_signals = noisy_signals[None,None,:].float()
            for i in range(1,4):
                ### transform
                ### time scaling
                noisy_signals = time_scaling(noisy_signals, np.power(2, -0.5))
                # print("Scale to: {}".format(np.power(2, -0.5)))
                
                ### forward
                estimate = get_estimate(model, noisy_signals, args)
                
                if args.antialiasing:
                    estimate = torch.from_numpy(lowpass(estimate.to('cpu').numpy(), sample_rate, cutoff=np.power(2, i*(-0.5))*sample_rate/2)).to(args.device).float()
                    
                ### transform back
                ### time scaling
                estimate_write = time_scaling(estimate, np.power(2, i*0.5))
                # print("Scale back: {}".format(np.power(2, i*0.5)))
                
                if estimate_sum.shape[-1] > estimate_write.shape[-1]:
                    estimate_sum[...,:estimate_write.shape[-1]] += estimate_write
                elif estimate_sum.shape[-1] < estimate_write.shape[-1]:
                    estimate_sum += estimate_write[...,:estimate_sum.shape[-1]]
                else:
                    estimate_sum += estimate_write
                    
                #save_wavs(estimate_write, noisy_signals, filenames, os.path.join(out_dir,args.method+'_'+args.transform + str(i)) , sr=sample_rate)
                        
            # Average aggregated estimate
            estimate_out = estimate_sum/4.
            # Apply selection table mask if requested (on final aggregated estimate)
            if args.selection_table:
                masked_estimates = []
                for i, fn in enumerate(filenames):
                    table = find_selection_table_for(fn)
                    events = load_events_seconds(table)
                    length_frames = estimate_out.shape[-1]
                    mask_1d = build_mask_from_events(length_frames, save_sr, events, estimate_out.device)
                    mask = mask_1d.view(1, 1, -1)
                    masked_estimates.append(estimate_out[i:i+1] * mask)
                estimate_out = torch.cat(masked_estimates, dim=0) if masked_estimates else estimate_out
            save_wavs(estimate_out, noisy_signals, filenames, os.path.join(out_dir,experiment+'_time_scale'), sr=save_sr)
                    


def denoise(args, model=None, local_out_dir=None):
    # if args.device == 'cpu' and args.num_workers > 1:
    #     torch.multiprocessing.set_sharing_strategy('file_system')
    sample_rate = 16000
    channels = 1
    # Load model
    if args.method=='demucs':
        if not model:
            # args.dns64=True
            model = biodenoising.denoiser.pretrained.get_model(args).to(args.device)
            sample_rate = model.sample_rate
            channels = model.chin
    if args.method=='biodenoising16k_dns48':
        if not model:
            args.biodenoising16k_dns48 = True
            model = biodenoising.denoiser.pretrained.get_model(args).to(args.device)
            sample_rate = model.sample_rate
            channels = model.chin
    elif args.method=='demucsv4':
        if not model:
            args.demucsv4 = True
            model = biodenoising.denoiser.pretrained.get_model(args).to(args.device)
            sample_rate = model.samplerate
            channels = model.audio_channels
            model.use_train_segment=False
    elif args.method=='cleanunet':
        if not model:
            args.cleanunet_speech = True
            model = biodenoising.denoiser.pretrained.get_model(args).to(args.device)
            sample_rate = 16000
            channels = 1
    
    if args.model_path:
        if 'dset=' in args.model_path:
            args.tag = os.path.basename(os.path.dirname(args.model_path)).split('dset=')[1].replace('biodenoising16k_', '').replace('biodenoising48k_', '')
        else:
            args.tag = os.path.basename(args.model_path).replace('.th', '')
    else:
        args.tag = None

    if args.method != 'noisereduce': 
        model.eval()
    if local_out_dir:
        out_dir = local_out_dir
    else:
        out_dir = args.out_dir
    
    dset = get_dataset(args.noisy_dir, sample_rate, channels, args.keep_original_sr)
    if dset is None:
        return
    loader = biodenoising.denoiser.distrib.loader(dset, batch_size=1, shuffle=False)
    
    if 'demucs' in args.method:
        biodenoising.denoiser.distrib.barrier()

    with ProcessPoolExecutor(args.num_workers) as pool:
        iterator = biodenoising.denoiser.utils.LogProgress(logger, loader, name="Denoising files")
        pendings = []
        for data in iterator:
            # Get batch data
            noisy_signals, filenames, data_sample_rate = data
            noisy_signals = noisy_signals.to(args.device)
            if args.device == 'cpu' and args.num_workers > 1:
                pendings.append(
                    pool.submit(_estimate_and_save,
                                model, noisy_signals, filenames, out_dir, sample_rate, data_sample_rate, args))
            else:
                if args.window_size > 0:
                    import asteroid
                    ola_model = asteroid.dsp.overlap_add.LambdaOverlapAdd(
                        nnet=model,  # function to apply to each segment.
                        n_src=1,  # number of sources in the output of nnet
                        window_size=args.window_size,  # Size of segmenting window
                        hop_size=args.window_size//4,  # segmentation hop size
                        window="hann",  # Type of the window (see scipy.signal.get_window
                        reorder_chunks=False,  # Whether to reorder each consecutive segment.
                        enable_grad=False,  # Set gradient calculation on of off (see torch.set_grad_enabled)
                    )
                    ola_model.window = ola_model.window.to(args.device)
                    _estimate_and_save(ola_model, noisy_signals, filenames, out_dir, sample_rate, data_sample_rate, args)
                else:
                    _estimate_and_save(model, noisy_signals, filenames, out_dir, sample_rate, data_sample_rate, args)

        if pendings:
            print('Waiting for pending jobs...')
            for pending in biodenoising.denoiser.utils.LogProgress(logger, pendings, updates=5, name="Denoising files"):
                pending.result()



def main() -> None:
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    denoise(args, local_out_dir=args.out_dir)


if __name__ == "__main__":
    main()
