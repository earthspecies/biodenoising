'''
python denoise.py --input /home/marius/data/biodenoising_validation/16000/noisy/ --output /home/marius/data/biodenoising_validation/16000/denoised/ 
'''
import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import os
import sys
import numpy as np

import torch
import torchaudio
import librosa
import norbert

from biodenoising.selection_table import (
    build_mask_from_events,
    find_selection_table_for,
    load_events_seconds,
)

from .demucs import DemucsStreamer
from .pretrained import add_model_flags, get_model
from .distrib import rank, loader, barrier
from .audio import find_audio_files, build_meta, Audioset
from .utils import LogProgress

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = set(['.wav','.mp3','.flac','.ogg','.aif','.aiff','.wmv','.WAV','.MP3','.FLAC','.OGG','.AIF','.AIFF','.WMV'])

def add_flags(parser):
    """
    Add the flags for the argument parser that are related to model loading and evaluation"
    """
    add_model_flags(parser)
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--dry', type=float, default=0,
                        help='dry/wet knob coefficient. 0 is only denoised, 1 only input signal.')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--streaming', action="store_true",
                        help="true streaming evaluation for Demucs")


parser = argparse.ArgumentParser(
        'denoise',
        description="Generate denoised files")
add_flags(parser)
parser.add_argument("--output", type=str, default="enhanced",
                    help="directory putting enhanced wav files")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="more loggging")
parser.add_argument("--method",choices=["demucs"], default="demucs",help="Method to use for denoising")
parser.add_argument("--transform",choices=["none", "time_scale"], default="none",help="Transform input by pitch shifting or time scaling")
parser.add_argument("--sample_rate",choices=[16000], default=16000,help="Sample rate of the model")
parser.add_argument("--keep_original_sr", action="store_true",help="keep the original sample rate of the audio rather than the model sample rate")
parser.add_argument('--selection_table', action="store_true", help="Enable event masking via selection tables (csv/tsv/txt) located next to audio files.")
parser.add_argument("--input", type=str, default=None,
                    help="path to the directory with noisy wav files")
parser.add_argument("--window_size", type=int, default=0,
                    help="size of the window for continuous processing")

def normalize(wav):
    return wav / max(wav.abs().max().item(), 1)

def get_estimate(model, noisy_signals, args):
    torch.set_num_threads(1)
    estimated_signals = torch.zeros_like(noisy_signals)
    for c in range(noisy_signals.shape[1]):
        noisy = noisy_signals[:,c:c+1,:]
        if args.method=='demucs' and args.streaming:
            streamer = DemucsStreamer(model, dry=args.dry)
            with torch.no_grad():
                estimate = torch.cat([
                    streamer.feed(noisy[0]),
                    streamer.flush()], dim=1)[None]
        else:
            with torch.no_grad():
                if hasattr(model, 'ola_forward'):
                    while noisy.ndim < 3:
                        noisy = noisy.unsqueeze(0)
                    estimate = model.forward(noisy)
                else:
                    estimate = model(noisy)
                estimate = (1 - args.dry) * estimate + args.dry * noisy
        estimated_signals[:,c:c+1,:] = estimate
    return estimated_signals

def time_scaling(signal, scaling):
    output_size = int(signal.shape[-1] * scaling)
    ref = torch.arange(output_size, device=signal.device, dtype=signal.dtype).div_(scaling)

    ref1 = ref.clone().type(torch.int64)
    ref2 = torch.min(ref1 + 1, torch.full_like(ref1, signal.shape[-1] - 1, dtype=torch.int64))
    r = ref - ref1.type(ref.type())
    scaled_signal = signal[..., ref1] * (1 - r) + signal[..., ref2] * r

    return scaled_signal


def save_wavs(estimates, noisy_sigs, filenames, out_dir, sr=16_000, write_noisy=False):
    # Write result
    if rank == 0:
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
    # wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def get_dataset(noisy_dir, sample_rate, channels, keep_original_sr):
    resample_to_sr = sample_rate if not keep_original_sr else None

    if args.noisy_dir:
        if os.path.isdir(noisy_dir):
            files = find_audio_files(noisy_dir)
        else:
            audio_files = [noisy_dir]
            files = build_meta(audio_files)
    else:
        logger.warning(
            "Small sample set was not provided by noisy_dir. "
            "Skipping denoising.")
        return None
    return Audioset(files, with_path=True,
                    sample_rate=sample_rate, channels=channels, convert=True, resample_to_sr=resample_to_sr)


def _estimate_and_save(model, noisy_signals, filenames, out_dir, sample_rate, data_sample_rate, args):
    save_sr = data_sample_rate if args.keep_original_sr else sample_rate
    ### Forward
    estimate = get_estimate(model, noisy_signals, args) 
    experiment = args.method 

    if args.transform == 'none':
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
        save_wavs(estimate, noisy_signals, filenames, out_dir, sr=save_sr)
    else:
        experiment += '_'+args.transform
        estimate_sum = estimate
        #noisy_signals = noisy_signals[None,None,:].float()
        for i in range(1,4):
            ### transform
            ### time scaling
            noisy_signals = time_scaling(noisy_signals, np.power(2, -0.5))
            # print("Scale to: {}".format(np.power(2, -0.5)))
            
            ### forward
            estimate = get_estimate(model, noisy_signals, args)
            
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
                                    
        estimate_out = estimate_sum/4.
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
        save_wavs(estimate_out, noisy_signals, filenames, out_dir, sr=save_sr)
                


def denoise(args, model=None, local_out_dir=None):
    # if args.device == 'cpu' and args.num_workers > 1:
    #     torch.multiprocessing.set_sharing_strategy('file_system')
    sample_rate = args.sample_rate
    channels = 1
    # Load model
    if args.method=='demucs':
        if not model:
            model = get_model(args).to(args.device)
            if args.sample_rate != model.sample_rate:
                logger.warning(f"Model sample rate is {model.sample_rate}, "
                            f"but the provided sample rate is {args.sample_rate}. "
                            f"Resampling will be performed.")
            sample_rate = model.sample_rate
            channels = model.chin
    else:
        sys.exit("Method not implemented")
    
    model.eval()
    
    if local_out_dir:
        out_dir = local_out_dir
    else:
        out_dir = args.out_dir
    
    dset = get_dataset(args.noisy_dir, sample_rate, channels, args.keep_original_sr)
    if dset is None:
        return
    dloader = loader(dset, batch_size=1, shuffle=False)
    
    if 'demucs' in args.method:
        barrier()

    with ProcessPoolExecutor(args.num_workers) as pool:
        iterator = LogProgress(logger, dloader, name="Denoising files")
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
            for pending in LogProgress(logger, pendings, updates=5, name="Denoising files"):
                pending.result()



if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    os.makedirs(args.output, exist_ok=True)
    if os.path.isdir(args.input):
        ### walk each subfolder recursively
        for root, dirs, files in os.walk(args.input):
            audio_files = [f for f in files if os.path.splitext(f)[1] in ALLOWED_EXTENSIONS]
            if len(audio_files) == 0:
                continue
            args.noisy_dir = root
            relative_path = os.path.relpath(root, args.input)
            args.out_dir = os.path.join(args.output, relative_path)
            os.makedirs(args.out_dir, exist_ok=True)
            denoise(args, local_out_dir=args.out_dir)
    elif os.path.splitext(args.input)[1] in ALLOWED_EXTENSIONS:
            args.noisy_dir = args.input
            os.makedirs(args.output, exist_ok=True)
            denoise(args, local_out_dir=args.output)
