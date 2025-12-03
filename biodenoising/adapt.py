from concurrent.futures import ProcessPoolExecutor
import json
import logging
import os
import scipy
import random
import numpy as np
import pandas as pd
import librosa
import torch
import torchaudio
import noisereduce
from . import denoiser
from . import datasets
from pathlib import Path
import argparse
import sys
import yaml
from .selection_table import (
    build_mask_from_events,
    find_selection_table_for,
    load_events_seconds,
)

logger = logging.getLogger(__name__)

class ConfigParser:
    def __init__(self, *pargs, **kwpargs):
        self.options = []
        self.pargs = pargs
        self.kwpargs = kwpargs
        self.conf_parser = argparse.ArgumentParser(add_help=False)
        self.conf_parser.add_argument("-c", "--config",
                                 default="biodenoising/conf/config_adapt.yaml",
                                 help="where to load YAML configuration")
        
    def add_argument(self, *args, **kwargs):
        self.options.append((args, kwargs))

    def parse(self, args=None):
        if args is None:
            args = sys.argv[1:]

        res, remaining_argv = self.conf_parser.parse_known_args(args)

        config_vars = {}
        if res.config is not None:
            with open(res.config, 'rb') as stream:
                config_vars = yaml.safe_load(stream.read())

        parser = argparse.ArgumentParser(
            *self.pargs,
            # Inherit options from config_parser
            parents=[self.conf_parser],
            # Don't mess with format of description
            formatter_class=argparse.RawDescriptionHelpFormatter,
            **self.kwpargs,
        )

        for opt_args, opt_kwargs in self.options:
            parser_arg = parser.add_argument(*opt_args, **opt_kwargs)
            if parser_arg.dest in config_vars:
                config_default = config_vars.pop(parser_arg.dest)
                expected_type = str
                if parser_arg.type is not None:
                    expected_type = parser_arg.type
                # Handle store_true/store_false actions (boolean flags)
                elif opt_kwargs.get('action') in ['store_true', 'store_false']:
                    expected_type = bool
                
                # Perform type check
                if not isinstance(config_default, expected_type):
                    parser.error('YAML configuration entry {} '
                                 'does not have type {}'.format(
                                     parser_arg.dest,
                                     expected_type))

                parser_arg.default = config_default
        
        
        for k,v in config_vars.items():
            # Only set defaults for keys that aren't already added as arguments
            # This prevents overriding command line arguments with config values
            if k not in [opt_kwargs.get('dest') or opt_args[0].lstrip('-') for opt_args, opt_kwargs in self.options]:
                parser.set_defaults(**{k:v})
                # Special handling for nested dset dictionary
                if k=='dset': 
                    for k1,v1 in v.items():
                        parser.set_defaults(**{k1:v1})
            
        return parser.parse_args(remaining_argv)

def get_estimate(model, noisy, args):
    torch.set_num_threads(1)
    if args.method=='demucs' and args.streaming:
        streamer = denoiser.demucs.DemucsStreamer(model, dry=args.dry)
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

def _ensure_int_sr(value):
    """Return a Python int sampling rate from various scalar-like types.

    Accepts Python numbers, NumPy scalars/arrays (0-D/1-D), Torch tensors
    (0-D/1-D), and simple lists/tuples. Falls back to int(value).
    """
    try:
        import numpy as _np  # local import to avoid polluting namespace
    except Exception:  # pragma: no cover
        _np = None
    try:
        import torch as _torch  # local import
    except Exception:  # pragma: no cover
        _torch = None

    # Torch tensor handling
    if _torch is not None and isinstance(value, _torch.Tensor):
        if value.ndim == 0:
            return int(value.item())
        if value.numel() == 1:
            return int(value.reshape(()).item())
        return int(value.flatten()[0].item())

    # NumPy handling
    if _np is not None:
        if isinstance(value, _np.ndarray):
            if value.ndim == 0:
                return int(value.item())
            if value.size >= 1:
                return int(value.flatten()[0].item())
        if isinstance(value, _np.generic):  # NumPy scalar
            return int(value.item())

    # List/tuple
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return int(value[0])

    # Fallback
    return int(value)

def highpass(wav, sample_rate, cutoff=20):
    [b,a] = scipy.signal.butter(4,cutoff, fs=sample_rate, btype='high')
    wav = scipy.signal.lfilter(b,a,wav)
    return wav

def save_wavs(estimates, filenames, out_dir, version='', sr=16_000):
    os.makedirs(out_dir, exist_ok=True)
    # Write result
    allf = []
    for estimate, filename in zip(estimates, filenames):
        filename = os.path.join(out_dir, os.path.basename(filename).rsplit(".", 1)[0])
        write(estimate, filename + str(version) + ".wav", sr=sr)
        allf.append(filename + str(version) + ".wav")
    return allf        


def write(wav, filename, sr=16_000):
    torchaudio.save(filename, wav.cpu(), sr)

def enhance_noise(noise, estimate):
    squeeze=False
    if noise.ndim == 3 and estimate.ndim == 3:
        noise = noise.squeeze(0)
        estimate = estimate.squeeze(0)
        squeeze=True
    ### compute magnitude spectrogram
    window = torch.hann_window(1024).to(noise.device)
    stft_audio = torch.stft(estimate, n_fft=1024, hop_length=512, win_length=1024, window=window, center=True, return_complex=True)
    stft_noise = torch.stft(noise, n_fft=1024, hop_length=512, win_length=1024, window=window, center=True, return_complex=True)
    pspec_audio = torch.abs(stft_audio)**2
    pspec_noise = torch.abs(stft_noise)**2
    mask = pspec_audio < pspec_noise
    #### inverse transform
    phase_noise = torch.angle(stft_noise)
    pspec_noise = torch.abs(stft_noise) * mask
    #### build a complex spectrogram
    stft_noise = pspec_noise * torch.exp(1j * phase_noise)
    noise = torch.istft(stft_noise, n_fft=1024, hop_length=512, win_length=1024, window=window, center=True)
    if squeeze:
        noise = noise.unsqueeze(0)
    return noise

def get_dataset(noisy_dir, sample_rate, channels, args=None):
    if noisy_dir:
        files = denoiser.audio.find_audio_files(noisy_dir)
    else:
        logger.warning(
            "Small sample set was not provided by noisy_dir. "
            "Skipping denoising.")
        return None
    
    # Set resample_to_sr based on keep_original_sr (mirror denoise.py)
    resample_to_sr = None
    if args is not None:
        keep_original = getattr(args, 'keep_original_sr', False)
        resample_to_sr = None if keep_original else sample_rate

    return denoiser.audio.Audioset(files, with_path=True,
                    sample_rate=sample_rate, channels=channels, 
                    convert=True, resample_to_sr=resample_to_sr)


def _compute_norms_from_events(signal: torch.Tensor, events, sample_rate: int):
    """
    Compute RMS norms for each (start, end) in seconds using the provided signal.
    Returns (norms_list, start_end_array)
    """
    if signal.ndim == 3:
        signal = signal.squeeze(0).squeeze(0)
    length_frames = signal.shape[-1]
    duration_s = float(length_frames) / float(sample_rate) if sample_rate > 0 else 0.0
    norms = []
    start_end = []
    # Normalize and clip events to [0, duration]
    cleaned = []
    for start_s, end_s in events:
        s = float(max(0.0, min(duration_s, start_s)))
        e = float(max(0.0, min(duration_s, end_s)))
        if e > s:
            cleaned.append((s, e))
    # Sort and merge overlapping/adjacent intervals (optional but safer)
    cleaned.sort(key=lambda x: x[0])
    merged = []
    for s, e in cleaned:
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    # Compute RMS per interval; fallback to full-length if empty
    if not merged and duration_s > 0:
        merged = [[0.0, duration_s]]
    for s, e in merged:
        start_idx = int(round(s * sample_rate))
        end_idx = int(round(e * sample_rate))
        start_idx = max(0, min(length_frames, start_idx))
        end_idx = max(0, min(length_frames, end_idx))
        if end_idx <= start_idx:
            continue
        seg = signal[start_idx:end_idx]
        if seg.numel() == 0:
            continue
        rms = torch.sqrt(torch.mean(seg.float() ** 2)).item()
        norms.append(rms)
        start_end.append((s, e))
    import numpy as _np
    return _np.asarray(norms, dtype=float), _np.asarray(start_end, dtype=float)


def _estimate_and_save(model, noisy_signals, filenames, out_dir, step, sample_rate, data_sample_rate, args):
    save_sr = _ensure_int_sr(data_sample_rate) if args.keep_original_sr else int(sample_rate)
    ### process
    if args.noise_reduce:
        noisy_signals = noisy_signals[0,0].to('cpu').numpy()  
        noisy_signals = noisereduce.reduce_noise(y=noisy_signals, sr=save_sr)
        noisy_signals = torch.from_numpy(noisy_signals[None,None,:]).to(args.device).float()
    
    ### Forward
    estimate = get_estimate(model, noisy_signals, args)

    if args.transform == 'none':
        # Apply selection table mask if requested
        if getattr(args, 'selection_table', False):
            masked_estimates = []
            for i, fn in enumerate(filenames):
                table = find_selection_table_for(fn)
                events = load_events_seconds(table)
                length_frames = estimate.shape[-1]
                mask_1d = build_mask_from_events(length_frames, save_sr, events, estimate.device)
                mask = mask_1d.view(1, 1, -1)
                masked_estimates.append(estimate[i:i+1] * mask)
            estimate = torch.cat(masked_estimates, dim=0) if masked_estimates else estimate
        save_wavs(estimate, filenames, os.path.join(out_dir,args.experiment), sr=save_sr)
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
                estimate = torch.from_numpy(lowpass(estimate.to('cpu').numpy(), save_sr, cutoff=np.power(2, i*(-0.5))*save_sr/2)).to(args.device).float()
                
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
                
            #save_wavs(estimate_write, filenames, os.path.join(out_dir,args.method+'_'+args.transform + str(i)) , sr=sample_rate)
        
        # Average aggregated estimate
        estimate_out = estimate_sum/4.
        # Apply selection table mask if requested (on final aggregated estimate)
        if getattr(args, 'selection_table', False):
            masked_estimates = []
            for i, fn in enumerate(filenames):
                table = find_selection_table_for(fn)
                events = load_events_seconds(table)
                length_frames = estimate_out.shape[-1]
                mask_1d = build_mask_from_events(length_frames, save_sr, events, estimate_out.device)
                mask = mask_1d.view(1, 1, -1)
                masked_estimates.append(estimate_out[i:i+1] * mask)
            estimate_out = torch.cat(masked_estimates, dim=0) if masked_estimates else estimate_out
        save_wavs(estimate_out, filenames, os.path.join(out_dir,args.experiment), sr=save_sr)
        
    return [],[]        

def _estimate_and_save_chunks(model, noisy_signals, filenames, out_subdir, step, sample_rate, data_sample_rate, args):
    save_sr = _ensure_int_sr(data_sample_rate) if args.keep_original_sr else int(sample_rate)
    original_noisy_signals = noisy_signals.clone()
    revecho=denoiser.augment.RevEcho(0.99)
    max_value = noisy_signals.abs().max()
    noisy_signals = noisy_signals[0,0].to('cpu').numpy()  
    if args.noise_reduce:
        noisy_signals = noisereduce.reduce_noise(y=noisy_signals, sr=save_sr)

    ### remove dc component
    noisy_signals = highpass(noisy_signals, save_sr, cutoff=args.highpass)
    noisy_signals = torch.from_numpy(noisy_signals[None,None,:]).to(args.device).float()
    
    if args.time_scale_factor != 0:
        noisy_signals_fwd = noisy_signals
        if args.antialiasing and args.time_scale_factor>0:
            ## anti-aliasing
            noisy_signals_fwd = torch.from_numpy(lowpass(noisy_signals.to('cpu').numpy(), save_sr, cutoff=save_sr//(args.time_scale_factor*4))).to(args.device).float()
        noisy_signals_fwd = time_scaling(noisy_signals_fwd, np.power(2, args.time_scale_factor*0.5))
    else:
        noisy_signals_fwd = noisy_signals
        
    ### Forward
    estimate = get_estimate(model, noisy_signals_fwd, args)
    
    if args.time_scale_factor != 0:
        if args.antialiasing and args.time_scale_factor>0:
            ## anti-aliasing
            estimate = torch.from_numpy(lowpass(estimate.to('cpu').numpy(), save_sr, cutoff=save_sr//(args.time_scale_factor*4))).to(args.device).float()
        estimate = time_scaling(estimate, np.power(2, -args.time_scale_factor*0.5))
        ### remove low frequency artifacts
        estimate = torch.from_numpy(highpass(estimate.to('cpu').numpy(), save_sr)).to(args.device).float()
    
    csv_path = os.path.join(out_subdir,args.experiment+'_detection')
    os.makedirs(csv_path, exist_ok=True)
    full_estimate_noise = noisy_signals - estimate
    if args.transform == 'none':
        if getattr(args, 'selection_table', False):
            # Use selection tables to derive start_end and norms
            events = load_events_seconds(find_selection_table_for(filenames[0]))

            ### mask the estimate
            length_frames = estimate.shape[-1]
            mask_1d = build_mask_from_events(length_frames, save_sr, events, estimate.device)
            mask = mask_1d.view(1, 1, -1)
            estimate = estimate * mask

            # Normalize events right after loading (N, 2) float array
            events = np.asarray(events, dtype=float).reshape(-1, 2)

            norms, start_end = _compute_norms_from_events(estimate.to('cpu'), events, save_sr)
            res = []
            res_noise = []
            if len(start_end) > 0:
                ### save csv
                filename = os.path.join(csv_path, os.path.basename(filenames[0]).rsplit(".", 1)[0]) + '.csv'
                with open(filename, 'w') as f:
                    f.write('start,end,rms\n')
                    for i in range(len(start_end)):
                        f.write(str(start_end[i][0]) + ',' + str(start_end[i][1]) + ',' + str(norms[i]) + '\n')
                ### save wav
                noise = original_noisy_signals-estimate
                if args.revecho > 0:
                    estimate = revecho(torch.stack((estimate,estimate)))[0]
                if args.amp_scale:
                    estimate = estimate * max_value / estimate.abs().max()
                estimate_chunks, estimate_noise_chunks = get_chunks(estimate, noise, save_sr, start_end, duration=args.segment, compute_noise=True)
                for j, chunk in enumerate(estimate_chunks):
                    allf = save_wavs(chunk, filenames, os.path.join(out_subdir,args.experiment), '_'+str(j), sr=save_sr)
                    res.append([allf[0],norms[j], os.path.join(out_subdir,args.experiment)])
                if args.noise_dir is None:
                    if estimate_noise_chunks is not None and len(estimate_noise_chunks) > 0:
                        for j, chunk in enumerate(estimate_noise_chunks):
                            allfnoise = save_wavs(chunk, filenames, os.path.join(out_subdir,args.experiment+'_noise'), '_'+str(j), sr=save_sr)
                            res_noise.append([allfnoise[0],1-np.array(norms).max(), os.path.join(out_subdir,args.experiment+'_noise')])
                    else:
                        # Extract noise between events with triangular windowing
                        stitched_noise = extract_noise_between_events(noise, start_end, save_sr, window_type='triangular')
                        allfnoise = save_wavs(stitched_noise, filenames, os.path.join(out_subdir,args.experiment+'_noise'), sr=save_sr)
                        res_noise.append([allfnoise[0],1., os.path.join(out_subdir,args.experiment+'_noise')])
                return res, res_noise  
            else:
                ### everything is noise 
                allfnoise = save_wavs(original_noisy_signals, filenames, os.path.join(out_subdir,args.experiment+'_noise'), sr=save_sr)
                res_noise.append([allfnoise[0],1., os.path.join(out_subdir,args.experiment+'_noise')])
                return res, res_noise
        else:
            norms, start_end = get_start_end(estimate.to('cpu').numpy().squeeze(), save_sr)
            res = []
            res_noise = []
            if len(start_end) > 0:
                ### save csv
                filename = os.path.join(csv_path, os.path.basename(filenames[0]).rsplit(".", 1)[0]) + '.csv'
                with open(filename, 'w') as f:
                    f.write('start,end,rms\n')
                    for i in range(len(start_end)):
                        f.write(str(start_end[i][0]) + ',' + str(start_end[i][1]) + ',' + str(norms[i]) + '\n')
        
                ### save wav
                noise = original_noisy_signals-estimate
                if args.noise_dir is None:
                    for i in range(3):
                        signal = get_estimate(model, noise, args)
                        noise = noise - signal
                    noise = 3 * noise  
                    noise = noise/noise.abs().max() if noise.abs().max() > 1 else noise
                    if args.noisy_estimate:
                        ### trim the first 0.5 and the last 0.5 of the noise
                        if noise.shape[-1] > save_sr * 2:
                            trim_samples = int(save_sr * 1)
                            noise = noise[..., trim_samples:-trim_samples]                        
                        noise = enhance_noise(noise, estimate)  
                        allfnoise = save_wavs(noise, filenames, os.path.join(out_subdir,args.experiment+'_noise'), sr=save_sr)
                        res_noise.append([allfnoise[0],1., out_subdir])
                if args.revecho > 0:
                    estimate = revecho(torch.stack((estimate,estimate)))[0]
                if args.amp_scale:
                    estimate = estimate * max_value / estimate.abs().max()
                estimate_chunks, estimate_noise_chunks = get_chunks(estimate, noise, save_sr, start_end, duration=args.segment, compute_noise=True)
                for j, chunk in enumerate(estimate_chunks):
                    allf = save_wavs(chunk, filenames, os.path.join(out_subdir,args.experiment), '_'+str(j), sr=save_sr)
                    res.append([allf[0],norms[j], os.path.join(out_subdir,args.experiment)])
                if estimate_noise_chunks is not None and args.noise_dir is None:
                    for j, chunk in enumerate(estimate_noise_chunks):
                        allfnoise = save_wavs(chunk, filenames, os.path.join(out_subdir,args.experiment+'_noise'), '_'+str(j), sr=save_sr)
                        res_noise.append([allfnoise[0],1-np.array(norms).max(), os.path.join(out_subdir,args.experiment+'_noise')])
                return res, res_noise    
            else:
                ### save csv
                filename = os.path.join(csv_path, os.path.basename(filenames[0]).rsplit(".", 1)[0]) + '.csv'
                with open(filename, 'w') as f:
                    f.write('start,end,rms\n')
                if args.noise_dir is None:
                    noise = original_noisy_signals-estimate
                    for i in range(3):
                        signal = get_estimate(model, noise, args)
                        noise = noise - signal
                    noise = 3 * noise  
                    noise = noise/noise.abs().max() if noise.abs().max() > 1 else noise
                    allfnoise = save_wavs(noise, filenames, os.path.join(out_subdir,args.experiment+'_noise'), sr=save_sr)
                    res_noise.append([allfnoise[0],1., os.path.join(out_subdir,args.experiment+'_noise')])
                return res,res_noise

    else:
        ### we sum all the results here
        estimate_sum = estimate
        # estimates = []
        for i in range(1,4): ### animal sounds sit usually in higher frequencies; shift them down
            noisy_signals = torch.from_numpy(highpass(noisy_signals.to('cpu').numpy().squeeze(), save_sr)).to(args.device)
            noisy_signals = noisy_signals[None,None,:].float()
            ### transform
            ### time scaling
            noisy_signals = time_scaling(noisy_signals, np.power(2, -0.5))
            
            ### forward
            estimate = get_estimate(model, noisy_signals, args)
                        
            ## anti-aliasing
            if i>0 and args.antialiasing:
                estimate = torch.from_numpy(lowpass(estimate.to('cpu').numpy(), save_sr, cutoff=save_sr//(i*4))).to(args.device).float()
            
            ### transform back
            ### time scaling
            estimate_write = time_scaling(estimate, np.power(2, i*0.5))
            # estimates.append(estimate_write)

            if estimate_sum.shape[-1] > estimate_write.shape[-1]:
                estimate_sum[...,:estimate_write.shape[-1]] += estimate_write
            elif estimate_sum.shape[-1] < estimate_write.shape[-1]:
                estimate_sum += estimate_write[...,:estimate_sum.shape[-1]]
            else:
                estimate_sum += estimate_write
        
        res = []
        res_noise = []
        if getattr(args, 'selection_table', False):
            events = load_events_seconds(find_selection_table_for(filenames[0]))
            
            ### mask the estimate
            length_frames = estimate_sum.shape[-1]
            mask_1d = build_mask_from_events(length_frames, save_sr, events, estimate_sum.device)
            mask = mask_1d.view(1, 1, -1)
            estimate_sum = estimate_sum * mask

            # Normalize events right after loading (N, 2) float array
            events = np.asarray(events, dtype=float).reshape(-1, 2)

            res = []
            res_noise = []
            norms, start_end = _compute_norms_from_events(estimate_sum.to('cpu'), events, save_sr)
            if len(start_end) > 0:
                ### save csv
                filename = os.path.join(csv_path, os.path.basename(filenames[0]).rsplit(".", 1)[0]) + '.csv'
                with open(filename, 'w') as f:
                    f.write('start,end,rms\n')
                    for i in range(len(start_end)):
                        f.write(str(start_end[i][0]) + ',' + str(start_end[i][1]) + ',' + str(norms[i]) + '\n')
                ### save wav
                noise = original_noisy_signals-estimate_sum
                if args.revecho > 0:
                    estimate_sum = revecho(torch.stack((estimate_sum,estimate_sum)))[0]
                if args.amp_scale:
                    estimate_sum = estimate_sum * max_value / estimate_sum.abs().max()
                estimate_chunks, estimate_noise_chunks = get_chunks(estimate_sum, noise/4., save_sr, start_end, duration=args.segment, compute_noise=True)
                for j, chunk in enumerate(estimate_chunks):
                    allf = save_wavs(chunk/4., filenames, os.path.join(out_subdir,args.experiment), '_'+str(j), sr=save_sr)
                    res.append([allf[0],norms[j], os.path.join(out_subdir,args.experiment)])
                if args.noise_dir is None:
                    if estimate_noise_chunks is not None and len(estimate_noise_chunks) > 0:
                        for j, chunk in enumerate(estimate_noise_chunks):
                            allfnoise = save_wavs(chunk/4., filenames, os.path.join(out_subdir,args.experiment+'_noise'), '_'+str(j), sr=save_sr)
                            res_noise.append([allfnoise[0],1-np.array(norms).max(), os.path.join(out_subdir,args.experiment+'_noise')])
                    else:
                        # Extract noise between events with triangular windowing
                        stitched_noise = extract_noise_between_events(noise, start_end, save_sr, window_type='triangular')
                        allfnoise = save_wavs(stitched_noise/4., filenames, os.path.join(out_subdir,args.experiment+'_noise'), sr=save_sr)
                        res_noise.append([allfnoise[0],1., os.path.join(out_subdir,args.experiment+'_noise')])
                return res, res_noise    
            else:
                ### everything is noise 
                allfnoise = save_wavs(original_noisy_signals, filenames, os.path.join(out_subdir,args.experiment+'_noise'), sr=save_sr)
                res_noise.append([allfnoise[0],1., os.path.join(out_subdir,args.experiment+'_noise')])
                return res, res_noise
        else:
            norms, start_end = get_start_end(estimate_sum.to('cpu').numpy().squeeze(), save_sr)
            if len(start_end) > 0:
                ### replace nans with minimum value
                norms[np.isnan(norms)] = norms.min()
                ### save csv
                filename = os.path.join(csv_path, os.path.basename(filenames[0]).rsplit(".", 1)[0]) + '.csv'
                with open(filename, 'w') as f:
                    f.write('start,end,rms\n')
                    for i in range(len(start_end)):
                        f.write(str(start_end[i][0]) + ',' + str(start_end[i][1]) + ',' + str(norms[i]) + '\n')
        
                ### save wav
                noise = original_noisy_signals-estimate
                if args.noise_dir is None:
                    for i in range(3):
                        signal = get_estimate(model, noise, args)
                        noise = noise - signal
                    noise = 3 * noise  
                    noise = noise/noise.abs().max() if noise.abs().max() > 1 else noise
                    if args.noisy_estimate:
                        ### trim the first 0.5 and the last 0.5 of the noise
                        if noise.shape[-1] > save_sr * 2:
                            trim_samples = int(save_sr * 1)
                            noise = noise[..., trim_samples:-trim_samples]                        
                        noise = enhance_noise(noise, estimate)  
                        allfnoise = save_wavs(noise, filenames, os.path.join(out_subdir,args.experiment+'_noise'), sr=save_sr)
                        res_noise.append([allfnoise[0],1., out_subdir])
                if args.revecho > 0:
                    estimate_sum = revecho(torch.stack((estimate_sum,estimate_sum)))[0]
                if args.amp_scale:
                    estimate_sum = estimate_sum * max_value / estimate_sum.abs().max()
                estimate_sum_chunks, estimate_noise_chunks = get_chunks(estimate_sum, noise/4., save_sr, start_end, duration=args.segment, compute_noise=True)
                for j, chunk in enumerate(estimate_sum_chunks):
                    allf = save_wavs(chunk/4., filenames, os.path.join(out_subdir,args.experiment), '_'+str(j), sr=save_sr)
                    res.append([allf[0],norms[j], os.path.join(out_subdir,args.experiment)])
                if estimate_noise_chunks is not None and args.noise_dir is None:
                    for j, chunk in enumerate(estimate_noise_chunks):
                        allfnoise = save_wavs(chunk/4., filenames, os.path.join(out_subdir,args.experiment+'_noise'), '_'+str(j), sr=save_sr)
                        res_noise.append([allfnoise[0],1-np.array(norms).max(), os.path.join(out_subdir,args.experiment+'_noise')])
                return res, res_noise    
            else:
                ### save csv
                filename = os.path.join(csv_path, os.path.basename(filenames[0]).rsplit(".", 1)[0]) + '.csv'
                with open(filename, 'w') as f:
                    f.write('start,end,rms\n')
                if args.noise_dir is None:
                    noise = original_noisy_signals-estimate
                    for i in range(3):
                        signal = get_estimate(model, noise, args)
                        noise = noise - signal
                    noise = 3 * noise  
                    noise = noise/noise.abs().max() if noise.abs().max() > 1 else noise
                    allfnoise = save_wavs(noise, filenames, os.path.join(out_subdir,args.experiment+'_noise'), sr=save_sr)
                    res_noise.append([allfnoise[0],1., out_subdir])
                return res,res_noise

def get_experiment_code(args,step):
    experiment = args.method + '_pretrained' if step==0 else args.method + '_step'+str(step)
    if args.noise_reduce:
        experiment += '_nr'
    if args.transform == 'none':
        experiment += '_none'
    elif args.transform == 'time_scale':
        experiment += '_time_scale'
    experiment += '_step'+str(step)
    return experiment
    
def denoise(args, step=0):
    args.experiment = get_experiment_code(args,step)
    # if args.device == 'cpu' and args.num_workers > 1:
    #     torch.multiprocessing.set_sharing_strategy('file_system')
    sample_rate = args.sample_rate
    channels = 1
    #### Load model
    if (len(args.model_path)==0 or not os.path.exists(args.model_path)) and args.method=='biodenoising16k_dns48':
        args.biodenoising16k_dns48 = True
    model = denoiser.pretrained.get_model(args).to(args.device)
    sample_rate = model.sample_rate
    channels = model.chin
    args.length = args.segment * sample_rate
    if os.path.exists(args.model_path):
        args.tag = args.method + '_step' + str(step)
    else:
        args.tag = None

    out_dir = args.out_dir
    
    dset = get_dataset(os.path.join(args.noisy_dir), sample_rate, channels, args)
    if dset is None:
        return
    loader = denoiser.distrib.loader(dset, batch_size=1, shuffle=False)
    
    denoiser.distrib.barrier()

    md = pd.DataFrame(columns=['fn','metric','dataset'])
    md_noise = pd.DataFrame(columns=['fn','metric','dataset'])
    npos = 0
    nneg = 0
    with ProcessPoolExecutor(np.maximum(1,args.num_workers)) as pool:
        iterator = denoiser.utils.LogProgress(logger, loader, name="Denoising files")
        pendings = []
        for data in iterator:
            # Get batch data
            noisy_signals, filenames, data_sample_rate = data
            noisy_signals = noisy_signals.to(args.device)
            if args.device == 'cpu' and args.num_workers > 1:
                if step<args.steps:
                    pendings.append(
                        pool.submit(_estimate_and_save_chunks,
                                    model, noisy_signals, filenames, out_dir, step, sample_rate, data_sample_rate, args))
                else:
                    pendings.append(
                        pool.submit(_estimate_and_save,
                                    model, noisy_signals, filenames, out_dir, step, sample_rate, data_sample_rate, args))
            else:
                res_noise = None
                if args.window_size > 0:
                    import asteroid
                    window_size_samples = int(args.window_size * _ensure_int_sr(data_sample_rate))
                    hop_size_samples = int(window_size_samples//4)
                    ola_model = asteroid.dsp.overlap_add.LambdaOverlapAdd(
                        nnet=model,  # function to apply to each segment.
                        n_src=1,  # number of sources in the output of nnet
                        window_size=window_size_samples,  # Size of segmenting window
                        hop_size=hop_size_samples,  # segmentation hop size
                        window="hann",  # Type of the window (see scipy.signal.get_window
                        reorder_chunks=False,  # Whether to reorder each consecutive segment.
                        enable_grad=False,  # Set gradient calculation on of off (see torch.set_grad_enabled)
                    )
                    ola_model.window = ola_model.window.to(args.device)
                    if step<args.steps:
                        res, res_noise = _estimate_and_save_chunks(ola_model, noisy_signals, filenames, out_dir, step, sample_rate, data_sample_rate, args)
                    else:
                        res, res_noise = _estimate_and_save(ola_model, noisy_signals, filenames, out_dir, step, sample_rate, data_sample_rate, args)
                else:
                    if step<args.steps:
                        res, res_noise = _estimate_and_save_chunks(model, noisy_signals, filenames, out_dir, step, sample_rate, data_sample_rate, args)
                    else:
                        res, res_noise = _estimate_and_save(model, noisy_signals, filenames, out_dir, step, sample_rate, data_sample_rate, args)
                if res_noise is not None and len(res)>0:
                    npos += 1
                    for r in res:
                        md.loc[len(md)] = r
                else:
                    nneg += 1
                if res_noise is not None and len(res_noise)>0:
                    for r in res_noise:
                        md_noise.loc[len(md_noise)] = r
        if pendings:
            print('Waiting for pending jobs...')
            res_noise = None
            for pending in denoiser.utils.LogProgress(logger, pendings, updates=5, name="Denoising files"):
                res, res_noise = pending.result()
                if len(res)>0:
                    npos += 1
                    for r in res:
                        md.loc[len(md)] = r
                else:
                    nneg += 1
                if res_noise is not None and len(res_noise)>0:
                    for r in res_noise:
                        md_noise.loc[len(md_noise)] = r
    if step<args.steps:
        print("denoised with calls %d files, without calls %d files." % (npos, nneg))
        md.to_csv(os.path.join( out_dir, args.experiment+".csv"), index=False)
        md_noise.to_csv(os.path.join( out_dir, args.experiment+"_noise.csv"), index=False)
    return model

def get_start_end(wav, sample_rate, smoothing_window=3, db_treshold=-40, min_duration=0.2):
    wav = noisereduce.reduce_noise(y=wav, sr=sample_rate)
    wav = highpass(wav, sample_rate, cutoff=60)
    window_size = int(0.1 * sample_rate)
    hop_size = int(0.02 * sample_rate)
    short_audio = False
    if wav.shape[-1] < 2 * window_size:
        ### adjust hop and window size
        window_size = int(wav.shape[-1] / 2) - 1
        hop_size = int(window_size / 2)
        short_audio = True
    spec = librosa.magphase(librosa.stft(wav, n_fft=window_size, hop_length=hop_size, win_length=window_size, window=np.ones, center=True))[0]
    frames2time = hop_size/sample_rate 
    rms = librosa.feature.rms(S=spec, frame_length=window_size, hop_length=hop_size, center=True, pad_mode='zero').squeeze()
    rms =  np.nan_to_num(rms)
    if hasattr(rms, "__len__"):
        if smoothing_window>len(rms):
            smoothing_window = len(rms)//3
        if smoothing_window>1:
            rms = scipy.signal.savgol_filter(rms, smoothing_window, 2) # window size 3, polynomial order 2
        # if min_duration>len(rms):
        #     min_duration = len(rms)/2 
        allowed = int(3 * sample_rate / hop_size)
        db_values = 20 * np.log10(rms)
        ### replace nans with minimum value
        db_values[np.isnan(db_values)] = db_values.min()
        if short_audio:
            start_end = [(0.,wav.shape[-1]/sample_rate)]
            norms = [rms.mean()]
            return norms, start_end
        start_end = []
        norms = []
        event_in_progress = False
        start_index = None
        for i, db in enumerate(db_values):
            if db >= db_treshold and not event_in_progress:
                ### Start of a new event
                start_index = i
                event_in_progress = True
            elif db < db_treshold and event_in_progress:
                ### End of the current event
                if i - start_index > int(min_duration * sample_rate / hop_size):
                    start_end.append([np.maximum(0,(start_index-1))*frames2time, (i - 1)*frames2time])
                    norms.append(rms[start_index:i-1].mean())
                    event_in_progress = False

        #### Handle case where the last event reaches the end of the list
        if event_in_progress and i - start_index > int(min_duration * sample_rate / hop_size):
            start_end.append((np.maximum(0,(start_index-1))*frames2time, (len(db_values) - 1)*frames2time))
            norms.append(rms[start_index : len(rms) - 1].mean())
        start_end = np.array(start_end)
        norms = np.array(norms)
    else:
        norms = np.array([])
        start_end = np.array([])
    return norms, start_end


def extract_noise_between_events(noise, start_end, sample_rate, window_type='triangular'):
    """
    Extract noise segments between events and stitch them together with a window.
    Events are enlarged by 0.2s before and 0.4s after to ensure clean noise extraction.
    
    Parameters
    ----------
    noise : torch.Tensor
        The noise signal tensor
    start_end : np.ndarray
        Array of (start, end) times in seconds
    sample_rate : int
        Sample rate of the audio
    window_type : str
        Type of window for stitching ('triangular', 'hann', 'hamming')
    
    Returns
    -------
    torch.Tensor
        Stitched noise signal
    """
    if len(start_end) == 0:
        return noise
    
    # Enlarge events by 0.2s before and 0.4s after, ensuring no negative timestamps
    enlarged_events = []
    for s, e in start_end:
        # Enlarge: 0.2s before start, 0.4s after end
        enlarged_start = max(0.0, s - 0.2)  # Ensure no negative start time
        enlarged_end = e + 0.4
        enlarged_events.append((enlarged_start, enlarged_end))
    
    # Convert to sample indices
    start_stop = [[int(s * sample_rate), int(e * sample_rate)] for s, e in enlarged_events]
    
    # Find gaps between enlarged events
    noise_segments = []
    last_end = 0
    
    for start_idx, end_idx in start_stop:
        if start_idx > last_end:
            # Extract noise segment between events
            segment = noise[..., last_end:start_idx]
            if segment.shape[-1] > 0:
                noise_segments.append(segment)
        last_end = end_idx
    
    # Add noise after last event if there's remaining audio
    if last_end < noise.shape[-1]:
        segment = noise[..., last_end:]
        if segment.shape[-1] > 0:
            noise_segments.append(segment)
    
    if not noise_segments:
        return noise
    
    # Stitch segments together with windowing
    if len(noise_segments) == 1:
        return noise_segments[0]
    
    # Create window for smooth transitions
    if window_type == 'triangular':
        window = torch.linspace(0, 1, 2, device=noise.device)
    elif window_type == 'hann':
        window = torch.hann_window(2, device=noise.device)
    elif window_type == 'hamming':
        window = torch.hamming_window(2, device=noise.device)
    else:
        window = torch.ones(2, device=noise.device)
    
    # Stitch with overlapping windows
    overlap_samples = min(100, noise_segments[0].shape[-1] // 4)  # 100 samples or 1/4 of first segment
    
    stitched = noise_segments[0]
    for i in range(1, len(noise_segments)):
        current_segment = noise_segments[i]
        
        if overlap_samples > 0 and stitched.shape[-1] > overlap_samples and current_segment.shape[-1] > overlap_samples:
            # Apply window to overlap region
            fade_out = window[1:].flip(0)  # 1 to 0
            fade_in = window[1:]  # 0 to 1
            
            # Fade out end of previous segment
            stitched[..., -overlap_samples:] *= fade_out
            
            # Fade in start of current segment
            current_segment[..., :overlap_samples] *= fade_in
            
            # Concatenate
            stitched = torch.cat([stitched, current_segment], dim=-1)
        else:
            # Simple concatenation if no overlap possible
            stitched = torch.cat([stitched, current_segment], dim=-1)
    
    return stitched


def get_chunks(audio, noise, sample_rate, start_end, duration=4., compute_noise=True, merge=False, amplitude_augment=False):
    duration_samples = duration * sample_rate
    offset = int(duration_samples / 2 )
    start_stop = [[np.maximum(0,int(s*sample_rate)),np.minimum(int(e*sample_rate),audio.shape[-1])] for s,e in start_end if s < e]
    ### add silence between start and end in audio 
    audio[...,0:start_stop[0][0]] = 0
    for i in range(1,len(start_stop)):
        if (start_stop[i][0] - start_stop[i-1][1]) > duration_samples/10:
            audio[...,start_stop[i-1][1]:start_stop[i][0]] = 0 
    audio[...,start_stop[-1][1]:] = 0
    
    ### events
    if merge:
        new_start_stop_offset = [[np.maximum(0, start_stop[i][0] - offset),np.minimum(start_stop[i][1]+offset, audio.shape[-1])] for i in range(len(start_stop))]
        new_start_stop = [new_start_stop_offset[0]]
        for i in range(1,len(new_start_stop_offset)):
            overlap = int(max(0, min(new_start_stop[-1][1], new_start_stop_offset[i][1]) - max(new_start_stop[-1][0], new_start_stop_offset[i][0])))
            if overlap < int(0.6 * duration_samples):
                new_start_stop.append(new_start_stop_offset[i])
    else:
        new_start_stop = [[np.maximum(0, start_stop[i][0] - offset),np.minimum(start_stop[i][1]+offset, audio.shape[-1])] for i in range(len(start_stop))]

    audio_signal = []
    for idx in new_start_stop:
        audio_signal=[audio[...,slice(int(idx[0]), np.minimum(int(idx[0])+duration_samples,int(idx[1])))] for idx in new_start_stop]
        new_audio_signal = []
        if amplitude_augment:
            for i, audio in enumerate(audio_signal):
                audio = audio * np.random.uniform(0.9, 1.1)
                
    ### noise
    audio_noise = None
    if compute_noise:
        start = 0
        noise_start_stop = []
        if start_stop[0][0] > duration_samples:
            noise_start_stop.append([0, start_stop[0][0]])
        for i in range(len(start_stop)-1):
            if start_stop[i+1][0] - start_stop[i][1] > duration_samples:
                noise_start_stop.append([start_stop[i][1], start_stop[i+1][0]])
        if (audio.shape[-1] - start_stop[-1][1]) > duration_samples:
            noise_start_stop.append([start_stop[-1][1], audio.shape[-1]])
        if len(noise_start_stop)>0:
            audio_noise = [noise[...,slice(int(idx[0]), int(idx[1]))] for idx in noise_start_stop]

    return audio_signal, audio_noise


def to_json_folder(data_dict, args):
    json_dict = {'train':[], 'valid':[]}
    for split, dirs in data_dict.items():
        for d in dirs:
            meta=denoiser.audio.find_audio_files(d)
            if 'valid' not in data_dict.keys() and split=='train':
                random.shuffle(meta)
                if args.num_valid > 0:
                    json_dict['valid'] += meta[:args.num_valid]
                json_dict['train'] += meta[args.num_valid:]
            else:
                json_dict[split] += meta
    return json_dict

def to_json_list(data_dict):
    json_dict={}
    for split, filelist in data_dict.items():
        meta=denoiser.audio.build_meta(filelist)
        if split not in json_dict.keys():
            json_dict[split] = []
        json_dict[split].extend(meta)
    return json_dict

def write_json(json_dict, filename, args):
    for split, meta in json_dict.items():       
        out_dir = os.path.join(args.out_dir, 'egs', args.experiment, split)
        os.makedirs(out_dir, exist_ok=True)
        fname = os.path.join(out_dir, filename)
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=4)
            
def generate_json(args, step=0):
    args.experiment = get_experiment_code(args,step)
    
    clean_dirs_dict = {}
    
    md = pd.read_csv(os.path.join(args.out_dir, args.experiment+".csv"))
    md.sort_values(by='metric',ascending=False,ignore_index=True)
    ### filter out top top_ratio of files
    if args.use_top<1:
        n_drop = int(len(md)*(1-args.use_top))
        md.drop(md.tail(n_drop).index,inplace=True)
        
    filenames = md['fn'].values.tolist()
    filenames = [f for f in filenames if os.path.exists(f)]
        
    if len(filenames)>args.num_valid:
        if args.num_valid>0:
            clean_dirs_dict['valid']=filenames[:args.num_valid]
        clean_dirs_dict['train']=filenames[args.num_valid:]
    else:
        clean_dirs_dict['train']=filenames

    json_dict = to_json_list(clean_dirs_dict)
    write_json(json_dict, 'clean.json', args)
    
    if args.noise_dir is not None:
        noise_dirs_dict = {'train':[os.path.join(args.noise_dir,f) for f in os.listdir(args.noise_dir)]}
        json_dict_noise = to_json_list(noise_dirs_dict)
    else:
        noise_dirs_dict = {}
        
        md_noise = pd.read_csv(os.path.join(args.out_dir, args.experiment+"_noise.csv"))
        md_noise.sort_values(by='metric',ascending=False,ignore_index=True)
        if args.use_top<1:
            n_drop = int(len(md_noise)*(1-args.use_top))
            md_noise.drop(md_noise.tail(n_drop).index,inplace=True)
        
        filenames_noise = md_noise['fn'].values.tolist()
        filenames_noise = [f for f in filenames_noise if os.path.exists(f)]

        if len(filenames_noise)>args.num_valid:
            if args.num_valid>0:
                noise_dirs_dict['valid']=filenames_noise[:args.num_valid]
            noise_dirs_dict['train']=filenames_noise[args.num_valid:]
        else:
            noise_dirs_dict['train']=filenames_noise

        json_dict_noise = to_json_list(noise_dirs_dict)
        

    write_json(json_dict_noise, 'noise.json', args)
    

def train(args,step=0):
    
    args.experiment = get_experiment_code(args,step)
    
    # if args.noise_dir is None:
    #     args.low_snr = 0 
    #     args.high_snr = 0
    
    train_path = os.path.join(args.out_dir, 'egs', args.experiment, 'train')
    valid_path = os.path.join(args.out_dir, 'egs', args.experiment, 'valid') if os.path.exists(os.path.join(args.out_dir, 'egs', args.experiment, 'valid', 'clean.json')) else None

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("denoise").setLevel(logging.DEBUG)
    
    experiment_logger = None
    if "cometml" in args:
        import comet_ml
        os.environ["COMET_API_KEY"] = args.cometml['api-key']
        experiment_logger = comet_ml.Experiment(args.cometml['api-key'], project_name=args.cometml['project'], log_code=False)
        experiment_logger.log_parameters(args)
        experiment_name = os.path.basename(os.getcwd())
        experiment_logger.set_name(experiment_name)
        

    denoiser.distrib.init(args)

    ##### Set the random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True) 
    rng = random.Random(args.seed)
    rngnp = np.random.default_rng(seed=args.seed)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed+worker_id)
        random.seed(worker_seed+worker_id)
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    rngth = torch.Generator(device=args.device)
    rngth.manual_seed(args.seed)

    if args.sample_rate == 48000:
        args.demucs.resample = 8
    
    if args.model=="biodenoising16k_dns48":
        model = denoiser.pretrained.get_model(args).to(args.device)
        sample_rate = model.sample_rate
        channels = model.chin
        if 'chout' in args.demucs:
            args.demucs['chout'] = args.demucs['chout']*args.nsources
    else:
        raise NotImplementedError
    
    if args.show:
        logger.info(model)
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2**20
        logger.info('Size: %.1f MB', mb)
        if hasattr(model, 'valid_length'):
            field = model.valid_length(1)
            logger.info('Field: %.1f ms', field / args.sample_rate * 1000)
        return

    assert args.batch_size % denoiser.distrib.world_size == 0
    args.batch_size //= denoiser.distrib.world_size
    length = int(args.segment * args.sample_rate)
    stride = int(args.stride * args.sample_rate)
    ##### This model requires a specific number of samples to avoid 0 padding during training
    if hasattr(model, 'valid_length'):
        length = model.valid_length(length)
    kwargs_valid = {"sample_rate": args.sample_rate,"seed": args.seed,"nsources": args.nsources,"exclude": args.exclude,"exclude_noise": args.exclude_noise, "rng":rng, "rngnp":rngnp, "rngth":rngth }
    kwargs_train = {"sample_rate": args.sample_rate,"seed": args.seed,"nsources": args.nsources,"exclude": args.exclude,"exclude_noise": args.exclude_noise, "rng":rng, "rngnp":rngnp, "rngth":rngth,
                    'repeat_prob': args.repeat_prob, 'random_repeat': args.random_repeat, 'random_pad': args.random_pad, 'silence_prob': args.silence_prob, 'noise_prob': args.noise_prob,
                    'normalize':args.normalize, 'random_gain':args.random_gain, 'low_gain':args.low_gain, 'high_gain':args.high_gain}
    # if 'seed=' in args.dset.train:
    #     args.dset.train = args.dset.train.replace('seed=', f'seed={args.seed}')
    # if args.continue_from and 'seed=' in args.continue_from:
    #     args.continue_from = args.continue_from.replace('seed=', f'seed={args.seed}')
    # if args.continue_pretrained and 'seed=' in args.continue_pretrained:
    #     args.continue_pretrained = args.continue_pretrained.replace('seed=', f'seed={args.seed}')
    
    ##### Building datasets and loaders
    if args.parallel_noise and not args.noise_dir:
        tr_dataset = datasets.NoiseCleanAdaptSetParNoise(
            train_path, length=length, stride=stride, pad=args.pad, epoch_size=args.epoch_size,
            low_snr=args.low_snr,high_snr=args.high_snr,**kwargs_train)
    else:
        tr_dataset = datasets.NoiseCleanAdaptSet(
            train_path, length=length, stride=stride, pad=args.pad, epoch_size=args.epoch_size,
            low_snr=args.low_snr,high_snr=args.high_snr,**kwargs_train)
    tr_loader = denoiser.distrib.loader(
        tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
    if valid_path:
        cv_dataset = datasets.NoiseCleanValidSet(
            valid_path, length=length, stride=0, pad=False, epoch_size=args.epoch_size,
            low_snr=args.low_snr,high_snr=args.high_snr,**kwargs_valid)
        cv_loader = denoiser.distrib.loader(
            cv_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers//4)
    else:
        cv_loader = None
    
    if hasattr(args, 'test_dir') and args.test_dir is not None and os.path.exists(args.test_dir):
        del kwargs_valid["exclude"]
        del kwargs_valid["exclude_noise"]
        del kwargs_valid["rng"]
        del kwargs_valid["rngnp"]
        del kwargs_valid["rngth"]
        if isinstance(args.test_dir, str):
            test_path = {'test':args.test_dir}
        tt_dataset = {}
        tt_loader = {}
        for key, value in test_path.items():
            tt_dataset[key] = denoiser.data.NoisyCleanSet(value, stride=0, pad=False, with_path=True, **kwargs_valid)
            tt_loader[key] = denoiser.distrib.loader(tt_dataset[key], batch_size=1, shuffle=False, num_workers=args.num_workers//4)
    else:
        tt_loader = None
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}

    print("Train size", len(tr_loader.dataset))
    
    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    args.lr = float(args.lr)
    if args.optim == "adam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        #total_steps = int(args.epochs * len(tr_loader))
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)#, cycle_momentum=False
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.optim == "lion":
        import lion_pytorch
        optimizer = lion_pytorch.Lion(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        #total_steps = int(args.epochs * len(tr_loader))
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)#, cycle_momentum=False
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)
    
    ##### Construct Solver
    solver = denoiser.solver.Solver(data, model, optimizer, args, rng=rng, rngnp=rngnp, rngth=rngth, seed=args.seed, experiment_logger=experiment_logger, scheduler=scheduler)
    solver.train()

def _infer_column(
    available_columns, keywords, fallback_name
):  # pragma: no cover - simple helper
    """
    Infer a column name from a list of available columns based on keywords.

    Parameters
    ----------
    available_columns : Iterable
        Column names available in the annotation table.
    keywords : list[str]
        Lowercase substrings to look for in the column names.
    fallback_name : Any
        Value to return if no suitable column is found.
    """
    cols_lower = {str(c).lower(): c for c in available_columns}
    for name_lower, original in cols_lower.items():
        if any(k in name_lower for k in keywords):
            return original
    return fallback_name


def preprocess_with_annotations(args):
    """
    Preprocess audio files using annotation files.
    
    For each audio file in noisy_dir, find a corresponding annotation file,
    extract segments according to annotations, and write them to new directories.
    """
    logger.info("Preprocessing audio files with annotations")
    
    if args.processed_dir is None:
        args.processed_dir = os.path.join(args.out_dir, "preprocessed")
    
    # Create output directories
    signal_dir = os.path.join(args.processed_dir, "signal")
    noise_dir = os.path.join(args.processed_dir, "noise")
    os.makedirs(signal_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)
    
    # Find all audio files in noisy_dir
    audio_files = []
    for ext in ['.wav', '.WAV', '.flac', '.FLAC', '.mp3', '.MP3']:
        audio_files.extend(list(Path(args.noisy_dir).glob(f"*{ext}")))
    
    if not audio_files:
        logger.error(f"No audio files found in {args.noisy_dir}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files")

    # Resolve which annotation extension to use when none is explicitly given.
    # Sequentially:
    #   1) If any *.csv exists in noisy_dir, use .csv for all files.
    #   2) Else, if any *.tsv exists, use .tsv.
    #   3) Else, if any *.txt exists, use .txt.
    # This avoids mixing extensions and matches the "check sequentially" requirement.
    ext_arg = getattr(args, "annotations_extension", None)
    resolved_ext = None
    if ext_arg and str(ext_arg).lower() != "auto":
        resolved_ext = ext_arg
    else:
        base_dir = Path(args.noisy_dir)
        for auto_ext in [".csv", ".tsv", ".txt"]:
            if any(base_dir.glob(f"*{auto_ext}")):
                resolved_ext = auto_ext
                break
        if resolved_ext is None:
            logger.warning(
                f"No annotation files with extensions .csv, .tsv, or .txt found in "
                f"{args.noisy_dir}; will attempt per-file lookup for all three."
            )
    processed_count = 0
    for audio_path in audio_files:
        # Find annotation file with the same base name
        base_name = audio_path.stem

        # If we resolved a single extension, only try that.
        # Otherwise, fall back to per-file probing of all three.
        candidate_paths = []
        if resolved_ext is not None:
            candidate_paths.append(Path(args.noisy_dir) / f"{base_name}{resolved_ext}")
        else:
            for auto_ext in [".csv", ".tsv", ".txt"]:
                candidate_paths.append(Path(args.noisy_dir) / f"{base_name}{auto_ext}")

        annotation_path = None
        for cand in candidate_paths:
            if cand.exists():
                annotation_path = cand
                break

        if annotation_path is None:
            logger.warning(f"No annotation file found for {audio_path}")
            continue
        
        # Load audio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:  # Convert stereo to mono if needed
                waveform = waveform.mean(dim=0, keepdim=True)
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            continue
        
        # Load annotations (support both CSV and TSV by letting pandas infer the separator)
        try:
            # sep=None with engine="python" lets pandas automatically detect comma, tab, etc.
            annotations = pd.read_csv(annotation_path, sep=None, engine="python")
        except Exception as e:
            logger.error(f"Error loading annotation file {annotation_path}: {e}")
            continue

        begin_col = args.annotations_begin_column
        end_col = args.annotations_end_column

        cols = list(annotations.columns)
        # Treat missing/empty/'auto' or non-existent names as "infer"
        if (
            begin_col is None
            or str(begin_col).strip() == ""
            or str(begin_col).lower() == "auto"
            or begin_col not in cols
        ):
            inferred_begin = _infer_column(cols, ["begin", "start"], None)
            if inferred_begin is not None:
                logger.info(
                    f"Inferred annotations begin column '{inferred_begin}' "
                    f"for file {annotation_path}"
                )
                begin_col = inferred_begin

        if (
            end_col is None
            or str(end_col).strip() == ""
            or str(end_col).lower() == "auto"
            or end_col not in cols
        ):
            inferred_end = _infer_column(cols, ["end", "stop"], None)
            if inferred_end is not None:
                logger.info(
                    f"Inferred annotations end column '{inferred_end}' "
                    f"for file {annotation_path}"
                )
                end_col = inferred_end

        # Check if required columns exist after inference
        if begin_col is None or end_col is None:
            logger.error(
                "Could not infer begin/end columns from annotation file "
                f"{annotation_path}. Available columns: {list(annotations.columns)}"
            )
            continue
        
        # Filter annotations by label if specified
        if args.annotations_label_column!='None' and args.annotations_label_column is not None and args.annotations_label_value is not None:
            annotations = annotations[annotations[args.annotations_label_column] == args.annotations_label_value]
            if len(annotations) == 0:
                logger.warning(f"No annotations with label {args.annotations_label_value} found in {annotation_path}")
                continue
        
        # Extract segments
        audio_duration = waveform.shape[1] / sample_rate
        segments = []
        for _, row in annotations.iterrows():
            start_time = float(row[begin_col])
            end_time = float(row[end_col])
            
            if start_time >= end_time or start_time < 0 or end_time > audio_duration:
                logger.warning(f"Invalid segment {start_time}-{end_time} in {annotation_path}")
                continue
                
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            
            segments.append((start_sample, end_sample))
        
        if not segments:
            logger.warning(f"No valid segments found in {annotation_path}")
            continue
        
        # Sort segments by start time
        segments.sort(key=lambda x: x[0])
        
        # Extract signal segments
        for i, (start_sample, end_sample) in enumerate(segments):
            segment = waveform[:, start_sample:end_sample]
            signal_file = os.path.join(signal_dir, f"{base_name}_segment_{i}.wav")
            torchaudio.save(signal_file, segment, sample_rate)
        
        # Extract noise segments (everything outside annotated segments)
        noise_parts = []
        last_end = 0
        for start_sample, end_sample in segments:
            if start_sample > last_end:
                noise_parts.append(waveform[:, last_end:start_sample])
            last_end = end_sample
        
        ### do not include last end part for the sake of doing few shot learning afterwards
        # if last_end < waveform.shape[1]:
        #     noise_parts.append(waveform[:, last_end:])
        
        if noise_parts:
            # Concatenate noise parts
            noise_segment = torch.cat(noise_parts, dim=1)
            noise_file = os.path.join(noise_dir, f"{base_name}_noise.wav")
            torchaudio.save(noise_file, noise_segment, sample_rate)
        
        processed_count += 1
    
    logger.info(f"Processed {processed_count} audio files")
    
    if processed_count > 0:
        # Update the input directories
        args.original_noisy_dir = args.noisy_dir
        args.noisy_dir = signal_dir
        args.original_noise_dir = args.noise_dir
        args.noise_dir = noise_dir
        logger.info(f"Updated noisy_dir to {args.noisy_dir}")
        logger.info(f"Updated noise_dir to {args.noise_dir}")
    else:
        logger.error("No audio files were successfully processed")

def run_adaptation(args):
    """
    Run the complete adaptation process with multiple steps.
    
    Args:
        args: Arguments containing configuration for the adaptation process
    """
    # Apply preprocessing if annotations are used
    if hasattr(args, 'annotations') and args.annotations:
        preprocess_with_annotations(args)
    
    os.makedirs(os.path.join(args.out_dir, 'checkpoints'), exist_ok=True)

    for step in range(args.steps):
        denoise(args, step=step)  # No need to capture the returned model
        generate_json(args, step=step)
        if step > 0:
            # Get the basename of the checkpoint file (without the directory path)
            checkpoint_basename = os.path.basename(args.checkpoint_file)
            args.continue_from = os.path.join(args.out_dir, 'checkpoints', checkpoint_basename)
            args.checkpoint_file = checkpoint_basename.replace('_step'+str(step-1)+'.th', '_step'+str(step)+'.th')
        else:
            args.continue_from = ''
            args.checkpoint_file = args.checkpoint_file.replace('.th', '_step0.th')
        args.checkpoint_file = os.path.join(args.out_dir, 'checkpoints', args.checkpoint_file)
        # Ensure history_file is just the basename before constructing the path
        history_basename = os.path.basename(args.history_file) if args.history_file else 'history.json'
        args.history_file = os.path.join(args.out_dir, 'checkpoints', history_basename)
        train(args, step=step)
        args.model_path = args.checkpoint_file
        args.lr = args.lr * 0.5
    
    # Run the training again on the whole unannotated audio
    if hasattr(args, 'annotations') and args.annotations:
        if hasattr(args, 'original_noisy_dir'):
            logger.info(f"Restoring original noisy_dir from {args.noisy_dir} to {args.original_noisy_dir}")
            args.noisy_dir = args.original_noisy_dir
        if hasattr(args, 'original_noise_dir') and args.noise_dir is not None:
            logger.info(f"Restoring original noise_dir from {args.noise_dir} to {args.original_noise_dir}")
            args.noise_dir = args.original_noise_dir
        ### delete the generated audio files in out_dir
        for file in os.listdir(args.out_dir):
            if file.endswith('.wav'):
                os.remove(os.path.join(args.out_dir, file))
        ### delete the generated json files in out_dir
        for file in os.listdir(args.out_dir):
            if file.endswith('.json'):
                os.remove(os.path.join(args.out_dir, file))
        args.steps = 2 * args.steps
        for step in range(step+1, args.steps):
            denoise(args, step=step)  # No need to capture the returned model
            generate_json(args, step=step)
            if step > 0:
                # Get the basename of the checkpoint file (without the directory path)
                checkpoint_basename = os.path.basename(args.checkpoint_file)
                args.continue_from = os.path.join(args.out_dir, 'checkpoints', checkpoint_basename)
                args.checkpoint_file = checkpoint_basename.replace('_step'+str(step-1)+'.th', '_step'+str(step)+'.th')
            else:
                args.continue_from = ''
                args.checkpoint_file = args.checkpoint_file.replace('.th', '_step0.th')
            args.checkpoint_file = os.path.join(args.out_dir, 'checkpoints', args.checkpoint_file)
            # Ensure history_file is just the basename before constructing the path
            history_basename = os.path.basename(args.history_file) if args.history_file else 'history.json'
            args.history_file = os.path.join(args.out_dir, 'checkpoints', history_basename)
            train(args, step=step)
            args.model_path = args.checkpoint_file
            args.lr = args.lr * 0.5
    
    # Final denoising step with selection table filtering if enabled
    denoise(args, step=step+1)  # Final denoising step, no need to return model


# ---------------------------------------------------------------------------
# Basic self-tests for annotation utilities
# ---------------------------------------------------------------------------

def test_infer_column_begin_end():
    """
    Simple sanity check for _infer_column keyword matching.

    This is intentionally lightweight so it can run in a regular pytest
    session without additional fixtures.
    """
    cols = ["Label", "Begin_Time", "End_Time"]
    begin = _infer_column(cols, ["begin", "start"], None)
    end = _infer_column(cols, ["end", "stop"], None)
    assert begin == "Begin_Time"
    assert end == "End_Time"


def test_infer_column_case_insensitive_and_fallback():
    """_infer_column should be case-insensitive and return fallback if needed."""
    cols = ["label", "START_SEC", "STOP_SEC"]
    begin = _infer_column(cols, ["begin", "start"], None)
    end = _infer_column(cols, ["end", "stop"], None)
    assert begin == "START_SEC"
    assert end == "STOP_SEC"

    # No matching keyword → fallback returned
    cols2 = ["Label", "Time"]
    begin2 = _infer_column(cols2, ["begin", "start"], "Begin")
    assert begin2 == "Begin"


def test_pandas_parses_csv_with_auto_sep(tmp_path=None):
    """
    Ensure that pd.read_csv with sep=None, engine='python' correctly
    parses a comma-separated annotations file.
    """
    import tempfile

    if tmp_path is None:
        tmp_dir = tempfile.mkdtemp(prefix="biodenoising_test_")
        tmp_path = Path(tmp_dir)

    csv_path = tmp_path / "example.csv"
    csv_path.write_text("Label,Begin,End\nCall,1.0,2.0\n", encoding="utf-8")

    df = pd.read_csv(csv_path, sep=None, engine="python")
    assert list(df.columns) == ["Label", "Begin", "End"]
    assert df.iloc[0]["Label"] == "Call"
    assert float(df.iloc[0]["Begin"]) == 1.0
    assert float(df.iloc[0]["End"]) == 2.0


def test_pandas_parses_tsv_with_auto_sep(tmp_path=None):
    """
    Ensure that pd.read_csv with sep=None, engine='python' correctly
    parses a tab-separated annotations file.
    """
    import tempfile

    if tmp_path is None:
        tmp_dir = tempfile.mkdtemp(prefix="biodenoising_test_")
        tmp_path = Path(tmp_dir)

    tsv_path = tmp_path / "example.tsv"
    tsv_path.write_text("Label\tBegin\tEnd\nCall\t3.0\t4.0\n", encoding="utf-8")

    df = pd.read_csv(tsv_path, sep=None, engine="python")
    assert list(df.columns) == ["Label", "Begin", "End"]
    assert df.iloc[0]["Label"] == "Call"
    assert float(df.iloc[0]["Begin"]) == 3.0
    assert float(df.iloc[0]["End"]) == 4.0
