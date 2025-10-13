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
import pandas as pd
import noisereduce
import librosa

import torch
import torchaudio
import julius 

import biodenoising

logger = logging.getLogger(__name__)

DATA_SPLITS = {}#{'sabiod_samples': 'valid'}
EXCLUDE = []
### classes for yamnet
positive_ids = list(range(0,46)) + list(range(51,56)) + list(range(59,276)) 
negative_ids = list(range(46,51)) + list(range(277,384)) + list(range(400,475)) + list(range(494,494)) + list(range(507,517)) 

def add_flags(parser):
    """
    Add the flags for the argument parser that are related to model loading and evaluation"
    """
    biodenoising.denoiser.pretrained.add_model_flags(parser)
    parser.add_argument('--device', default="cuda")
    parser.add_argument('--dry', type=float, default=0,
                        help='dry/wet knob coefficient. 0 is only denoised, 1 only input signal.')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--streaming', action="store_true",
                        help="true streaming evaluation for Demucs")


parser = argparse.ArgumentParser(
        'denoise',
        description="Generate denoised files")
add_flags(parser)
parser.add_argument("--out_dir", type=str, default="enhanced",
                    help="directory putting enhanced wav files")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="more loggging")
parser.add_argument("--step", default=0, type=int, help="step")
parser.add_argument("--force_sample_rate", default=0, type=int, help="Force the model to take samples of this sample rate")
parser.add_argument("--time_scale_factor", default=0, type=int, help="If the model has a different sample rate, play the audio slower or faster with this factor. If force_sample_rate this automatically changes.")
parser.add_argument("--segment", default=4, type=int, help="minimum segment size in seconds")
parser.add_argument("--method",choices=["biodenoising16k_dns48","noisereduce","demucs", "cleanunet"], default="biodenoising16k_dns48",help="Pre-trained model to use for denoising")
parser.add_argument("--tag", default="",help="This is used to tag the models at steps>0 with the origin of training data at step 0")
parser.add_argument("--seed", default=-1, type=int, help="seed for step>0")
parser.add_argument("--transform",choices=["none","time_scale"], default="none",help="Transform input by pitch shifting or time scaling")
parser.add_argument("--filtering",choices=["sklearn","yamnet"], default="sklearn",help="The method to use for filtering bad separations/peak detection. Yamnet requires tensorflow.")
parser.add_argument('--revecho', type=float, default=0,help='revecho probability')
parser.add_argument('--antialiasing', action="store_true",help="use an antialiasing filter when time scaling back")
parser.add_argument('--amp_scale', action="store_true",help="scale to the amplitude of the input")
parser.add_argument('--noise_reduce', action="store_true",help="use noisereduce preprocessing")
parser.add_argument("--noisy_dir", type=str, default=None,
                    help="path to the parent directory containing subdirectories with noisy wav files")
parser.add_argument("--rir_dir", type=str, default=None,
                    help="path to the directory containing room impulse responses")


def normalize(wav):
    return wav / max(wav.abs().max().item(), 1)

def highpass(wav, sample_rate, cutoff=20):
    [b,a] = scipy.signal.butter(4,cutoff, fs=sample_rate, btype='high')
    wav = scipy.signal.lfilter(b,a,wav)
    return wav

def lowpass(wav, sample_rate, cutoff=20):
    [b,a] = scipy.signal.butter(4,cutoff, fs=sample_rate, btype='low')
    wav = scipy.signal.lfilter(b,a,wav)
    return wav

def add_reverb(clean, rir_files, sample_rate):
    if random.random() < 0.5:
        # add reverb with selected RIR
        rir_index = random.randint(0,len(rir_files)-1)
        my_rir = rir_files[rir_index]
        samples_rir, fs_rir = torchaudio.load(my_rir)
        if fs_rir != sample_rate:
            # resampler = torchaudio.transforms.Resample(fs_rir, sample_rate, dtype=samples_rir.dtype)
            # samples_rir = resampler(samples_rir)
            samples_rir = julius.resample_frac(samples_rir, fs_rir, sample_rate)
        samples_rir = torch.nan_to_num(samples_rir)
        clean = torch.nan_to_num(clean)
        if clean.shape[-1]<samples_rir.shape[-1]:
            clean = torch.nn.functional.pad(clean, (0, samples_rir.shape[-1]-clean.shape[-1]), mode='constant', value=0)
        if samples_rir.ndim==1:
            samples_rir = np.array(samples_rir)
        else:
            samples_rir = samples_rir[0, :]
        samples_rir = samples_rir[int(sample_rate * 0.3) : int(sample_rate * 1.3)]   
        samples_rir = samples_rir / (torch.linalg.vector_norm(samples_rir, ord=2) + 1e-8)
        samples_rir = samples_rir.unsqueeze(0).unsqueeze(0).to(clean.device)
        augmented = torchaudio.functional.fftconvolve(clean, samples_rir)
        if torch.isnan(torch.abs(augmented).sum()) or torch.abs(augmented).sum().detach().item()==0:
            return clean
        return augmented[...,:clean.shape[-1]]
    else:
        return clean

def get_start_end(wav, sample_rate, smoothing_window=0, db_treshold=-40, min_duration=0.2):
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
            smoothing_window = np.maximum(1,len(rms)//3)
        if smoothing_window>1:
            rms = scipy.signal.savgol_filter(rms, smoothing_window, 2) # window size 3, polynomial order 2
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
            elif db < (db_treshold-10) and event_in_progress:
                ### End of the current event, leave room for release time
                if i - start_index > int(min_duration * sample_rate / hop_size):
                    start_end.append((np.maximum(0,(start_index-1))*frames2time, (i - 1)*frames2time))
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


def get_chunks(audio, sample_rate, start_end, duration=4., merge=False):
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

    audio_signal=[audio[...,slice(int(idx[0]), np.minimum(int(idx[0])+duration_samples,int(idx[1])))] for idx in new_start_stop]
    
    return audio_signal

def get_estimate(model, noisy, args):
    torch.set_num_threads(1)
    if args.method in ['demucs','biodenoising16k_dns48'] and args.streaming:
        streamer = biodenoising.denoiser.demucs.DemucsStreamer(model, dry=args.dry)
        with torch.no_grad():
            estimate = torch.cat([
                streamer.feed(noisy[0]),
                streamer.flush()], dim=1)[None]
    else:
        with torch.no_grad():
            estimate = model(noisy)
            estimate = (1 - args.dry) * estimate + args.dry * noisy
            estimate = normalize(estimate)
    return estimate

def time_scaling(signal, scaling):
    output_size = int(signal.shape[-1] * scaling)
    ref = torch.arange(output_size, device=signal.device, dtype=signal.dtype).div_(scaling)

    ref1 = ref.clone().type(torch.int64)
    ref2 = torch.min(ref1 + 1, torch.full_like(ref1, signal.shape[-1] - 1, dtype=torch.int64))
    r = ref - ref1.type(ref.type())
    scaled_signal = signal[..., ref1] * (1 - r) + signal[..., ref2] * r

    return scaled_signal

def save_wavs(estimates, noisy_sigs, filenames, out_dir, version=None, sr=16_000, write_noisy=False):
    # Write result
    allf = []
    for estimate, noisy, filename in zip(estimates, noisy_sigs, filenames):
        filename = os.path.join(out_dir, os.path.basename(filename).rsplit(".", 1)[0])
        if write_noisy:
            write(estimate, filename + str(version) + "_enhanced.wav", sr=sr)
            write(noisy, filename + str(version) +"_noisy.wav", sr=sr)
        else:
            write(estimate, filename + str(version) + ".wav", sr=sr)
        allf.append(filename + str(version) + ".wav")
    return allf  

def ensure_sample_rate(waveform, original_sample_rate,
                        desired_sample_rate=32000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        waveform = librosa.resample(waveform, orig_sr=original_sample_rate, target_sr=desired_sample_rate)
    return desired_sample_rate, waveform

def write(wav, filename, sr=16_000):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)

def get_dataset(noisy_dir, sample_rate, channels):
    print("Processing dataset {}".format(noisy_dir))
    if args.noisy_dir:
        files = biodenoising.denoiser.audio.find_audio_files(noisy_dir)
    else:
        logger.warning(
            "Small sample set was not provided by noisy_dir. "
            "Skipping denoising.")
        return None
    return biodenoising.denoiser.audio.Audioset(files, with_path=True,
                    sample_rate=sample_rate, channels=channels, convert=True)

def _estimate_and_save(model, noisy_signals, filenames, subdir, out_subdir, rir_files, clsmodel, sample_rate, args):
    if args.method=='noisereduce':
        noisy_signals = noisy_signals[0,0].to('cpu').numpy()  
        estimate = noisereduce.reduce_noise(y=noisy_signals, sr=sample_rate)
        norms, start_end = get_start_end()
        estimate = torch.from_numpy(estimate[None,None,:]).to(args.device).float()
        res = []
        if len(rir_files) > 0:
            estimate = add_reverb(estimate, rir_files, sample_rate)
        elif args.revecho > 0:
            estimate = revecho(torch.stack((estimate,estimate)))[0]
        if args.amp_scale:
            estimate = estimate * max_value / estimate.abs().max()
        allf = save_wavs(estimate, noisy_signals, filenames, out_subdir, sr=sample_rate)
        return [[allf[0],0, subdir]]    
    else:
        revecho=biodenoising.denoiser.augment.RevEcho(0.99)
        # std = noisy_signals.std(dim=-1, keepdim=True)
        # noisy_signals = noisy_signals / (1e-3 + std)
        max_value = noisy_signals.abs().max()
        ### process
        noisy_signals = noisy_signals[0,0].to('cpu').numpy()  
        if args.noise_reduce:
            noisy_signals = noisereduce.reduce_noise(y=noisy_signals, sr=sample_rate)
        ### dc component
        noisy_signals = highpass(noisy_signals, sample_rate, cutoff=2)
        noisy_signals = torch.from_numpy(noisy_signals[None,None,:]).to(args.device).float()
        
        if args.time_scale_factor != 0:
            noisy_signals_fwd = noisy_signals
            if args.antialiasing and args.time_scale_factor>0:
                ## anti-aliasing
                noisy_signals_fwd = torch.from_numpy(lowpass(noisy_signals.to('cpu').numpy(), sample_rate, cutoff=sample_rate//(args.time_scale_factor*4))).to(args.device).float()
            noisy_signals_fwd = time_scaling(noisy_signals_fwd, np.power(2, args.time_scale_factor*0.5))
        else:
            noisy_signals_fwd = noisy_signals
            
        ### Forward
        estimate = get_estimate(model, noisy_signals_fwd, args)
        
        if args.time_scale_factor != 0:
            if args.antialiasing and args.time_scale_factor>0:
                ## anti-aliasing
                estimate = torch.from_numpy(lowpass(estimate.to('cpu').numpy(), sample_rate, cutoff=sample_rate//(args.time_scale_factor*4))).to(args.device).float()
            estimate = time_scaling(estimate, np.power(2, -args.time_scale_factor*0.5))
        
            ### remove low frequency artifacts
            estimate = torch.from_numpy(highpass(estimate.to('cpu').numpy(), sample_rate)).to(args.device).float()
        
        if args.transform == 'none':
            norms, start_end = get_start_end(estimate.to('cpu').numpy().squeeze(), sample_rate)
            if len(start_end) > 0:
                res = []
                if len(rir_files) > 0:
                    estimate = add_reverb(estimate, rir_files, sample_rate)
                elif args.revecho > 0:
                    estimate = revecho(torch.stack((estimate,estimate)))[0]
                if args.amp_scale:
                    estimate = estimate * max_value / estimate.abs().max()
                estimate = get_chunks(estimate, sample_rate, start_end, duration=args.segment)
                for j, chunk in enumerate(estimate):
                    allf = save_wavs(chunk, noisy_signals, filenames, out_subdir, '_'+str(j), sr=sample_rate)
                    res.append([allf[0],norms[j], subdir])
                return res    
            else:
                return None
        else:
            ### we sum all the results here
            estimate_sum = estimate
            # estimates = []
            for i in range(1,4): ### animal sounds sit usually in higher frequencies; shift them down
                noisy_signals = torch.from_numpy(highpass(noisy_signals.to('cpu').numpy().squeeze(), sample_rate)).to(args.device)
                noisy_signals = noisy_signals[None,None,:].float()
                ### transform
                ### time scaling
                noisy_signals = time_scaling(noisy_signals, np.power(2, -0.5))
                
                ### forward
                estimate = get_estimate(model, noisy_signals, args)
                            
                ## anti-aliasing
                if i>0 and args.antialiasing:
                    estimate = torch.from_numpy(lowpass(estimate.to('cpu').numpy(), sample_rate, cutoff=sample_rate//(i*4))).to(args.device).float()
                
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
            
            norms, start_end = get_start_end(estimate.to('cpu').numpy().squeeze(), sample_rate)
            if len(start_end) > 0:
                res = []
                if len(rir_files) > 0:
                    estimate_sum = add_reverb(estimate_sum, rir_files, sample_rate)
                elif args.revecho > 0:
                    estimate_sum = revecho(torch.stack((estimate_sum,estimate_sum)))[0]
                if args.amp_scale:
                    estimate_sum = estimate_sum * max_value / estimate_sum.abs().max()
                estimate = get_chunks(estimate, sample_rate, start_end, duration=args.segment)
                for j, chunk in enumerate(estimate):
                    allf = save_wavs(chunk/4., noisy_signals, filenames, out_subdir, '_'+str(j), sr=sample_rate)
                    res.append([allf[0],norms[j], subdir])
                return res    
            else:
                return None
            
def denoise(args, model=None, local_out_dir=None):    
    sample_rate = args.force_sample_rate if args.force_sample_rate else 16000
    channels = 1
    args.length = args.segment * sample_rate
    # Load model
    if args.step > 0:
        assert args.model_path is not None, "model_path must be provided when step > 0"
    if args.method in ['demucs','biodenoising16k_dns48']:
        # args.dns64=True
        if not model:
            model = biodenoising.denoiser.pretrained.get_model(args).to(args.device)
        if args.force_sample_rate:
            sample_rate = args.force_sample_rate
            args.model_sample_rate = model.sample_rate
            args.time_scale_factor = int(np.floor(np.log2(args.force_sample_rate/model.sample_rate)))
            # if hasattr(model, 'upsample'):
            #     model.upsample = 2 * model.upsample
        else:
            sample_rate = model.sample_rate
            args.model_sample_rate = model.sample_rate
        args.length = args.segment * sample_rate
        if hasattr(model, 'valid_length'):
            args.length = model.valid_length(args.length)
        channels = model.chin
    elif args.method=='cleanunet':
        if not model:
            args.cleanunet_speech = True
            model = biodenoising.denoiser.pretrained.get_model(args).to(args.device)
        if args.force_sample_rate:
            sample_rate = args.force_sample_rate
            args.model_sample_rate = 16000
        else:
            sample_rate = 16000
            args.model_sample_rate = 16000
        args.length = args.segment * sample_rate
        channels = 1
    if args.method != 'noisereduce': 
        model.eval()
    if local_out_dir:
        out_dir = local_out_dir
    else:
        out_dir = args.out_dir
        
    rir_files = []
    if args.rir_dir is not None and os.path.isdir(args.rir_dir):
        rir_files = [os.path.join(root,file) for root,fdir,files in os.walk(args.rir_dir) for file in files if file.endswith('.wav') and not file.startswith('.')]
        rir_files.sort()
    
    if args.filtering=='yamnet':    
        import tensorflow_hub
        import tensorflow as tf 
        ### yamnet
        with tf.device('/CPU:0'):
            clsmodel = tensorflow_hub.load('https://tfhub.dev/google/yamnet/1')
    else:
        clsmodel = None
            
    subdirs = [ f.name for f in os.scandir(args.noisy_dir) if f.is_dir() and f not in tuple(EXCLUDE)]
    md = pd.DataFrame(columns=['fn','metric','dataset'])
    log = ''
    for root, dirs, files in os.walk(args.noisy_dir):
        if not dirs:
            subdir = os.path.basename(root)
            dset = get_dataset(root, sample_rate, channels)
            if dset is None:
                return
            loader = biodenoising.denoiser.distrib.loader(dset, batch_size=1)

            split = DATA_SPLITS[subdir] if subdir in DATA_SPLITS else 'train'
            print(subdir, split)
            if args.method == 'noisereduce':
                out_subdir = os.path.join(out_dir, split,args.method)
            else:
                if args.noise_reduce:
                    out_subdir = os.path.join(out_dir, split,'clean_'+args.method+ '_' + args.transform + '_nr'  + args.tag +'_step'+str(args.step))
                else:
                    out_subdir = os.path.join(out_dir, split,'clean_'+args.method+ '_' + args.transform + args.tag +'_step'+str(args.step))
                if args.seed>=0:
                    out_subdir += ',seed='+str(args.seed)
            if args.method in ['demucs','biodenoising16k_dns48']:
                if biodenoising.denoiser.distrib.rank == 0:
                    os.makedirs(out_subdir, exist_ok=True)
                biodenoising.denoiser.distrib.barrier()
            else:
                os.makedirs(out_subdir, exist_ok=True)

            npos = 0
            nneg = 0
            with ProcessPoolExecutor(args.num_workers) as pool:
                iterator = biodenoising.denoiser.utils.LogProgress(logger, loader, name="Denoising files")
                pendings = []
                for data in iterator:
                    # Get batch data
                    noisy_signals, filenames, _ = data
                    noisy_signals = noisy_signals.to(args.device)
                    if args.device == 'cpu' and args.num_workers > 1:
                        pendings.append(
                            pool.submit(_estimate_and_save,
                                        model, noisy_signals, filenames, subdir, out_subdir, rir_files, clsmodel, sample_rate, args))
                    else:
                        res = _estimate_and_save(model, noisy_signals, filenames, subdir, out_subdir, rir_files, clsmodel, sample_rate, args)
                        if res is not None:
                            npos += 1
                            for r in res:
                                md.loc[len(md)] = r
                        else:
                            nneg += 1
                if pendings:
                    print('Waiting for pending jobs...')
                    for pending in biodenoising.denoiser.utils.LogProgress(logger, pendings, updates=5, name="Denoising files"):
                        res = pending.result()
                        if res is not None:
                            npos += 1
                            for r in res:
                                md.loc[len(md)] = r
                        else:
                            nneg += 1
            print(subdir+": denoised calls %d files, without calls %d files." % (npos, nneg))
            log += subdir+": denoised calls %d files, without calls %d files.\n" % (npos, nneg)
    if args.method == 'noisereduce':
        md.to_csv(os.path.join( out_dir, args.method+".csv"), index=False)
        with open(os.path.join( out_dir, args.method+".log"), 'w') as f:
            f.write(log)
    else:
        if args.noise_reduce:
            experiment = 'clean_'+args.method+ '_' + args.transform + '_nr'  + args.tag +'_step'+str(args.step)
        else:
            experiment = 'clean_'+args.method+ '_' + args.transform + args.tag +'_step'+str(args.step)
        if args.seed>=0:
            experiment += ',seed='+str(args.seed)
        md.to_csv(os.path.join( out_dir, experiment+".csv"), index=False)
        with open(os.path.join( out_dir, experiment+".log"), 'w') as f:
            f.write(log)


if __name__ == "__main__":
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    denoise(args, local_out_dir=args.out_dir)

# python generate_training.py --out_dir /home/$USER/data/biodenoising48k/ --noisy_dir /home/$USER/data/biodenoising48k/dev/noisy/ --rir_dir /home/$USER/data/biodenoising16k/rir/ --method demucs --transform none --device cpu --force_sample_rate 48000 