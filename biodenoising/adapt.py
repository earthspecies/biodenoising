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

logger = logging.getLogger(__name__)

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


def get_dataset(noisy_dir, sample_rate, channels):
    if noisy_dir:
        files = denoiser.audio.find_audio_files(noisy_dir)
    else:
        logger.warning(
            "Small sample set was not provided by noisy_dir. "
            "Skipping denoising.")
        return None
    return denoiser.audio.Audioset(files, with_path=True,
                    sample_rate=sample_rate, channels=channels, convert=True)


def _estimate_and_save(model, noisy_signals, filenames, out_dir, step, sample_rate, args):
    ### process
    if args.noise_reduce:
        noisy_signals = noisy_signals[0,0].to('cpu').numpy()  
        noisy_signals = noisereduce.reduce_noise(y=noisy_signals, sr=sample_rate)
        noisy_signals = torch.from_numpy(noisy_signals[None,None,:]).to(args.device).float()
    
    ### Forward
    estimate = get_estimate(model, noisy_signals, args)

    if args.transform == 'none':
        save_wavs(estimate, filenames, os.path.join(out_dir,args.experiment), sr=sample_rate)
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
                
            #save_wavs(estimate_write, filenames, os.path.join(out_dir,args.method+'_'+args.transform + str(i)) , sr=sample_rate)
                    
        save_wavs(estimate_sum/4., filenames, os.path.join(out_dir,args.experiment), sr=sample_rate)
        
    return [],[]        

def _estimate_and_save_chunks(model, noisy_signals, filenames, out_subdir, step, sample_rate, args):
    original_noisy_signals = noisy_signals.clone()
    revecho=denoiser.augment.RevEcho(0.99)
    max_value = noisy_signals.abs().max()
    noisy_signals = noisy_signals[0,0].to('cpu').numpy()  
    if args.noise_reduce:
        noisy_signals = noisereduce.reduce_noise(y=noisy_signals, sr=sample_rate)

    ### remove dc component
    noisy_signals = highpass(noisy_signals, sample_rate, cutoff=args.highpass)
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
    
    full_estimate_noise = noisy_signals - estimate
    if args.transform == 'none':
        peaks, norms = get_peaks(estimate.to('cpu').numpy().squeeze(), sample_rate, peak_height=args.peak_height)
        res = []
        res_noise = []
        if len(peaks) > 0:
            noise = original_noisy_signals-estimate
            if args.noise_dir is None:
                for i in range(3):
                    signal = get_estimate(model, noise, args)
                    noise = noise - signal
                noise = 3 * noise  
                noise = noise/noise.abs().max() if noise.abs().max() > 1 else noise
        
            if args.revecho > 0:
                estimate = revecho(torch.stack((estimate,estimate)))[0]
            if args.amp_scale:
                estimate = estimate * max_value / estimate.abs().max()
            estimate, estimate_noise = get_chunks(estimate, noise, sample_rate, peaks, duration_samples=args.length)
            for j, chunk in enumerate(estimate):
                allf = save_wavs(chunk, filenames, os.path.join(out_subdir,args.experiment), '_'+str(j), sr=sample_rate)
                res.append([allf[0],norms[j], os.path.join(out_subdir,args.experiment)])
            if estimate_noise is not None and args.noise_dir is None:
                for j, chunk in enumerate(estimate_noise):
                    allfnoise = save_wavs(chunk, filenames, os.path.join(out_subdir,args.experiment+'_noise'), '_'+str(j), sr=sample_rate)
                    res_noise.append([allfnoise[0],1-np.array(norms).max(), os.path.join(out_subdir,args.experiment+'_noise')])
            return res, res_noise    
        else:
            if args.noise_dir is None:
                noise = original_noisy_signals-estimate
                for i in range(3):
                    signal = get_estimate(model, noise, args)
                    noise = noise - signal
                noise = 3 * noise  
                noise = noise/noise.abs().max() if noise.abs().max() > 1 else noise
                allfnoise = save_wavs(noise, filenames, os.path.join(out_subdir,args.experiment+'_noise'), sr=sample_rate)
                res_noise.append([allfnoise[0],1., os.path.join(out_subdir,args.experiment+'_noise')])
            return res,res_noise

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
        
        res = []
        res_noise = []
        peaks, norms = get_peaks(estimate_sum.to('cpu').numpy().squeeze(), sample_rate, peak_height=args.peak_height)
        if len(peaks) > 0:
            noise = original_noisy_signals-estimate
            if args.noise_dir is None:
                for i in range(3):
                    signal = get_estimate(model, noise, args)
                    noise = noise - signal
                noise = 3 * noise  
                noise = noise/noise.abs().max() if noise.abs().max() > 1 else noise

            if args.revecho > 0:
                estimate_sum = revecho(torch.stack((estimate_sum,estimate_sum)))[0]
            if args.amp_scale:
                estimate_sum = estimate_sum * max_value / estimate_sum.abs().max()
            estimate_sum, estimate_noise = get_chunks(estimate_sum, noise/4., sample_rate, peaks, duration_samples=args.length)
            for j, chunk in enumerate(estimate_sum):
                allf = save_wavs(chunk/4., filenames, os.path.join(out_subdir,args.experiment), '_'+str(j), sr=sample_rate)
                res.append([allf[0],norms[j], os.path.join(out_subdir,args.experiment)])
            if estimate_noise is not None and args.noise_dir is None:
                for j, chunk in enumerate(estimate_noise):
                    allfnoise = save_wavs(chunk/4., filenames, os.path.join(out_subdir,args.experiment+'_noise'), '_'+str(j), sr=sample_rate)
                    res_noise.append([allfnoise[0],1-np.array(norms).max(), os.path.join(out_subdir,args.experiment+'_noise')])
            return res, res_noise    
        else:
            if args.noise_dir is None:
                noise = original_noisy_signals-estimate
                for i in range(3):
                    signal = get_estimate(model, noise, args)
                    noise = noise - signal
                noise = 3 * noise  
                noise = noise/noise.abs().max() if noise.abs().max() > 1 else noise
                allfnoise = save_wavs(noise, filenames, os.path.join(out_subdir,args.experiment+'_noise'), sr=sample_rate)
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
    if args.noise_reduce:
        args.peak_height *= 0.5
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
    
    dset = get_dataset(os.path.join(args.noisy_dir), sample_rate, channels)
    if dset is None:
        return
    loader = denoiser.distrib.loader(dset, batch_size=1, shuffle=False)
    
    denoiser.distrib.barrier()

    md = pd.DataFrame(columns=['fn','metric','dataset'])
    md_noise = pd.DataFrame(columns=['fn','metric','dataset'])
    npos = 0
    nneg = 0
    with ProcessPoolExecutor(args.num_workers) as pool:
        iterator = denoiser.utils.LogProgress(logger, loader, name="Denoising files")
        pendings = []
        for data in iterator:
            # Get batch data
            noisy_signals, filenames = data
            noisy_signals = noisy_signals.to(args.device)
            if args.device == 'cpu' and args.num_workers > 1:
                if step<args.steps:
                    pendings.append(
                        pool.submit(_estimate_and_save_chunks,
                                    model, noisy_signals, filenames, out_dir, step, sample_rate, args))
                else:
                    pendings.append(
                        pool.submit(_estimate_and_save,
                                    model, noisy_signals, filenames, out_dir, step, sample_rate, args))
            else:
                res_noise = None
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
                    if step<args.steps:
                        res, res_noise = _estimate_and_save_chunks(ola_model, noisy_signals, filenames, out_dir, step, sample_rate, args)
                    else:
                        res, res_noise = _estimate_and_save(ola_model, noisy_signals, filenames, out_dir, step, sample_rate, args)
                else:
                    if step<args.steps:
                        res, res_noise = _estimate_and_save_chunks(model, noisy_signals, filenames, out_dir, step, sample_rate, args)
                    else:
                        res, res_noise = _estimate_and_save(model, noisy_signals, filenames, out_dir, step, sample_rate, args)
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
        print("denoised with peaks %d files, without peaks %d files." % (npos, nneg))
        md.to_csv(os.path.join( out_dir, args.experiment+".csv"), index=False)
        md_noise.to_csv(os.path.join( out_dir, args.experiment+"_noise.csv"), index=False)
    return model

def get_peaks(wav, sample_rate, smoothing_window=6, peak_window=10, peak_height=0.01):
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
        peaks, _ = scipy.signal.find_peaks(rms, height=peak_height, distance=peak_window)
        allowed = int(3 * sample_rate / 512)
        norms = []
        for p in peaks:
            start = np.maximum(0, p-allowed)
            end = np.minimum(len(rms), p+allowed)
            ####norms.append(scipy.stats.kurtosis(spec[:,start:end].sum(axis=0), fisher=False)/rms[start:end].mean())
            norms.append(rms[start:end].max()-rms[start:end].min())
        peaks =  peaks * frames2time
    else:
        peaks = np.array([])
        norms = np.array([])
    
    return peaks, norms

def get_chunks(audio, noise, sample_rate, peaks, duration_samples=4*16000, compute_noise=True, offset=0.1*16000):
    peaks_samples = peaks * sample_rate
    left_shift = np.ceil(duration_samples / 2)
    right_shift = duration_samples - left_shift
    peaks_samples = np.clip(peaks_samples, left_shift, audio.shape[-1] - right_shift)
    start_stop = np.unique(np.stack([np.maximum(0,peaks_samples - left_shift), np.minimum(peaks_samples + right_shift,audio.shape[-1])], axis=-1), axis=0)
    new_start_stop = [start_stop[0]]
    for i in range(1,len(start_stop)):
        overlap = int(max(0, min(new_start_stop[-1][1], start_stop[i][1]) - max(new_start_stop[-1][0], start_stop[i][0])))
        if overlap < 0.6 * duration_samples:
            new_start_stop.append(start_stop[i])
    new_start_stop = np.array(new_start_stop)
    audio_signal=torch.stack([audio[...,slice(int(idx[0]), np.minimum(int(idx[0])+duration_samples,int(idx[1])))] for idx in new_start_stop])
    
    audio_noise = None
    if compute_noise:
        ### extract noise chunks from new_start_stop
        start = 0
        noise_start_stop = []
        for i in range(len(new_start_stop)-1):
            end = new_start_stop[i+1][0] - offset
            if end - start > duration_samples:
                noise_start_stop.append([start, end])
            start = end + offset
        if (audio.shape[-1] - new_start_stop[-1][1]) > duration_samples:
            noise_start_stop.append([new_start_stop[-1][1], audio.shape[-1]])
        if len(noise_start_stop)>0:
            audio_noise = [noise[...,slice(int(idx[0]), int(idx[1]))] for idx in noise_start_stop]

    return audio_signal, audio_noise


def to_json_folder(data_dict, args):
    json_dict = {'train':[], 'valid':[]}
    for split, dirs in data_dict.items():
        for d in dirs:
            meta=biodenoising.denoiser.audio.find_audio_files(d)
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
        md.drop(df.tail(n_drop).index,inplace=True)
        
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
        noise_dirs = {'train':[os.path.join(args.noise_dir,f) for f in os.listdir(os.path.join(args.noise_dir))]}
        json_dict_noise = to_json_folder(noise_dirs, args)
    else:
        noise_dirs_dict = {}
        
        md_noise = pd.read_csv(os.path.join(args.out_dir, args.experiment+"_noise.csv"))
        md_noise.sort_values(by='metric',ascending=False,ignore_index=True)
        if args.use_top<1:
            n_drop = int(len(md_noise)*(1-args.use_top))
            md_noise.drop(df.tail(n_drop).index,inplace=True)
        
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
    
    train_path = os.path.join(os.path.join(args.out_dir, 'egs', args.experiment, 'train'))
    valid_path = os.path.join(os.path.join(args.out_dir, 'egs', args.experiment, 'valid')) if os.path.exists(os.path.join(os.path.join(args.out_dir, 'egs', args.experiment, 'valid'))) else None
    test_path = '/home/marius/data/zebra_finch_denoising/16000'
    # test_path = ''
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
    
    if os.path.exists(test_path):
        del kwargs_valid["exclude"]
        del kwargs_valid["exclude_noise"]
        del kwargs_valid["rng"]
        del kwargs_valid["rngnp"]
        del kwargs_valid["rngth"]
        if isinstance(test_path, str):
            test_path = {'test':test_path}
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

