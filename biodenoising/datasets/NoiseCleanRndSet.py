import os
import json
import random
import numpy as np
from .. import denoiser
import string
import torch
import torchaudio
import logging
from ..denoiser.noise import get_noise

logger = logging.getLogger(__name__)

def time_scaling(signal, scaling):
    output_size = int(signal.shape[-1] * scaling)
    ref = torch.arange(output_size, device=signal.device, dtype=signal.dtype).div_(scaling)

    ref1 = ref.clone().type(torch.int64)
    ref2 = torch.min(ref1 + 1, torch.full_like(ref1, signal.shape[-1] - 1, dtype=torch.int64))
    r = ref - ref1.type(ref.type())
    scaled_signal = signal[..., ref1] * (1 - r) + signal[..., ref2] * r

    return scaled_signal

def snr_scale(clean, noise, snr):
    energy_signal = torch.linalg.vector_norm(clean, ord=2, dim=-1) ** 2
    energy_noise = torch.linalg.vector_norm(noise, ord=2, dim=-1) ** 2
    original_snr_db = 10 * (torch.log10(energy_signal) - torch.log10(energy_noise))
    scale = 10 ** ((original_snr_db - snr) / 20.0)  # (*,)
    scaled_noise = scale.unsqueeze(-1) * noise
    return scaled_noise
    
class NoiseCleanRndSet:
    def __init__(self, json_dir, length=None, stride=None, epoch_size=0,
                    pad=True, sample_rate=None, low_snr=0, high_snr=20, seed=0,
                    rng=None, rngnp=None, rngth=None, nsources=1, silence_prob=0, noise_prob=0,
                    exclude=None, exclude_noise=None, repeat_prob=0.5, random_repeat=False, 
                    random_pad=False, use_subset_noise=False, use_subset_clean=False,
                    normalize=False, random_gain=False, low_gain=0.5, high_gain=1., balance_clean=None):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param size: the iteration size
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        :param low_snr: the minimum snr for the mixture
        :param high_snr: the maximum snr for the mixture
        :param seed: the random seed
        :param nsources: the number of sources in the mixture
        :param exclude: list of files to exclude
        :param exclude_noise: list of noise files to exclude
        """
        noise_json = os.path.join(json_dir, 'noise.json')
        clean_json = os.path.join(json_dir, 'clean.json')
        with open(noise_json, 'r') as f:
            self.noise = json.load(f)
        with open(clean_json, 'r') as f:
            self.clean = json.load(f)
        if balance_clean is not None and use_subset_clean:
            self.balance_clean = balance_clean
            len_main_subset = len([c for c in self.clean if os.path.basename(c[0]).startswith(balance_clean)])
            self.weights = {}
            if len_main_subset ==0: 
                print('No clean files found in subset {}'.format(balance_clean))
                self.balance_clean = None
                balance_clean = None
                use_subset_clean = False
                self.weights = None
            else:
                self.len_main_subset = len_main_subset
                logger.info('Found {} clean files in main subset {}'.format(len_main_subset, balance_clean))
        if exclude is not None and len(exclude)>0:
            if isinstance(exclude, str):
                exclude = exclude.split(',')
            self.clean = [c for c in self.clean if all([not os.path.basename(c[0]).startswith(e) for e in exclude])]
        if exclude_noise is not None and len(exclude_noise)>0:
            if isinstance(exclude_noise, str):
                exclude_noise = exclude_noise.split(',')
            self.noise = [c for c in self.noise if all([not os.path.basename(c[0]).startswith(e) for e in exclude_noise])]
        assert len(self.clean)>0, 'No clean files found'
        assert len(self.noise)>0, 'No noise files found'
        assert epoch_size>0, 'epoch_size must be greater than 0'
        self.epoch_size = epoch_size
        self.sample_rate = sample_rate
        self.low_snr = low_snr
        self.high_snr = high_snr
        self.seed = seed
        self.nsources = nsources
        self.silence_prob = silence_prob
        self.noise_prob = noise_prob
        self.normalize = normalize
        self.random_gain = random_gain
        self.low_gain = low_gain
        self.high_gain = high_gain
        
        self.rng = rng if rng is not None else random.Random(self.seed)
        self.rngnp = rngnp if rngnp is not None else np.random.default_rng(seed=self.seed)
        self.rngth = rngth if rngth is not None else torch.manual_seed(self.seed)

        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate, 'repeat_prob': repeat_prob, 'random_repeat': random_repeat, 'random_pad': random_pad, 'use_subset': use_subset_clean, 'random_obj':self.rng}
        kw_noise = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate, 'repeat_prob': 1., 'random_repeat': False, 'random_pad': False, 'use_subset': use_subset_noise, 'random_obj':self.rng}
        self.clean_set = denoiser.audio.Audioset(self.clean, **kw)
        self.noise_set = denoiser.audio.Audioset(self.noise, **kw_noise)
        
        for subset in self.clean_set.subsets.keys():
            logger.info("Found {} clean files in subset {}".format(sum(self.clean_set.num_examples_subsets[subset]), subset))
            if balance_clean is not None and use_subset_clean:
                self.weights[subset] = len_main_subset/len([c for c in self.clean if os.path.basename(c[0]).startswith(subset)])
        if balance_clean is not None and use_subset_clean:
            logger.info("Weights: {} subsets {}".format(self.weights,self.clean_set.subsets.keys()))
        if balance_clean is not None and use_subset_clean:
            self.weights_normed = {}
            if len(self.weights.values()) >0:
                sum_weights = sum(self.weights.values())
                for k,v in self.weights.items():
                    self.weights_normed[k] = v/sum_weights
        for subset in self.noise_set.subsets.keys():
            logger.info("Found {} noise files in subset {}".format(sum(self.noise_set.num_examples_subsets[subset]), subset))
        logger.info("Found {} clean files".format(self.clean_set.total_examples()))
        logger.info("Found {} noise files".format(self.noise_set.total_examples()))
        # self.clean_set.files, self.clean_set.num_examples = self.shuffle(self.clean_set.files, self.clean_set.num_examples)
        # self.noise_set.files, self.noise_set.num_examples = self.shuffle(self.noise_set.files, self.noise_set.num_examples)
    
    def shuffle(self,list1,list2):
        zipped = list(zip(list1, list2))
        self.rng.shuffle(zipped)
        list1, list2 = zip(*zipped)
        return list1, list2

    def __getitem__(self, index):
        idx_noise = self.rng.randrange(len(self.noise_set))
        idx_clean = self.rng.randrange(len(self.clean_set))
        noise = self.noise_set[idx_noise]
        clean = self.clean_set[idx_clean]
        noise = torch.nan_to_num(noise, nan=1e-8, posinf=1, neginf=-1)
        
        ### add generated noise
        if self.noise_prob > 0 and self.rng.random() < self.noise_prob:
            other_noise = torch.from_numpy(get_noise(2*noise.shape[-1],self.rngnp)).type_as(noise)
            other_noise = (other_noise - other_noise.min())/(other_noise.max() - other_noise.min())*(noise.max()-noise.min()) + noise.min() 
            gain = self.rngnp.uniform(low=0.05, high=0.5)      
            other_noise = time_scaling(other_noise, self.rngnp.uniform(0.5, 2.))
            other_noise = other_noise[...,:noise.shape[-1]]
            noise += gain*other_noise

            
        if self.silence_prob > 0 and self.rng.random() < self.silence_prob:
            clean = torch.zeros_like(noise)
            noisy = noise 
        else:
            clean = self.clean_set[idx_clean]
            clean = torch.nan_to_num(clean, nan=1e-8, posinf=1, neginf=-1)
            
            if self.normalize:
                clean = self.normalize_audio(clean)
            
                if self.random_gain:
                    gain = self.rngnp.uniform(low=self.low_gain, high=self.high_gain) 
                    clean = gain*clean
                
            ### remix with specified snr    
            snr = self.rngnp.uniform(self.low_snr, self.high_snr) 
    
            snr = torch.tensor([snr])
            
            noise = snr_scale(clean, noise, snr)
            noisy = clean + noise

        # ### for debugging purposes, write wav files
        # randomfn = ''.join(random.choices(string.ascii_lowercase,k=6))+'.wav'
        # os.makedirs(os.path.abspath('tmp'),exist_ok=True)
        # outfile = os.path.abspath(os.path.join('tmp',randomfn))
        # torchaudio.save(uri=outfile,src=noisy,sample_rate=self.sample_rate,format='wav',encoding='PCM_F',channels_first=True)
        # torchaudio.save(uri=outfile.replace('.wav','-source.wav'),src=clean.detach().cpu(),sample_rate=self.sample_rate,format='wav',encoding='PCM_F',channels_first=True)
        # torchaudio.save(uri=outfile.replace('.wav','-noise.wav'),src=noise.detach().cpu(),sample_rate=self.sample_rate,format='wav',encoding='PCM_F',channels_first=True)
        
        return noisy, clean

    def __len__(self):
        return self.epoch_size
    
    def normalize_audio(self, S, norm=float('inf'), axis=-1, threshold = None, fill=None):
        ''' 
        Adapted from librosa.util.normalize
        '''
        ### Avoid div-by-zero
        if threshold is None:
            threshold = torch.finfo(S.dtype).tiny
            
        ### All norms only depend on magnitude, let's do that first
        mag = np.abs(S).float()

        ### For max/min norms, filling with 1 works
        fill_norm = 1    
        
        if norm is None:
            return S

        elif norm == float('inf'):
            length, _ = torch.max(mag, axis=axis, keepdims=True)

        elif norm == float('-inf'):
            length, _ = torch.min(mag, axis=axis, keepdims=True)

        elif norm == 0:
            if fill is True:
                raise ParameterError("Cannot normalize with norm=0 and fill=True")

            length = torch.sum(mag > 0, axis=axis, keepdims=True).float()

        elif norm > 0:
            length = torch.sum(mag**norm, axis=axis, keepdims=True) ** (1.0 / norm)

            if axis is None:
                fill_norm = mag.size ** (-1.0 / norm)
            else:
                fill_norm = mag.shape[axis] ** (-1.0 / norm)

        else:
            raise ParameterError(f"Unsupported norm: {repr(norm)}")

        ### indices where norm is below the threshold
        small_idx = length < threshold

        Snorm = torch.empty_like(S)
        if fill is None:
            # Leave small indices un-normalized
            length[small_idx] = 1.0
            Snorm[:] = S / length

        elif fill:
            ### If we have a non-zero fill value, we locate those entries by
            ### doing a nan-divide.
            ### If S was finite, then length is finite (except for small positions)
            length[small_idx] = np.nan
            Snorm[:] = S / length
            Snorm[torch.isnan(Snorm)] = fill_norm
        else:
            ### Set small values to zero by doing an inf-divide.
            ### This is safe (by IEEE-754) as long as S is finite.
            length[small_idx] = float('inf')
            Snorm[:] = S / length

        return Snorm


class NoiseCleanValidSet(NoiseCleanRndSet):
    def __init__(self, json_dir, **kwargs):
        NoiseCleanRndSet.__init__(self, json_dir=json_dir, **kwargs)
        self.snrs = [torch.tensor(self.rngnp.uniform(self.low_snr, self.high_snr)) for _ in range(np.maximum(len(self.noise_set),len(self.clean_set)))]
    
    def __getitem__(self, index):
        idx_noise = index % len(self.noise_set)
        idx_clean = index % len(self.clean_set)
        noise = self.noise_set[idx_noise]
        clean = self.clean_set[idx_clean]
        # ### normalize
        # gain = np.random.uniform(low=0.4, high=0.9)
        # noise = self.normalize_audio(noise)
        # clean = self.normalize_audio(clean)
        noise = torch.nan_to_num(noise, nan=1e-8, posinf=1, neginf=-1)
        clean = torch.nan_to_num(clean, nan=1e-8, posinf=1, neginf=-1)
        
        ### remix with specified snr 
        #snr = self.rngnp.uniform(self.low_snr, self.high_snr) 
        snr = self.snrs[index]
        
        # noisy = noise + clean
        snr = torch.tensor([snr])
        noisy = torchaudio.functional.add_noise(waveform=clean, noise=noise, snr=snr, lengths=None)
        
        # ### for debugging purposes, write wav files
        # randomfn = ''.join(random.choices(string.ascii_lowercase,k=6))+'.wav'
        # outfile = os.path.abspath(os.path.join('tmp',randomfn))
        # torchaudio.save(filepath=outfile,src=self.normalize_audio(noisy),sample_rate=self.sample_rate,format='wav',encoding='PCM_F',channels_first=True)
        # torchaudio.save(filepath=outfile.replace('.wav','-source.wav'),src=clean.detach().cpu(),sample_rate=self.sample_rate,format='wav',encoding='PCM_F',channels_first=True)
        # torchaudio.save(filepath=outfile.replace('.wav','-noise.wav'),src=noise.detach().cpu(),sample_rate=self.sample_rate,format='wav',encoding='PCM_F',channels_first=True)
        
        return noisy, clean

    def __len__(self):
        return np.maximum(len(self.noise_set),len(self.clean_set))

class NoiseClean2BalancedSet(NoiseCleanRndSet):
    def __init__(self, json_dir, **kwargs):
        kwargs['use_subset_noise'] = True
        kwargs['use_subset_clean'] = True
        kwargs['balance_clean'] = 'asa'
        NoiseCleanRndSet.__init__(self, json_dir=json_dir, **kwargs)
    
    def __len__(self):
        return self.len_main_subset * len(self.clean_set.subsets)
    
    def __getitem__(self, index):
        subset_id_noise = index % len(self.noise_set.subsets)
        self.noise_set.set_subset(subset_id_noise)
        idx_noise = self.rng.randrange(len(self.noise_set))
        
        subset_id_clean = index % len(self.clean_set.subsets)
        self.clean_set.set_subset(subset_id_clean)
        idx_clean = self.rng.randrange(len(self.clean_set))
        
        noise = self.noise_set[idx_noise]
        noise = torch.nan_to_num(noise, nan=1e-8, posinf=1, neginf=-1)
        
        ### add generated noise
        if self.noise_prob > 0 and self.rng.random() < self.noise_prob:
            other_noise = torch.from_numpy(get_noise(2*noise.shape[-1],self.rngnp)).type_as(noise)
            other_noise = (other_noise - other_noise.min())/(other_noise.max() - other_noise.min())*(noise.max()-noise.min()) + noise.min() 
            gain = self.rngnp.uniform(low=0.05, high=0.5)      
            other_noise = time_scaling(other_noise, self.rngnp.uniform(0.5, 2.))
            other_noise = other_noise[...,:noise.shape[-1]]
            noise += gain*other_noise

            
        if self.silence_prob > 0 and self.rng.random() < self.silence_prob:
            clean = torch.zeros_like(noise)
            noisy = noise 
        else:
            clean = self.clean_set[idx_clean]
            clean = torch.nan_to_num(clean, nan=1e-8, posinf=1, neginf=-1)
            
            if self.normalize:
                clean = self.normalize_audio(clean)
            
                if self.random_gain:
                    gain = self.rngnp.uniform(low=self.low_gain, high=self.high_gain) 
                    clean = gain*clean
                
            ### remix with specified snr    
            snr = self.rngnp.uniform(self.low_snr, self.high_snr) 
    
            snr = torch.tensor([snr])
            
            noise = snr_scale(clean, noise, snr)
            noisy = clean + noise

        return noisy, clean
    
class NoiseClean1BalancedSet(NoiseCleanRndSet):
    def __init__(self, json_dir, **kwargs):
        kwargs['use_subset_noise'] = True
        #kwargs['use_subset_clean'] = True
        NoiseCleanRndSet.__init__(self, json_dir=json_dir, **kwargs)
    
    def __len__(self):
        return len(self.clean_set)
    
    def __getitem__(self, index):
        subset_id = index % len(self.noise_set.subsets)
        self.noise_set.set_subset(subset_id)
        #idx_noise = random.randrange(len(self.noise_set))
        idx_noise = self.rng.randrange(len(self.noise_set))
        idx_clean = index
        noise = self.noise_set[idx_noise]
        noise = torch.nan_to_num(noise, nan=1e-8, posinf=1, neginf=-1)
        
        ### add generated noise
        if self.noise_prob > 0 and self.rng.random() < self.noise_prob:
            other_noise = torch.from_numpy(get_noise(2*noise.shape[-1],self.rngnp)).type_as(noise)
            other_noise = (other_noise - other_noise.min())/(other_noise.max() - other_noise.min())*(noise.max()-noise.min()) + noise.min() 
            gain = self.rngnp.uniform(low=0.05, high=0.5)      
            other_noise = time_scaling(other_noise, self.rngnp.uniform(0.5, 2.))
            other_noise = other_noise[...,:noise.shape[-1]]
            noise += gain*other_noise

            
        if self.silence_prob > 0 and self.rng.random() < self.silence_prob:
            clean = torch.zeros_like(noise)
            noisy = noise 
        else:
            clean = self.clean_set[idx_clean]
            clean = torch.nan_to_num(clean, nan=1e-8, posinf=1, neginf=-1)
            
            if self.normalize:
                clean = self.normalize_audio(clean)
            
                if self.random_gain:
                    gain = self.rngnp.uniform(low=self.low_gain, high=self.high_gain) 
                    clean = gain*clean
                
            ### remix with specified snr    
            snr = self.rngnp.uniform(self.low_snr, self.high_snr) 
    
            snr = torch.tensor([snr])
            
            noise = snr_scale(clean, noise, snr)
            noisy = clean + noise

        return noisy, clean
    
class NoiseClean1WeightedSet(NoiseCleanRndSet):
    def __init__(self, json_dir, **kwargs):
        kwargs['use_subset_noise'] = True
        kwargs['use_subset_clean'] = True
        kwargs['balance_clean'] = 'asa'
        NoiseCleanRndSet.__init__(self, json_dir=json_dir, **kwargs)
    
    def __len__(self):
        if self.balance_clean is not None:
            length = 0 
            for k,weight in self.weights.items():
                length += int(weight * len(self.clean_set.subsets[k]))
        else:
            length = len(self.clean_set)
        return length
    
    def __getitem__(self, index):
        subset_id_noise = index % len(self.noise_set.subsets)
        self.noise_set.set_subset(subset_id_noise)
        #idx_noise = random.randrange(len(self.noise_set))
        idx_noise = self.rng.randrange(len(self.noise_set))
        
        if hasattr(self, 'weights_normed'):
            subset_id_clean = self.rngnp.choice(list(range(len(self.clean_set.subsets))), p=list(self.weights_normed.values()))
            self.clean_set.set_subset(subset_id_clean)
        # idx_clean = index
        idx_clean = self.rng.randrange(len(self.clean_set))
        
        noise = self.noise_set[idx_noise]
        noise = torch.nan_to_num(noise, nan=1e-8, posinf=1, neginf=-1)
        
        ### add generated noise
        if self.noise_prob > 0 and self.rng.random() < self.noise_prob:
            other_noise = torch.from_numpy(get_noise(2*noise.shape[-1],self.rngnp)).type_as(noise)
            other_noise = (other_noise - other_noise.min())/(other_noise.max() - other_noise.min())*(noise.max()-noise.min()) + noise.min() 
            gain = self.rngnp.uniform(low=0.05, high=0.5)      
            other_noise = time_scaling(other_noise, self.rngnp.uniform(0.5, 2.))
            other_noise = other_noise[...,:noise.shape[-1]]
            noise += gain*other_noise

            
        if self.silence_prob > 0 and self.rng.random() < self.silence_prob:
            clean = torch.zeros_like(noise)
            noisy = noise 
        else:
            clean = self.clean_set[idx_clean]
            clean = torch.nan_to_num(clean, nan=1e-8, posinf=1, neginf=-1)
            
            if self.normalize:
                clean = self.normalize_audio(clean)
            
                if self.random_gain:
                    gain = self.rngnp.uniform(low=self.low_gain, high=self.high_gain) 
                    clean = gain*clean
                
            ### remix with specified snr    
            snr = self.rngnp.uniform(self.low_snr, self.high_snr) 
    
            snr = torch.tensor([snr])
            
            noise = snr_scale(clean, noise, snr)
            noisy = clean + noise 
        return noisy, clean
    

class NoiseCleanAdaptSet(NoiseCleanRndSet):
    def __init__(self, json_dir, **kwargs):
        NoiseCleanRndSet.__init__(self, json_dir=json_dir, **kwargs)
    
    def __len__(self):
        return len(self.clean_set)
    
    def __getitem__(self, index):
        
        idx_noise = self.rng.randrange(len(self.noise_set))
        noise = self.noise_set[idx_noise]
        
        ### add generated noise
        if self.noise_prob > 0 and self.rng.random() < self.noise_prob:
            other_noise = torch.from_numpy(get_noise(2*noise.shape[-1],self.rngnp)).type_as(noise)
            other_noise = (other_noise - other_noise.min())/(other_noise.max() - other_noise.min())*(noise.max()-noise.min()) + noise.min() 
            gain = self.rngnp.uniform(low=0.05, high=0.5)      
            other_noise = time_scaling(other_noise, self.rngnp.uniform(0.5, 2.))
            other_noise = other_noise[...,:noise.shape[-1]]
            noise += gain*other_noise

            
        if self.silence_prob > 0 and self.rng.random() < self.silence_prob:
            clean = torch.zeros_like(noise)
            noisy = noise 
        else:
            clean = self.clean_set[index]
            clean = torch.nan_to_num(clean, nan=1e-8, posinf=1, neginf=-1)
            
            if self.normalize:
                clean = self.normalize_audio(clean)
            
                if self.random_gain:
                    gain = self.rngnp.uniform(low=self.low_gain, high=self.high_gain) 
                    clean = gain*clean
                
            ### remix with specified snr    
            snr = self.rngnp.uniform(self.low_snr, self.high_snr) 
    
            snr = torch.tensor([snr])
            
            noise = snr_scale(clean, noise, snr)
            noisy = clean + noise

        return noisy, clean
    
class NoiseCleanAdaptSetParNoise(NoiseCleanRndSet):
    def __init__(self, json_dir, **kwargs):
        kwargs['use_subset_noise'] = True
        kwargs['use_subset_clean'] = True
        NoiseCleanRndSet.__init__(self, json_dir=json_dir, **kwargs)
    
    def __len__(self):
        length = 0 
        for k,w in self.clean_set.subsets.items():
            length += len(self.clean_set.subsets[k])
        return length
    
    def __getitem__(self, index):
        subsets_clean = list(self.clean_set.subsets.keys())
        subset_id_clean = index % len(subsets_clean)
        subset_clean = subsets_clean[subset_id_clean]
        self.clean_set.set_subset(subset_id_clean)
        idx_clean = self.rng.randrange(len(self.clean_set))
        
        clean = self.clean_set[idx_clean]

        subsets_noise = list(self.noise_set.subsets.keys())
        if subset_clean in subsets_noise:
            subset_id_noise = subsets_noise.index(subset_clean)
            self.noise_set.set_subset(subset_id_noise)
            idx_noise = self.rng.randrange(len(self.noise_set))
        else:
            subset_id_noise = index % len(subsets_noise)
            self.noise_set.set_subset(subset_id_noise)
            idx_noise = self.rng.randrange(len(self.noise_set))
        noise = self.noise_set[idx_noise]

        ### add generated noise
        if self.noise_prob > 0 and self.rng.random() < self.noise_prob:
            other_noise = torch.from_numpy(get_noise(2*noise.shape[-1],self.rngnp)).type_as(noise)
            other_noise = (other_noise - other_noise.min())/(other_noise.max() - other_noise.min())*(noise.max()-noise.min()) + noise.min() 
            gain = self.rngnp.uniform(low=0.05, high=0.5)      
            other_noise = time_scaling(other_noise, self.rngnp.uniform(0.5, 2.))
            other_noise = other_noise[...,:noise.shape[-1]]
            noise += gain*other_noise

            
        if self.silence_prob > 0 and self.rng.random() < self.silence_prob:
            clean = torch.zeros_like(noise)
            noisy = noise 
        else:
            clean = torch.nan_to_num(clean, nan=1e-8, posinf=1, neginf=-1)
            
            if self.normalize:
                clean = self.normalize_audio(clean)
            
                if self.random_gain:
                    gain = self.rngnp.uniform(low=self.low_gain, high=self.high_gain) 
                    clean = gain*clean
                
            ### remix with specified snr    
            snr = self.rngnp.uniform(self.low_snr, self.high_snr) 
    
            snr = torch.tensor([snr])
            
            noise = snr_scale(clean, noise, snr)
            noisy = clean + noise

        return noisy, clean