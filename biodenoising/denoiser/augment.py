# Adapted from https://github.com/facebookresearch/demucs under the MIT License 
# Original Copyright (c) Earth Species Project. This work is based on Facebook's denoiser. 

#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import random
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F

from . import dsp


class Remix(nn.Module):
    """Remix.
    Mixes different noises with clean speech within a given batch
    """
    def __init__(self, rngth=None):
        """__init__.

        """
        super().__init__()
        self.rngth = rngth 

    def forward(self, sources):
        noise, clean = sources
        bs, *other = noise.shape
        device = noise.device
        perm = th.argsort(th.rand(bs, device=device, generator=self.rngth), dim=0)
        return th.stack([noise[perm], clean])

class Mixup(nn.Module):
    """Remix.
    Mixup transformations, sums different clean speech within a given batch
    """
    def __init__(self, prob=0., rngth=None, seed=None):
        """__init__.

        """
        super().__init__()
        if rngth is None:
            self.rngth = th.Generator(device='cpu').manual_seed(seed)
        else:
            self.rngth = rngth 
        self.prob = prob

    def forward(self, sources):
        noise, clean = sources
        bs, *other = noise.shape
        device = noise.device
        ### clean mixup
        mixup_mask = th.rand(bs, device=device, generator=self.rngth) <= self.prob
        perm = th.argsort(th.rand(bs, device=device, generator=self.rngth), dim=0)
        clean[mixup_mask] = (clean[mixup_mask] + clean[perm][mixup_mask]) / 2
        # ### noise mixup
        # mixup_mask = th.rand(bs, device=device, generator=self.rngth) <= self.prob
        # perm = th.argsort(th.rand(bs, device=device, generator=self.rngth), dim=0)
        # noise[mixup_mask] = (noise[mixup_mask] + noise[perm][mixup_mask]) / 2
        return th.stack([noise, clean])



class RevEcho(nn.Module):
    """
    Hacky Reverb but runs on GPU without slowing down training.
    This reverb adds a succession of attenuated echos of the input
    signal to itself. Intuitively, the delay of the first echo will happen
    after roughly 2x the radius of the room and is controlled by `first_delay`.
    Then RevEcho keeps adding echos with the same delay and further attenuation
    until the amplitude ratio between the last and first echo is 1e-3.
    The attenuation factor and the number of echos to adds is controlled
    by RT60 (measured in seconds). RT60 is the average time to get to -60dB
    (remember volume is measured over the squared amplitude so this matches
    the 1e-3 ratio).

    At each call to RevEcho, `first_delay`, `initial` and `RT60` are
    sampled from their range. Then, to prevent this reverb from being too regular,
    the delay time is resampled uniformly within `first_delay +- 10%`,
    as controlled by the `jitter` parameter. Finally, for a denser reverb,
    multiple trains of echos are added with different jitter noises.

    Args:
        - initial: amplitude of the first echo as a fraction
            of the input signal. For each sample, actually sampled from
            `[0, initial]`. Larger values means louder reverb. Physically,
            this would depend on the absorption of the room walls.
        - rt60: range of values to sample the RT60 in seconds, i.e.
            after RT60 seconds, the echo amplitude is 1e-3 of the first echo.
            The default values follow the recommendations of
            https://arxiv.org/ftp/arxiv/papers/2001/2001.08662.pdf, Section 2.4.
            Physically this would also be related to the absorption of the
            room walls and there is likely a relation between `RT60` and
            `initial`, which we ignore here.
        - first_delay: range of values to sample the first echo delay in seconds.
            The default values are equivalent to sampling a room of 3 to 10 meters.
        - repeat: how many train of echos with differents jitters to add.
            Higher values means a denser reverb.
        - jitter: jitter used to make each repetition of the reverb echo train
            slightly different. For instance a jitter of 0.1 means
            the delay between two echos will be in the range `first_delay +- 10%`,
            with the jittering noise being resampled after each single echo.
        - keep_clean: fraction of the reverb of the clean speech to add back
            to the ground truth. 0 = dereverberation, 1 = no dereverberation.
        - sample_rate: sample rate of the input signals.
    """

    def __init__(self, proba=0.5, initial=0.3, rt60=(0.3, 1.3), first_delay=(0.01, 0.03),
                 repeat=3, jitter=0.1, keep_clean=0.1, sample_rate=16000, rng=None, seed=42):
        super().__init__()
        self.proba = proba
        self.initial = initial
        self.rt60 = rt60
        self.first_delay = first_delay
        self.repeat = repeat
        self.jitter = jitter
        self.keep_clean = keep_clean
        self.sample_rate = sample_rate  
        self.seed = seed
        self.rng = rng if rng is not None else random.Random(self.seed)

    def _reverb(self, source, initial, first_delay, rt60):
        """
        Return the reverb for a single source.
        """
        length = source.shape[-1]
        reverb = th.zeros_like(source)
        for _ in range(self.repeat):
            frac = 1  # what fraction of the first echo amplitude is still here
            echo = initial * source
            while frac > 1e-3:
                # First jitter noise for the delay
                jitter = 1 + self.jitter * self.rng.uniform(-1, 1)
                delay = min(
                    1 + int(jitter * first_delay * self.sample_rate),
                    length)
                # Delay the echo in time by padding with zero on the left
                echo = F.pad(echo[:, :, :-delay], (delay, 0))
                reverb += echo

                # Second jitter noise for the attenuation
                jitter = 1 + self.jitter * self.rng.uniform(-1, 1)
                # we want, with `d` the attenuation, d**(rt60 / first_ms) = 1e-3
                # i.e. log10(d) = -3 * first_ms / rt60, so that
                attenuation = 10**(-3 * jitter * first_delay / rt60)
                echo *= attenuation
                frac *= attenuation
        return reverb

    def forward(self, wav):
        if random.random() >= self.proba:
            return wav
        noise, clean = wav
        # Sample characteristics for the reverb
        initial = random.random() * self.initial
        first_delay = random.uniform(*self.first_delay)
        rt60 = random.uniform(*self.rt60)

        reverb_noise = self._reverb(noise, initial, first_delay, rt60)
        # Reverb for the noise is always added back to the noise
        noise += reverb_noise
        reverb_clean = self._reverb(clean, initial, first_delay, rt60)
        # Split clean reverb among the clean speech and noise
        clean += self.keep_clean * reverb_clean
        noise += (1 - self.keep_clean) * reverb_clean

        return th.stack([noise, clean])


class BandMask(nn.Module):
    """BandMask.
    Maskes bands of frequencies. Similar to Park, Daniel S., et al.
    "Specaugment: A simple data augmentation method for automatic speech recognition."
    (https://arxiv.org/pdf/1904.08779.pdf) but over the waveform.
    """

    def __init__(self, maxwidth=0.2, bands=120, sample_rate=16_000, rng=None, seed=42):
        """__init__.

        :param maxwidth: the maximum width to remove
        :param bands: number of bands
        :param sample_rate: signal sample rate
        """
        super().__init__()
        self.maxwidth = maxwidth
        self.bands = bands
        self.sample_rate = sample_rate
        self.seed = seed
        self.rng = rng if rng is not None else random.Random(self.seed)
        

    def forward(self, wav):
        bands = self.bands
        bandwidth = int(abs(self.maxwidth) * bands)
        mels = dsp.mel_frequencies(bands, 40, self.sample_rate/2) / self.sample_rate
        low = self.rng.randrange(bands)
        high = self.rng.randrange(low, min(bands, low + bandwidth))
        filters = dsp.LowPassFilters([mels[low], mels[high]]).to(wav.device)
        low, midlow = filters(wav)
        # band pass filtering
        out = wav - midlow + low
        return out


class Shift(nn.Module):
    """Shift."""

    def __init__(self, shift=8192, same=False, rngth=None):
        """__init__.

        :param shift: randomly shifts the signals up to a given factor
        :param same: shifts both clean and noisy files by the same factor
        """
        super().__init__()
        self.shift = shift
        self.same = same
        self.rngth = rngth 
        

    def forward(self, wav):
        sources, batch, channels, length = wav.shape
        length = length - self.shift
        if self.shift > 0:
            if not self.training:
                wav = wav[..., :length]
            else:
                offsets = th.randint(
                    self.shift,
                    [1 if self.same else sources, batch, 1, 1], device=wav.device, generator=self.rngth)
                offsets = offsets.expand(sources, -1, channels, -1)
                indexes = th.arange(length, device=wav.device)
                wav = wav.gather(3, indexes + offsets)
        return wav


class TimeScale(nn.Module):
    """Fast time scale."""

    def __init__(self, scale=2.0, target=1, rngnp=None, seed=42):
        """__init__.

        :param scale: randomly scales up to this maximum factor
        """
        super().__init__()
        self.scale = scale
        self.target = target
        self.seed = seed
        self.rngnp = rngnp if rngnp is not None else np.random.default_rng(seed=self.seed)
        

    def forward(self, wav):
        sources, batch, channels, length = wav.shape
        
        ### what to augment: noise, clean, or both
        if self.target==-1:
            targets = [i for i in range(wav.shape[0])]
        else: 
            targets = [self.target]
            
        for t in targets:
            signal = wav[t]
            scaling = np.power(self.scale, self.rngnp.uniform(-1, 1))
            output_size = int(signal.shape[-1] * scaling)
            ref = th.arange(output_size, device=signal.device, dtype=signal.dtype).div_(scaling)

            ref1 = ref.clone().type(th.int64)
            ref2 = th.min(ref1 + 1, th.full_like(ref1, signal.shape[-1] - 1, dtype=th.int64))
            r = ref - ref1.type(ref.type())
            scaled_signal = signal[..., ref1] * (1 - r) + signal[..., ref2] * r
            
            ## trim or zero pad to the original size 
            if scaled_signal.shape[-1] > signal.shape[-1]:
                nframes_offset = (scaled_signal.shape[-1] - signal.shape[-1]) // 2
                scaled_signal = scaled_signal[...,nframes_offset:nframes_offset+signal.shape[-1]]
            else:
                nframes_diff = (signal.shape[-1] - scaled_signal.shape[-1])
                pad_left = int(np.random.uniform() * nframes_diff)
                pad_right = nframes_diff - pad_left
                scaled_signal = F.pad(input=scaled_signal, pad=(pad_left, pad_right, 0, 0, 0, 0), mode='constant', value=0)
                
            wav[t] = scaled_signal
        
        return wav
    

class Flip(nn.Module):

    def __init__(self, p = 0., rngth=None):
        super(Flip, self).__init__()

        self.p = p
        self.rngth = rngth

    def forward(self, x):
        if x.dim() > 2:
            flip_mask = th.rand(x.shape[0], device=x.device, generator=self.rngth) <= self.p
            x[flip_mask] = x[flip_mask].flip(-1)
        else:
            if th.rand(1, generator=self.rngth) <= self.p:
                x = x.flip(0)

        return x


# class RandomNoise(nn.Module):
#     """Add randomly generated noise."""

#     def __init__(self, rng=None, rngnp=None, rngth=None, seed=42):
#         """__init__.

#         :param scale: randomly scales up to this maximum factor
#         """
#         super().__init__()
#         self.seed = seed
#         self.rng = rng if rng is not None else random.Random(self.seed)
#         self.rngnp = rngnp if rngnp is not None else np.random.default_rng(seed=self.seed)
#         self.rngth = rngth if rngth is not None else th.manual_seed(self.seed)
        

#     def forward(self, wav):

#         other_noise = torch.from_numpy(utils.noise.get_noise(len(noise["audio"]))).type_as(noise["audio"])
#         if other_noise.sum() != 0:
#             other_noise = (other_noise - other_noise.min())/(other_noise.max() - other_noise.min())*(noise["audio"].max()-noise["audio"].min()) + noise["audio"].min() 
#             gain = np.random.uniform(low=0.05, high=0.5)
#             noise["audio"] = noise["audio"] + gain*other_noise

#         ### time scaling the noise
#         if random.random() < self.conf["data"]["prob_scaling"]: 
#             noise["audio"] = utils.time_scaling(noise["audio"], self.conf["data"]["max_time_scaling"])          

#     return wav
