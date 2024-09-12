import torch
import numpy as np


def noise_psd(N, psd = lambda f: 1):
        X_white = np.fft.rfft(np.random.randn(N))
        S = psd(np.fft.rfftfreq(N))
        # Normalize S
        S = S / np.sqrt(np.mean(S**2))
        X_shaped = X_white * S
        return np.fft.irfft(X_shaped)

def PSDGenerator(f):
    return lambda N: noise_psd(N, f)

@PSDGenerator
def white_noise(f):
    return 1

@PSDGenerator
def blue_noise(f):
    return np.sqrt(f)

@PSDGenerator
def violet_noise(f):
    return f

@PSDGenerator
def brownian_noise(f):
    return 1/np.where(f == 0, float('inf'), f)

@PSDGenerator
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))


def get_noise(N,rngnp):
    noises = {0: white_noise, 1: blue_noise, 2: violet_noise, 3: brownian_noise, 4: pink_noise}
    noise_type = rngnp.integers(low=0,high=len(noises))
    noise = noises[noise_type](N)
    # nframes_offset = int(rngnp.uniform() * N//2)
    # noise = noise[nframes_offset:nframes_offset+N]
    return noise