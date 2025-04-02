# Adapted from https://github.com/facebookresearch/demucs under the MIT License 
# Original Copyright (c) Earth Species Project. This work is based on Facebook's denoiser. 

#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import os
import logging

import torch.hub

from .cleanunet import CleanUNet
from .demucs import Demucs
from .htdemucs import HTDemucs
from .utils import deserialize_model, load_model_state_dict
from .states import set_state

logger = logging.getLogger(__name__)
ROOT = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/"
DNS_48_URL = ROOT + "dns48-11decc9d8e3f0998.th"
DNS_64_URL = ROOT + "dns64-a7761ff99a7d5bb6.th"
MASTER_64_URL = ROOT + "master64-8a5dfb4bb92753dd.th"
VALENTINI_NC = ROOT + 'valentini_nc-93fc4337.th'  # Non causal Demucs on Valentini
### list of all music demucs models https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/remote/files.txt
DEMUCSV4_URL =  "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th"
CLEANUNET_URL = "https://github.com/NVIDIA/CleanUNet/raw/main/exp/DNS-large-full/checkpoint/pretrained.pkl"

BIO_ROOT = "https://storage.googleapis.com/esp-public-files/biodenoising/"
BIO_DNS_48_URL = BIO_ROOT + "model-16kHz-dns48.th"

def _demucs(pretrained, url, **kwargs):
    if '/demucs/' not in url:
        model = Demucs(**kwargs, sample_rate=16_000)
    if pretrained:
        #import pdb; pdb.set_trace()
        if '/demucs/' in url:
            full_model = torch.hub.load_state_dict_from_url(url, map_location='cpu')
            args = full_model["args"]
            kwargs = dict(kwargs, **full_model["kwargs"])
            model = HTDemucs(*args, **kwargs)
            state_dict = full_model['state']
            state_dict = set_state(model, state_dict)
        else:
            state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']['state']
            model = load_model_state_dict(model, state_dict)
    return model

def biodenoising16k_dns48(pretrained=True):
    return _demucs(pretrained, BIO_DNS_48_URL, hidden=48)

def dns48(pretrained=True):
    return _demucs(pretrained, DNS_48_URL, hidden=48)


def dns64(pretrained=True):
    return _demucs(pretrained, DNS_64_URL, hidden=64)


def master64(pretrained=True):
    return _demucs(pretrained, MASTER_64_URL, hidden=64)


def valentini_nc(pretrained=True):
    return _demucs(pretrained, VALENTINI_NC, hidden=64, causal=False, stride=2, resample=2)

def demucsv4(pretrained=True):
    return _demucs(pretrained, DEMUCSV4_URL)

def cleanunet_speech(args):
    if 'cleanunet' not in args:
        args.cleanunet = {
            "channels_input": 1,
            "channels_output": 1,
            "channels_H": 64,
            "max_H": 768,
            "encoder_n_layers": 8,
            "kernel_size": 4,
            "stride": 2,
            "tsfm_n_layers": 5, 
            "tsfm_n_head": 8,
            "tsfm_d_model": 512, 
            "tsfm_d_inner": 2048
        }
    model = CleanUNet(**args.cleanunet)
    checkpoint = torch.hub.load_state_dict_from_url(CLEANUNET_URL, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model



def add_model_flags(parser):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-m", "--model_path", help="Path to local trained model.")
    group.add_argument("--biodenoising16k_dns48", action="store_true",
                       help="Use the biodenoising 16kHz pre-trained real time H=48 model.")
    group.add_argument("--dns48", action="store_true",
                       help="Use pre-trained real time H=48 model trained on DNS.")
    group.add_argument("--dns64", action="store_true",
                       help="Use pre-trained real time H=64 model trained on DNS.")
    group.add_argument("--master64", action="store_true",
                       help="Use pre-trained real time H=64 model trained on DNS and Valentini.")
    group.add_argument("--valentini_nc", action="store_true",
                       help="Use pre-trained H=64 model trained on Valentini, non causal.")
    group.add_argument("--demucsv4", action="store_true",
                       help="Use pre-trained music hybrid-transformer demucs.")
    group.add_argument("--cleanunet_speech", action="store_true",
                       help="Use pre-trained cleanunet model trained on DNS.")


def get_model(args):
    """
    Load local model package or torchhub pre-trained model.
    """
    if args.model_path:
        logger.info("Loading model from %s", args.model_path)
        pkg = torch.load(args.model_path, 'cpu', weights_only=False)
        if 'model' in pkg:
            # if 'best_state' in pkg:
            #     logger.info("Loading best model state.")
            #     pkg['model']['state'] = pkg['best_state']
            model = deserialize_model(pkg['model'])
        else:
            
            model = deserialize_model(pkg)
    elif args.biodenoising16k_dns48:
        logger.info("Loading the biodenoising 16kHz pre-trained real time H=64 model.")
        model = biodenoising16k_dns48()
    elif args.dns64:
        logger.info("Loading pre-trained real time H=64 model trained on DNS.")
        model = dns64()
    elif args.master64:
        logger.info("Loading pre-trained real time H=64 model trained on DNS and Valentini.")
        model = master64()
    elif args.demucsv4:
        logger.info("Loading pre-trained music hybrid-transformer demucs (v4).")
        model = demucsv4()
    elif args.valentini_nc:
        logger.info("Loading pre-trained H=64 model trained on Valentini.")
        model = valentini_nc()
    elif args.cleanunet_speech:
        logger.info("Loading pre-trained cleanunet model trained on DNS.")
        model = cleanunet_speech(args)
    else:
        logger.info("Loading pre-trained real time H=48 model trained on DNS.")
        model = dns48()
    logger.debug(model)
    return model
