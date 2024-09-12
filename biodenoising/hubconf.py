# Adapted from https://github.com/facebookresearch/demucs under the MIT License 
# Original Copyright (c) Earth Species Project. This work is based on Facebook's denoiser. 

#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

dependencies = ['torch']
from denoiser.pretrained import dns48, dns64, master64, demucsv4  # noqa
