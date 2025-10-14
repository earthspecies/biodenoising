import argparse
import biodenoising
from biodenoising.adapt import ConfigParser  # Import ConfigParser from the package
import logging
import os
import sys
import yaml
import pandas as pd
import soundfile as sf
import torchaudio
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# No need to define ConfigParser here anymore, as we're importing it

# Use the imported ConfigParser
parser = ConfigParser()
parser.add_argument("--steps", default=5, type=int, help="Number of steps to use for adaptation")
parser.add_argument("--noisy_dir", type=str, default=None,
                    help="path to the directory with noisy wav files")
parser.add_argument("--noise_dir", type=str, default=None,
                    help="path to the directory with noise wav files")
parser.add_argument("--test_dir", type=str, default=None,
                    help="for evaluation purpose only: path to the directory containing clean.json and noise.json files")
parser.add_argument("--out_dir", type=str, default="enhanced",
                    help="directory putting enhanced wav files")
parser.add_argument('--noisy_estimate', action="store_true",help="compute the noise as the difference between the noisy and the estimated signal")
parser.add_argument("--cfg", type=str, default="biodenoising/conf/config_adapt.yaml",
                    help="path to the directory with noise wav files")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs per step")
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="more loggging")
parser.add_argument("--method",choices=["biodenoising16k_dns48"], default="biodenoising16k_dns48",help="Method to use for denoising")
parser.add_argument("--segment", default=4, type=int, help="minimum segment size in seconds")
parser.add_argument("--highpass", default=20, type=int, help="apply a highpass filter with this cutoff before separating")
parser.add_argument("--peak_height", default=0.008, type=float, help="filter segments with rms lower than this value")
parser.add_argument("--transform",choices=["none", "time_scale"], default="none",help="Transform input by pitch shifting or time scaling")
parser.add_argument('--revecho', type=float, default=0,help='revecho probability')
parser.add_argument("--use_top", default=1., type=float, help="use the top ratio of files for training, sorted by their rms")
parser.add_argument('--num_valid', type=float, default=0,help='the number of files to use for validation')
parser.add_argument('--antialiasing', action="store_true",help="use an antialiasing filter when time scaling back")
parser.add_argument('--keep_original_sr', action="store_true",help="keep the original sample rate of the audio rather than the model sample rate")
parser.add_argument("--force_sample_rate", default=0, type=int, help="Force the model to take samples of this sample rate")
parser.add_argument("--time_scale_factor", default=0, type=int, help="If the model has a different sample rate, play the audio slower or faster with this factor. If force_sample_rate this automatically changes.")
parser.add_argument('--noise_reduce', action="store_true",help="use noisereduce preprocessing")
parser.add_argument('--amp_scale', action="store_true",help="scale to the amplitude of the input")
parser.add_argument('--interactive', action="store_true",help="pause at each step to allow the user to delete some files and continue")
parser.add_argument("--window_size", type=int, default=0,
                    help="size of the window for continuous processing")
parser.add_argument('--selection_table', action="store_true", help="Enable event masking via selection tables (csv/tsv/txt) located next to audio files.")
parser.add_argument('--device', default="cuda")
parser.add_argument('--dry', type=float, default=0.1,
                    help='dry/wet knob coefficient. 0 is only denoised, 1 only input signal.')
parser.add_argument('--num_workers', type=int, default=5)
parser.add_argument('--annotations', action="store_true", default=False, 
                    help="Use annotation files to extract segments from audio files")
parser.add_argument('--annotations_begin_column', type=str, default="Begin", 
                    help="Column name for segment start time in annotation files")
parser.add_argument('--annotations_end_column', type=str, default="End", 
                    help="Column name for segment end time in annotation files")
parser.add_argument('--annotations_label_column', type=str, default=None, 
                    help="Column name for segment label in annotation files")
parser.add_argument('--annotations_label_value', type=str, default=None, 
                    help="Filter annotations by this label value")
parser.add_argument('--annotations_extension', type=str, default=".csv", 
                    help="Extension of annotation files")
parser.add_argument('--processed_dir', type=str, default=None, 
                    help="Directory for storing preprocessed audio segments")

def run_adaptation_main(args):
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    
    # Call the refactored adaptation function from the module
    model = biodenoising.adapt.run_adaptation(args)
    
    return model

def main() -> None:
    args = parser.parse()
    if args.method == 'biodenoising16k_dns48':
        args.biodenoising16k_dns48 = True
    run_adaptation_main(args)


if __name__ == "__main__":
    main()