# Adapted from https://github.com/facebookresearch/demucs under the MIT License 
# Original Copyright (c) Earth Species Project. This work is based on Facebook's denoiser. 

#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import sys
import os 

import torch
import torchmetrics
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio

from .data import NoisyCleanSet
from .enhance import add_flags, get_estimate
from . import distrib, pretrained
from .utils import bold, LogProgress

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
        'denoiser.evaluate',
        description='Denoising using Demucs - Evaluate model performance')
add_flags(parser)
parser.add_argument('--data_dir', help='directory including noisy.json and clean.json files')
parser.add_argument('--matching', default="sort", help='set this to dns for the dns dataset.')
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="More loggging")


def evaluate(args, model=None, data_loader=None, experiment_logger=None, window_size=0, hop_size=0):
    total_sisdri = 0
    total_sisdr = 0
    total_sisdrn = 0
    all_sisdri = []
    all_sisdr = []
    all_sisdrn = []
    total_cnt = 0
    updates = 5

    # Load model
    if not model:
        model = pretrained.get_model(args).to(args.device)
    model.eval()
    
    if window_size > 0:
        import asteroid
        ola_model = asteroid.dsp.overlap_add.LambdaOverlapAdd(
            nnet=model,  # function to apply to each segment.
            n_src=1,  # number of sources in the output of nnet
            window_size=window_size,  # Size of segmenting window
            hop_size=hop_size,  # segmentation hop size
            window="hann",  # Type of the window (see scipy.signal.get_window)
            reorder_chunks=False,  # Whether to reorder each consecutive segment.
            enable_grad=False,  # Set gradient calculation on of off (see torch.set_grad_enabled)
        )
        ola_model.window = ola_model.window.to(args.device)
    else:
        ola_model = None

    # Load data
    if data_loader is None:
        dataset = NoisyCleanSet(args.data_dir,
                                matching=args.matching, sample_rate=args.sample_rate, with_path=True)
        data_loader = distrib.loader(dataset, batch_size=1, num_workers=2)
    pendings = []
    with ProcessPoolExecutor(args.num_workers) as pool:
        with torch.no_grad():
            iterator = LogProgress(logger, data_loader, name="Eval estimates")
            for i, data in enumerate(iterator):
                # Get batch data
                tnoisy, tclean = [x for x in data]
                if len(tnoisy) > 1 and isinstance(tnoisy[1][0], str):
                    noisy = tnoisy[0].to(args.device)
                    clean = tclean[0].to(args.device)
                    filename = str(os.path.basename(tnoisy[1][0]).rsplit(".", 1)[0])
                else:
                    noisy = tnoisy.to(args.device)
                    clean = tclean.to(args.device)
                    filename = str(i)
                #### If device is CPU, we do parallel evaluation in each CPU worker.
                if args.device == 'cpu':
                    pendings.append(
                        pool.submit(_estimate_and_run_metrics, clean, model, noisy, args, filename, experiment_logger))
                else:
                    if ola_model is not None and noisy.shape[-1] > (2*window_size):
                        estimate = get_estimate(ola_model, noisy, args)
                    else:
                        estimate = get_estimate(model, noisy, args)
                    sisdri_i, sisdr_i, sisdrn_i = _run_metrics(clean, estimate, noisy, args, sr=args.sample_rate, filename=filename, experiment_logger=experiment_logger)
                    total_sisdri += sisdri_i
                    total_sisdr += sisdr_i
                    total_sisdrn += sisdrn_i
                    all_sisdri.append(sisdri_i)
                    all_sisdr.append(sisdr_i)
                    all_sisdrn.append(sisdrn_i)
                total_cnt += clean.shape[0]

        if args.device == 'cpu':
            for pending in LogProgress(logger, pendings, updates, name="Eval metrics"):
                sisdri_i, sisdr_i, sisdrn_i = pending.result()
                total_sisdri += sisdri_i
                total_sisdr += sisdr_i
                total_sisdrn += sisdrn_i
                all_sisdri.append(sisdri_i)
                all_sisdr.append(sisdr_i)
                all_sisdrn.append(sisdrn_i)


    metrics = [total_sisdri, total_sisdr, total_sisdrn]
    sisdri_mean, sisdr_mean, sisdrn_mean = distrib.average([m/total_cnt for m in metrics], total_cnt)
    sisdri_median = torch.median(torch.tensor(all_sisdri)).item()
    sisdr_median = torch.median(torch.tensor(all_sisdr)).item()
    sisdrn_median = torch.median(torch.tensor(all_sisdrn)).item()
    logger.info(bold(f'SISDR performance: sisdri_mean={round(sisdri_mean,2)}, sisdr_mean={round(sisdr_mean,2)}, sisdri_median={round(sisdri_median,2)}, sisdr_median={round(sisdr_median,2)}'))
    return sisdri_mean, sisdr_mean, sisdri_median, sisdr_median, sisdrn_mean, sisdrn_median


def _estimate_and_run_metrics(clean, model, noisy, args, filename, experiment_logger=None):
    estimate = get_estimate(model, noisy, args)
    return _run_metrics(clean, estimate, noisy, args, sr=args.sample_rate, filename=filename, experiment_logger=experiment_logger)


def _run_metrics(clean, estimate, noisy, args, sr, filename, experiment_logger=None):
    sisdr_noisy = scale_invariant_signal_distortion_ratio(noisy, clean)
    sisdr = scale_invariant_signal_distortion_ratio(estimate[:,0:1,:], clean)
    sisdri = sisdr - sisdr_noisy
    metadata = {'sisdr': sisdr.mean().item(), 'sisdr_noisy': sisdr_noisy.mean().item(), 'sisdri': sisdri.mean().item()}
    if estimate.shape[1] > 1:
        sisdr_noise = scale_invariant_signal_distortion_ratio(estimate[:,1:2,:], noisy-clean)
        metadata.update({'sisdr_noise': sisdr_noise.mean().item()})
        if experiment_logger is not None:
            experiment_logger.log_audio(estimate[:,1:2,:].squeeze().detach().cpu().numpy(),
                sample_rate=sr,
                file_name=filename + "_noise_est.wav",
                metadata=metadata, overwrite=False,
                step=experiment_logger.step)
    else:
        sisdr_noise = torch.zeros_like(sisdr)
    if experiment_logger is not None:
        experiment_logger.log_audio(noisy.squeeze().detach().cpu().numpy(),
            sample_rate=sr,
            file_name=filename + "_noisy.wav",
            metadata=metadata, overwrite=False,
            step=experiment_logger.step)
        experiment_logger.log_audio(estimate[:,0,:].squeeze().detach().cpu().numpy(),
            sample_rate=sr,
            file_name=filename + "_enhanced.wav",
            metadata=metadata, overwrite=False,
            step=experiment_logger.step)
        experiment_logger.log_audio(clean.squeeze().detach().cpu().numpy(),
            sample_rate=sr,
            file_name=filename + "_clean.wav",
            metadata=metadata, overwrite=False,
            step=experiment_logger.step)
    return sisdri.mean().item(), sisdr.mean().item(), sisdr_noise.mean().item()


def main():
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    sisdri, sisdr = evaluate(args)
    json.dump({'sisdri': sisdri, 'sisdr': sisdr}, sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
