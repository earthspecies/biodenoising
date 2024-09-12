#!/usr/bin/env python3
# Adapted from https://github.com/facebookresearch/demucs under the MIT License 
# Original Copyright (c) Earth Species Project. This work is based on Facebook's denoiser. 

#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import hydra
import random
import numpy as np
import biodenoising

logger = logging.getLogger(__name__)

def run(args):
    experiment_logger = None
    if "cometml" in args:
        import comet_ml
        os.environ["COMET_API_KEY"] = args.cometml['api-key']
        experiment_logger = comet_ml.Experiment(args.cometml['api-key'], project_name=args.cometml['project'], log_code=False)
        experiment_logger.log_parameters(args)
        experiment_name = os.path.basename(os.getcwd())
        experiment_logger.set_name(experiment_name)
        
    import torch
    
    biodenoising.denoiser.distrib.init(args)

    ### set the random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True) 
    rng = random.Random(args.seed)
    rngnp = np.random.default_rng(seed=args.seed)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    rngth = torch.Generator(device=args.device)
    rngth.manual_seed(args.seed)


    if args.sample_rate == 48000:
        args.demucs.resample = 8
    if args.model=="demucs":
        if 'chout' in args.demucs:
            args.demucs['chout'] = args.demucs['chout']*args.nsources
        model = biodenoising.denoiser.demucs.Demucs(**args.demucs, sample_rate=args.sample_rate)
        if args.teacher_student:
            model_teacher = biodenoising.denoiser.demucs.Demucs(**args.demucs, sample_rate=args.sample_rate).to(torch.device("cpu"))
    elif args.model=="cleanunet":
        model = biodenoising.denoiser.cleanunet.CleanUNet(**args.cleanunet)
        if args.teacher_student:
            model_teacher = biodenoising.denoiser.cleanunet.CleanUNet(**args.cleanunet).to(torch.device("cpu"))
    
    if args.show:
        logger.info(model)
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2**20
        logger.info('Size: %.1f MB', mb)
        if hasattr(model, 'valid_length'):
            field = model.valid_length(1)
            logger.info('Field: %.1f ms', field / args.sample_rate * 1000)
        return

    assert args.batch_size % biodenoising.denoiser.distrib.world_size == 0
    args.batch_size //= biodenoising.denoiser.distrib.world_size
    length = int(args.segment * args.sample_rate)
    stride = int(args.stride * args.sample_rate)
    # Demucs requires a specific number of samples to avoid 0 padding during training
    if hasattr(model, 'valid_length'):
        length = model.valid_length(length)
    kwargs_valid = {"sample_rate": args.sample_rate,"seed": args.seed,"nsources": args.nsources,"exclude": args.exclude,"exclude_noise": args.exclude_noise, "rng":rng, "rngnp":rngnp, "rngth":rngth }
    kwargs_train = {"sample_rate": args.sample_rate,"seed": args.seed,"nsources": args.nsources,"exclude": args.exclude,"exclude_noise": args.exclude_noise, "rng":rng, "rngnp":rngnp, "rngth":rngth,
                    'repeat_prob': args.repeat_prob, 'random_repeat': args.random_repeat, 'random_pad': args.random_pad, 'silence_prob': args.silence_prob, 'noise_prob': args.noise_prob,
                    'normalize':args.normalize, 'random_gain':args.random_gain, 'low_gain':args.low_gain, 'high_gain':args.high_gain}
    if 'seed=' in args.dset.train:
        args.dset.train = args.dset.train.replace('seed=', f'seed={args.seed}')
    if args.continue_from and 'seed=' in args.continue_from:
        args.continue_from = args.continue_from.replace('seed=', f'seed={args.seed}')
    if args.continue_pretrained and 'seed=' in args.continue_pretrained:
        args.continue_pretrained = args.continue_pretrained.replace('seed=', f'seed={args.seed}')
    # Building datasets and loaders
    tr_dataset = biodenoising.datasets.NoiseClean1WeightedSet(
        args.dset.train, length=length, stride=stride, pad=args.pad, epoch_size=args.epoch_size,
        low_snr=args.dset.low_snr,high_snr=args.dset.high_snr,**kwargs_train)
    tr_loader = biodenoising.denoiser.distrib.loader(
        tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
    if args.dset.valid:
        # cv_dataset = biodenoising.denoiser.data.NoisyCleanSet(args.dset.valid, **kwargs)
        # cv_loader = biodenoising.denoiser.distrib.loader(cv_dataset, batch_size=1, num_workers=args.num_workers)
        cv_dataset = biodenoising.datasets.NoiseCleanValidSet(
            args.dset.valid, length=length, stride=0, pad=False, epoch_size=args.epoch_size,
            low_snr=args.dset.low_snr,high_snr=args.dset.high_snr,**kwargs_valid)
        cv_loader = biodenoising.denoiser.distrib.loader(
            cv_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers//4)
    else:
        cv_loader = None
    if args.dset.test:
        del kwargs_valid["exclude"]
        del kwargs_valid["exclude_noise"]
        del kwargs_valid["rng"]
        del kwargs_valid["rngnp"]
        del kwargs_valid["rngth"]
        if isinstance(args.dset.test, str):
            args.dset.test = {'biodenoising':args.dset.test}
        tt_dataset = {}
        tt_loader = {}
        for key, value in args.dset.test.items():
            tt_dataset[key] = biodenoising.denoiser.data.NoisyCleanSet(value, stride=0, pad=False, with_path=True, **kwargs_valid)
            tt_loader[key] = biodenoising.denoiser.distrib.loader(tt_dataset[key], batch_size=1, shuffle=False, num_workers=args.num_workers//4)
    else:
        tt_loader = None
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}
    
    if args.continue_pretrained:
        args.epochs = np.maximum(1, np.ceil(args.full_size / len(tr_loader.dataset)))
    else:
        args.epochs = np.maximum(1, np.ceil(args.full_size / len(tr_loader.dataset)))
    print("Train size", len(tr_loader.dataset))
    # args.lr = args.lr * args.batch_size / 16
    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    if args.optim == "adam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        total_steps = int(args.epochs * len(tr_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)#, cycle_momentum=False
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.optim == "lion":
        import lion_pytorch
        optimizer = lion_pytorch.Lion(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        total_steps = int(args.epochs * len(tr_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)#, cycle_momentum=False
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)
    
    # Construct Solver
    if args.teacher_student:
        solver = biodenoising.denoiser.solver.TeacherStudentSolver(data, model, model_teacher, optimizer, args, rng=rng, rngnp=rngnp, rngth=rngth, seed=args.seed, experiment_logger=experiment_logger, scheduler=scheduler)
    else:
        solver = biodenoising.denoiser.solver.Solver(data, model, optimizer, args, rng=rng, rngnp=rngnp, rngth=rngth, seed=args.seed, experiment_logger=experiment_logger, scheduler=scheduler)
    solver.train()


def _main(args):
    global __file__
    # Updating paths in config
    for key, value in args.dset.items():
        if key=='test':
            ### replace all subkeys 
            for k,v in value.items():
                args.dset.test[k] = hydra.utils.to_absolute_path(v.replace('<<username>>', os.getenv('USER')))     
        elif isinstance(value, str) and key not in ["matching"]:
            args.dset[key] = hydra.utils.to_absolute_path(value)
    args.continue_pretrained = args.continue_pretrained.replace('<<username>>', os.getenv('USER'))
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("denoise").setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    if args.ddp and args.rank is None:
        biodenoising.denoiser.executor.start_ddp_workers(args)
    else:
        run(args)


@hydra.main(config_path="biodenoising/conf/config.yaml")
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)


if __name__ == "__main__":
    main()
