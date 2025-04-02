# Adapted from https://github.com/facebookresearch/demucs under the MIT License 
# Original Copyright (c) Earth Species Project. This work is based on Facebook's denoiser. 

#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import json
import logging
from pathlib import Path
import os
import time
import copy
import numpy as np
import random

import torch
import torch.nn.functional as F

from .loss import singlesrc_neg_sisdr, RMSSmoothLoss
from . import augment, distrib, pretrained
from .enhance import enhance
from .evaluate import evaluate
from .stft_loss import MultiResolutionSTFTLoss
from .utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress, load_model_state_dict, apply_output_transform

logger = logging.getLogger(__name__)


class Solver(object):
    def __init__(self, data, model, optimizer, args, rng=None, rngnp=None, rngth=None, seed=42, experiment_logger=None, scheduler = None):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.tt_loader = data['tt_loader']
        self.model = model
        self.dmodel = distrib.wrap(model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        if args.swa_scheduler:
            self.swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, anneal_epochs=5, swa_lr=args.lr)
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.dmodel)
            self.swa_start = args.swa_start
        else:
            self.swa_scheduler = None
            self.swa_model = None
            self.swa_start = 0
        self.experiment_logger = experiment_logger
        self.nsources = args.nsources
        self.epochs = int(args.epochs)
        self.sample_rate = args.sample_rate
        self.seed = seed
        self.device = args.device
        self.rng = rng if rng is not None else random.Random(args.seed)
        self.rngnp = rngnp if rngnp is not None else np.random.default_rng(seed=args.seed)
        self.rngth = rngth 
        if rngth is not None:
            self.rngth = rngth
        else:
            self.rngth = torch.Generator(device=args.device)
            self.rngth.manual_seed(args.seed)
                
        # data augment
        augments = []
        if args.remix:
            augments.append(augment.Remix(rngth=self.rngth))
        if args.flip:
            augments.append(augment.Flip(args.flip,rngth=self.rngth))
        if args.bandmask:
            augments.append(augment.BandMask(args.bandmask, sample_rate=args.sample_rate,rng=self.rng,seed=args.seed))
        if args.shift:
            augments.append(augment.Shift(args.shift, args.shift_same,rngth=self.rngth))
        if args.timescale:
            augments.append(augment.TimeScale(args.timescale,rngnp=self.rngnp,seed=args.seed))
        if args.revecho:
            augments.append(
                augment.RevEcho(args.revecho,rng=self.rng,seed=args.seed))
        if args.mixup:
            augments.append(augment.Mixup(prob=args.mixup,rngth=self.rngth,seed=args.seed))
        self.augment = torch.nn.Sequential(*augments)

        # Training config
        
        logger.info("Training for %d epochs", self.epochs)  

        # Checkpoints
        self.continue_from = args.continue_from
        self.eval_every = args.eval_every
        self.checkpoint = args.checkpoint
        if self.checkpoint:
            self.checkpoint_file = Path(args.checkpoint_file)
            self.best_file = Path(args.best_file)
            logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.history_file = args.history_file

        self.best_state = None
        self.restart = args.restart
        self.history = []  # Keep track of loss
        self.samples_dir = args.samples_dir  # Where to save samples
        self.num_prints = args.num_prints  # Number of times to log per epoch
        self.args = args
        self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=args.stft_sc_factor,
                                                factor_mag=args.stft_mag_factor,
                                                mask=args.stft_mask, threshold=args.stft_mask_threshold).to(self.device)
        self.rmsloss = RMSSmoothLoss().to(self.device)
        self._reset()

    def _serialize(self, suffix=""):
        package = {}
        package['model'] = serialize_model(self.model)
        package['optimizer'] = self.optimizer.state_dict()
        package['scheduler'] = self.scheduler.state_dict() if self.scheduler else None
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        tmp_path = str(self.checkpoint_file)+ ".tmp"
        torch.save(package, tmp_path)
        # renaming is sort of atomic on UNIX (not really true on NFS)
        # but still less chances of leaving a half written checkpoint behind.
        os.rename(tmp_path, str(self.checkpoint_file).replace('.th', suffix + '.th'))

        # Saving only the latest best model.
        if len(suffix) == 0 and self.best_state is not None:
            model = package['model']
            model['state'] = self.best_state
            tmp_path = str(self.best_file) + ".tmp"
            torch.save(model, tmp_path)
            os.rename(tmp_path, self.best_file)

    def _reset(self):
        """_reset."""
        load_from = None
        load_best = False
        keep_history = True
        # Reset
        if self.checkpoint and self.checkpoint_file.exists() and not self.restart:
            load_from = self.checkpoint_file
        elif self.continue_from:
            load_from = self.continue_from
            load_best = self.args.continue_best
            keep_history = False

        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            package = torch.load(load_from, 'cpu', weights_only=False)
            if load_best:
                self.model.load_state_dict(package['best_state'])
            else:
                self.model.load_state_dict(package['model']['state'])
            if 'optimizer' in package and not load_best:
                self.optimizer.load_state_dict(package['optimizer'])
            if 'scheduler' in package and not load_best:
                self.scheduler.load_state_dict(package['scheduler'])
                
            if keep_history:
                self.history = package['history']
                if self.scheduler is not None:
                    self.scheduler.last_step = len(self.history)  * len(self.tr_loader)
            self.best_state = package['best_state']
        continue_pretrained = self.args.continue_pretrained
        if continue_pretrained:
            logger.info("Fine tuning from pre-trained model %s", continue_pretrained)
            if continue_pretrained.endswith(".th"):
                package = torch.load(continue_pretrained, 'cpu', weights_only=False)
                self.model.load_state_dict(package['model']['state'])
            else:
                if self.args.model=="cleanunet":
                    model = getattr(pretrained, continue_pretrained)(self.args)
                else:
                    model = getattr(pretrained, self.args.continue_pretrained)()
                load_model_state_dict(self.model,model.state_dict())
    
    def train(self):
        if self.args.save_again:
            self._serialize()
            return
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {info}")
            if self.experiment_logger is not None:
                self.experiment_logger.log_metrics(metrics, epoch=epoch+1)
                
        
        # initial evaluation
        if self.tt_loader:
            for key, value in self.tt_loader.items():
                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set %s...', key)
                # We switch to the best known model for testing
                # with swap_state(self.model, self.best_state):
                sisdri_mean, sisdr_mean, sisdri_median, sisdr_median, sisdrn_mean, sisdrn_median = evaluate(self.args, self.model, self.tt_loader[key], self.experiment_logger, self.args.eval_window_size, self.sample_rate)
                if sisdrn_mean!=0:
                    metrics={'test_'+key+'_sisdri_mean': sisdri_mean, 'test_'+key+'_sisdr_mean': sisdr_mean, 'test_'+key+'_sisdrn_mean': sisdr_mean, 'test_'+key+'_sisdri_median': sisdri_median, 'test_'+key+'_sisdr_median': sisdr_median, 'test_'+key+'_sisdrn_median': sisdrn_median}
                else:
                    metrics={'test_'+key+'_sisdri_mean': sisdri_mean, 'test_'+key+'_sisdr_mean': sisdr_mean, 'test_'+key+'_sisdri_median': sisdri_median, 'test_'+key+'_sisdr_median': sisdr_median}
                info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
                logger.info('-' * 70)
                logger.info(bold(f"Initial evaluation | {info}"))
        
        for epoch in range(len(self.history), self.epochs):
            # Train one epoch
            self.model.train()
            start = time.time()
            logger.info('-' * 70)
            logger.info("Training...")
            train_loss = self._run_one_epoch(epoch)
            logger.info(
                bold(f'Train Summary | End of Epoch {epoch + 1} | '
                    f'Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f}'))
            metrics = {'train_loss': train_loss}
            if self.experiment_logger is not None:
                self.experiment_logger.step = epoch+1
            if self.cv_loader:
                # Cross validation
                logger.info('-' * 70)
                logger.info('Cross validation...')
                self.model.eval()
                with torch.no_grad():
                    valid_loss = self._run_one_epoch(epoch, cross_valid=True)
                logger.info(
                    bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                        f'Time {time.time() - start:.2f}s | Valid Loss {valid_loss:.5f}'))
                sisdri_mean, sisdr_mean, sisdri_median, sisdr_median, sisdrn_mean, sisdrn_median = evaluate(self.args, self.model, self.cv_loader)
                if sisdrn_mean!=0:
                    metrics.update({'valid_sisdri_mean': sisdri_mean, 'valid_sisdr_mean': sisdr_mean, 'valid_sisdrn_mean': sisdr_mean, 'valid_sisdri_median': sisdri_median, 'valid_sisdr_median': sisdr_median, 'valid_sisdrn_median': sisdrn_median})
                else:
                    metrics.update({'valid_sisdri_mean': sisdri_mean, 'valid_sisdr_mean': sisdr_mean, 'valid_sisdri_median': sisdri_median, 'valid_sisdr_median': sisdr_median})
            else:
                valid_loss = 0
                sisdr_mean = 1000
            if self.cv_loader:
                best_loss = min(pull_metric(self.history, 'valid_loss') + [valid_loss])
                best_sisdr = max(pull_metric(self.history, 'valid_sisdr_mean') + [sisdr_mean])
                metrics.update({'valid_loss': valid_loss, 'best_loss': best_loss, 'best_sisdr_mean': best_sisdr})
                # Save the best model
                # if valid_loss == best_loss:
                #     logger.info(bold('New best valid loss %.4f'), valid_loss)
                #     self.best_state = copy_state(self.model.state_dict())
                if sisdr_mean == best_sisdr:
                    logger.info(bold('New best sisdr %.4f'), sisdr_mean)
                    self.best_state = copy_state(self.model.state_dict())

            # evaluate and enhance samples every 'eval_every' argument number of epochs
            # also evaluate on last epoch
            if ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1) and self.tt_loader:
                for key, value in self.tt_loader.items():
                    # Evaluate on the testset
                    logger.info('-' * 70)
                    logger.info('Evaluating on the test set %s...', key)
                    # We switch to the best known model for testing
                    # with swap_state(self.model, self.best_state):
                    sisdri_mean, sisdr_mean, sisdri_median, sisdr_median, sisdrn_mean, sisdrn_median = evaluate(self.args, self.model, self.tt_loader[key], self.experiment_logger, self.args.eval_window_size, self.sample_rate)
                    if sisdrn_mean!=0:
                        metrics.update({'test_'+key+'_sisdri_mean': sisdri_mean, 'test_'+key+'_sisdr_mean': sisdr_mean, 'test_'+key+'_sisdrn_mean': sisdr_mean, 'test_'+key+'_sisdri_median': sisdri_median, 'test_'+key+'_sisdr_median': sisdr_median, 'test_'+key+'_sisdrn_median': sisdrn_median})
                    else:
                        metrics.update({'test_'+key+'_sisdri_mean': sisdri_mean, 'test_'+key+'_sisdr_mean': sisdr_mean, 'test_'+key+'_sisdri_median': sisdri_median, 'test_'+key+'_sisdr_median': sisdr_median})

                    # # enhance some samples
                    # logger.info('Enhance and save samples...')
                    # enhance(self.args, self.model, self.samples_dir)

            self.history.append(metrics)
            info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))
            if self.experiment_logger is not None:
                self.experiment_logger.log_metrics(metrics, epoch=epoch+1)
            if distrib.rank == 0:
                json.dump(self.history, open(self.history_file, "w"), indent=2)
                # Save model each epoch
                if self.checkpoint:
                    # if epoch == 0:
                    #     self._serialize(suffix="_init")
                    # elif epoch == self.epochs - 1:
                    #     self._serialize(suffix="_final")
                    self._serialize()
                    logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())
        
        if self.swa_model is not None:
            torch.optim.swa_utils.update_bn(self.tr_loader, self.swa_model)
            self.dmodel = self.swa_model.module

    def _run_one_epoch(self, epoch, cross_valid=False):
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # get a different order for distributed training, otherwise this will get ignored
        data_loader.epoch = epoch

        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)
        for i, data in enumerate(logprog):
            noisy, clean = [x.to(self.device) for x in data]
            
            if not cross_valid:
                sources = torch.stack([noisy - clean, clean])
                sources = self.augment(sources)
                noise, clean = sources
                noisy = noise + clean
                sources = torch.cat([clean, noisy - clean], dim=1)
            if self.nsources>1:
                sources = torch.cat([clean, noisy - clean], dim=1)
            else:
                sources = clean
            
            noisy = torch.nan_to_num(noisy, nan=1e-8, posinf=1, neginf=-1)
            estimate = self.dmodel(noisy)
            estimate = torch.nan_to_num(estimate, nan=1e-8, posinf=1, neginf=-1)
            sources = torch.nan_to_num(sources, nan=1e-8, posinf=1, neginf=-1)
            # apply a loss function after each layer
            with torch.autograd.set_detect_anomaly(True):
                if self.args.loss == 'l1':
                    loss = F.l1_loss(sources, estimate)
                elif self.args.loss == 'l2':
                    loss = F.mse_loss(sources, estimate)
                elif self.args.loss == 'huber':
                    loss = F.smooth_l1_loss(sources, estimate)
                elif self.args.loss == 'sisdr':
                    loss = singlesrc_neg_sisdr(sources, estimate)
                else:
                    raise ValueError(f"Invalid loss {self.args.loss}")
                # MultiResolution STFT loss
                if self.args.stft_loss:
                    sc_loss, mag_loss = self.mrstftloss(estimate[:,0,:], sources[:,0,:])
                    if self.nsources>1 and estimate.shape[1]>1:
                        for s in range(1,estimate.shape[1]):
                            sc_l, mag_l = self.mrstftloss(estimate[:,s,:], sources[:,s,:])
                            sc_loss += sc_l
                            mag_loss += mag_l
                    loss += sc_loss + mag_loss
                if self.args.rms_loss>0:
                    loss += self.args.rms_loss * self.rmsloss(estimate, sources)
                loss = torch.clamp(loss,min=-self.args.clamp_loss, max=self.args.clamp_loss)
                loss = torch.nan_to_num(loss, nan=1e-8, posinf=1, neginf=-1)
                # optimize model in training mode
                if not cross_valid:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.dmodel.parameters(), self.args.clip_grad_norm)
                    self.optimizer.step()
                    if self.swa_scheduler is not None and epoch > self.swa_start:
                        self.swa_model.update_parameters(self.dmodel)
                        self.swa_scheduler.step()
                    elif self.scheduler is not None:
                        self.scheduler.step()
                        
            total_loss += loss.item()
            logprog.update(loss=format(total_loss / (i + 1), ".5f"))
            # Just in case, clear some memory
            del loss, estimate
        return distrib.average([total_loss / (i + 1)], i + 1)[0]


class TeacherStudentSolver(object):
    def __init__(self, data, model, teacher, optimizer, args, experiment_logger=None):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.tt_loader = data['tt_loader']
        self.model = model
        self.teacher = teacher
        self.dmodel = distrib.wrap(model)
        self.optimizer = optimizer
        self.experiment_logger = experiment_logger
        self.nsources = args.nsources
        
        self.student_step = 1
        self.student_order = 1
        self.tr_step = 1
        
        # data augment
        augments = []
        if args.remix:
            augments.append(augment.Remix())
        if args.bandmask:
            augments.append(augment.BandMask(args.bandmask, sample_rate=args.sample_rate))
        if args.shift:
            augments.append(augment.Shift(args.shift, args.shift_same))
        if args.revecho:
            augments.append(
                augment.RevEcho(args.revecho))
        if args.timescale:
            augments.append(augment.TimeScale(args.timescale))
        if args.mixup:
            augments.append(augment.MixUp(args.mixup))
        if args.flip:
            augments.append(augment.Flip())
        
        self.augment = torch.nn.Sequential(*augments)

        # Training config
        self.device = args.device
        self.epochs = args.epochs

        # Checkpoints
        self.continue_from = args.continue_from
        self.eval_every = args.eval_every
        self.checkpoint = args.checkpoint
        if self.checkpoint:
            self.checkpoint_file = Path(args.checkpoint_file)
            self.best_file = Path(args.best_file)
            logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.history_file = args.history_file

        self.best_state = None
        self.restart = args.restart
        self.history = []  # Keep track of loss
        self.samples_dir = args.samples_dir  # Where to save samples
        self.num_prints = args.num_prints  # Number of times to log per epoch
        self.args = args
        self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=args.stft_sc_factor,
                                                factor_mag=args.stft_mag_factor).to(self.device)
        self._reset()

    def _serialize(self):
        package = {}
        package['model'] = serialize_model(self.model)
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        tmp_path = str(self.checkpoint_file) + ".tmp"
        torch.save(package, tmp_path)
        # renaming is sort of atomic on UNIX (not really true on NFS)
        # but still less chances of leaving a half written checkpoint behind.
        os.rename(tmp_path, self.checkpoint_file)

        # Saving only the latest best model.
        model = package['model']
        model['state'] = self.best_state
        tmp_path = str(self.best_file) + ".tmp"
        torch.save(model, tmp_path)
        os.rename(tmp_path, self.best_file)

    def _reset(self):
        """_reset."""
        load_from = None
        load_best = False
        keep_history = True
        # Reset
        if self.checkpoint and self.checkpoint_file.exists() and not self.restart:
            load_from = self.checkpoint_file
        elif self.continue_from:
            load_from = self.continue_from
            load_best = self.args.continue_best
            keep_history = False
        
        if self.args.continue_pretrained:
            logger.info("Teacher model from pre-trained model %s", self.args.continue_pretrained)
            model = getattr(pretrained, self.args.continue_pretrained)()
            load_model_state_dict(self.teacher,model.state_dict())    
        elif load_from:
            logger.info(f'Loading teacher from checkpoint model: {load_from}')
            package = torch.load(load_from, 'cpu')
            if load_best:
                self.teacher.load_state_dict(package['best_state'])
            else:
                self.teacher.load_state_dict(package['model']['state'])
            if 'optimizer' in package and not load_best:
                self.optimizer.load_state_dict(package['optimizer'])
            if keep_history:
                self.history = package['history']
            self.best_state = package['best_state']
        else:
            raise ValueError("No teacher model to load")
    
    def train(self):
        if self.args.save_again:
            self._serialize()
            return
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {info}")
            if self.experiment_logger is not None:
                self.experiment_logger.log_metrics(metrics, epoch=epoch+1)

        for epoch in range(len(self.history), self.epochs):
            if epoch>len(self.history):
                self.setup_models()
            
            # Train one epoch
            self.model.train()
            self.teacher.eval()
            
            start = time.time()
            logger.info('-' * 70)
            logger.info("Training...")
            train_loss = self._run_one_epoch(epoch)
            logger.info(
                bold(f'Train Summary | End of Epoch {epoch + 1} | '
                    f'Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f}'))
            metrics = {'train_loss': train_loss}
            if self.experiment_logger is not None:
                self.experiment_logger.step = epoch+1
            if self.cv_loader:
                # Cross validation
                logger.info('-' * 70)
                logger.info('Cross validation...')
                self.model.eval()
                with torch.no_grad():
                    valid_loss = self._run_one_epoch(epoch, cross_valid=True)
                logger.info(
                    bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                        f'Time {time.time() - start:.2f}s | Valid Loss {valid_loss:.5f}'))
                sisdri_mean, sisdr_mean, sisdri_median, sisdr_median, sisdrn_mean, sisdrn_median = evaluate(self.args, self.model, self.cv_loader)
                if sisdrn_mean!=0:
                    metrics.update({'valid_sisdri_mean': sisdri_mean, 'valid_sisdr_mean': sisdr_mean, 'valid_sisdrn_mean': sisdr_mean, 'valid_sisdri_median': sisdri_median, 'valid_sisdr_median': sisdr_median, 'valid_sisdrn_median': sisdrn_median})
                else:
                    metrics.update({'valid_sisdri_mean': sisdri_mean, 'valid_sisdr_mean': sisdr_mean, 'valid_sisdri_median': sisdri_median, 'valid_sisdr_median': sisdr_median})
            else:
                valid_loss = 0

            best_loss = min(pull_metric(self.history, 'valid_loss') + [valid_loss])
            metrics.update({'valid_loss': valid_loss, 'best_loss': best_loss})
            # Save the best model
            if valid_loss == best_loss:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_state = copy_state(self.model.state_dict())

            # evaluate and enhance samples every 'eval_every' argument number of epochs
            # also evaluate on last epoch
            if ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1) and self.tt_loader:
                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                # We switch to the best known model for testing
                # with swap_state(self.model, self.best_state):
                sisdri_mean, sisdr_mean, sisdri_median, sisdr_median, sisdrn_mean, sisdrn_median = evaluate(self.args, self.model, self.tt_loader, self.experiment_logger)
                if sisdrn_mean!=0:
                    metrics.update({'test_sisdri_mean': sisdri_mean, 'test_sisdr_mean': sisdr_mean, 'test_sisdrn_mean': sisdr_mean, 'test_sisdri_median': sisdri_median, 'test_sisdr_median': sisdr_median, 'test_sisdrn_median': sisdrn_median})
                else:
                    metrics.update({'test_sisdri_mean': sisdri_mean, 'test_sisdr_mean': sisdr_mean, 'test_sisdri_median': sisdri_median, 'test_sisdr_median': sisdr_median})

                # # enhance some samples
                # logger.info('Enhance and save samples...')
                # enhance(self.args, self.model, self.samples_dir)

            self.history.append(metrics)
            info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))
            if self.experiment_logger is not None:
                self.experiment_logger.log_metrics(metrics, epoch=epoch+1)
            if distrib.rank == 0:
                json.dump(self.history, open(self.history_file, "w"), indent=2)
                # Save model each epoch
                if self.checkpoint:
                    self._serialize()
                    logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())
            self.student_step += 1
            self.tr_step += 1

    def _run_one_epoch(self, epoch, cross_valid=False):
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # get a different order for distributed training, otherwise this will get ignored
        data_loader.epoch = epoch

        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)
        for i, data in enumerate(logprog):
            noisy, clean = [x.to(self.device) for x in data]
            if not cross_valid:
                # Teacher-student training
                with torch.no_grad():
                    # Teacher's estimates
                    teacher_estimates = self.teacher(noisy.cpu()).detach()
                    # teacher_estimates = apply_output_transform(
                    #     teacher_estimates, noisy.std(), noisy.mean(), noisy, self.args)
                    if teacher_estimates.shape[1] == 1:
                        cleaned = teacher_estimates.cuda()
                    else:
                        cleaned, noise_est = teacher_estimates[:, 0:1].cuda(), teacher_estimates[:, 1:].cuda()
                    del teacher_estimates
                    if self.args.other_noise:
                        noise = noisy - clean
                    else:
                        noise = noise_est if noise_est is not None else noisy - cleaned
                sources = torch.stack([noise, cleaned])        
                sources = self.augment(sources)
                noise, clean = sources
                noisy = noise + clean
            if self.nsources>1:
                sources = torch.cat([clean, noisy - clean], dim=1)
            else:
                sources = clean
            
            noisy = torch.nan_to_num(noisy, nan=1e-8, posinf=1, neginf=-1)
            estimate = self.dmodel(noisy)
            estimate = torch.nan_to_num(estimate, nan=1e-8, posinf=1, neginf=-1)
            sources = torch.nan_to_num(sources, nan=1e-8, posinf=1, neginf=-1)
            # apply a loss function after each layer
            with torch.autograd.set_detect_anomaly(True):
                if self.args.loss == 'l1':
                    loss = F.l1_loss(sources, estimate)
                elif self.args.loss == 'l2':
                    loss = F.mse_loss(sources, estimate)
                elif self.args.loss == 'huber':
                    loss = F.smooth_l1_loss(sources, estimate)
                else:
                    raise ValueError(f"Invalid loss {self.args.loss}")
                # MultiResolution STFT loss
                if self.args.stft_loss:
                    sc_loss, mag_loss = self.mrstftloss(estimate[:,0,:], sources[:,0,:])
                    if self.nsources>1 and estimate.shape[1]>1:
                        for s in range(1,estimate.shape[1]):
                            sc_l, mag_l = self.mrstftloss(estimate[:,s,:], sources[:,s,:])
                            sc_loss += sc_l
                            mag_loss += mag_l
                    loss += sc_loss + mag_loss
                loss = torch.clamp(loss,min=-30., max=+30.)
                loss = torch.nan_to_num(loss, nan=1e-8, posinf=1, neginf=-1)
                # optimize model in training mode
                if not cross_valid:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.dmodel.parameters(), 10)
                    self.optimizer.step()
            
            total_loss += loss.item()
            logprog.update(loss=format(total_loss / (i + 1), ".5f"))
            # Just in case, clear some memory
            del loss, estimate, sources, noisy, clean
            
        return distrib.average([total_loss / (i + 1)], i + 1)[0]


    def setup_models(self):
            # Figure out which student order is and replace teacher if needed
            if self.args.n_epochs_teacher_update>0:
                update_needed = self.tr_step // self.args.n_epochs_teacher_update + 1 > self.student_order
                # if update_needed and self.args.student_depth_growth > 1.:
                #     # Sequential teacher update protocol.
                #     # Replace old teacher with the newest student and update order
                #     del self.teacher
                #     new_teacher_w = copy.deepcopy(self.model.state_dict())
                #     del self.model
                #     self.teacher.load_state_dict(new_teacher_w)
                #     del new_teacher_w
                #     old_student_depth = self.args.student_depth_growth ** (self.student_order - 1)
                #     new_student_growth = self.args.student_depth_growth ** self.student_order
                #     self.student = self.get_new_student(depth_growth=new_student_growth)
                #     logger.info(f"Replaced old teacher with latest student: {old_student_depth} -> {new_student_growth}")
                #     self.student_step = 1
                #     self.student_order = self.tr_step // self.args.n_epochs_teacher_update
                # elif update_needed and self.args.teacher_momentum > 0.:
                if update_needed and self.args.teacher_momentum > 0.:
                    # Exponential moving average protocol.
                    t_momentum = self.args.teacher_momentum
                    new_teacher_w = copy.deepcopy(self.teacher.state_dict())
                    student_w = self.model.state_dict()
                    for key in new_teacher_w.keys():
                        new_teacher_w[key] = (
                                t_momentum * new_teacher_w[key] + (1.0 - t_momentum) * student_w[key].cpu())
                    self.teacher.load_state_dict(new_teacher_w)
                    del new_teacher_w
                    self.teacher.eval()
                    logger.info(f"Updated the teacher with EMA in the {self.student_order}-th student order.")
                    self.student_step = 1
                    self.student_order = self.tr_step // self.args.n_epochs_teacher_update
                    
