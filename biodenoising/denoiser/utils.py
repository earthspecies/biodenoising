# Adapted from https://github.com/facebookresearch/demucs under the MIT License 
# Original Copyright (c) Earth Species Project. This work is based on Facebook's denoiser. 

#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import functools
import logging
from contextlib import contextmanager
import inspect
import time
import sys

import torch
logger = logging.getLogger(__name__)


def capture_init(init):
    """capture_init.

    Decorate `__init__` with this, and you can then
    recover the *args and **kwargs passed to it in `self._init_args_kwargs`
    """
    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__


def deserialize_model(package, strict=False):
    """deserialize_model.

    """
    klass = package['class']
    kwargs = package['kwargs']
    if 'sample_rate' not in kwargs:
        logger.warning(
            "Training sample rate not available!, 16kHz will be assumed. "
            "If you used a different sample rate at train time, please fix your checkpoint "
            "with the command `./train.py [TRAINING_ARGS] save_again=true.")
    if strict:
        model = klass(*package['args'], **kwargs)
    else:
        sig = inspect.signature(klass)
        kw = package['kwargs']
        for key in list(kw):
            if key not in sig.parameters:
                logger.warning("Dropping inexistant parameter %s", key)
                del kw[key]
        model = klass(*package['args'], **kw)
    model.load_state_dict(package['state'])
    return model


def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


def serialize_model(model):
    args, kwargs = model._init_args_kwargs
    state = copy_state(model.state_dict())
    return {"class": model.__class__, "args": args, "kwargs": kwargs, "state": state}


@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


def pull_metric(history, name):
    out = []
    for metrics in history:
        if name in metrics:
            out.append(metrics[name])
    return out


class LogProgress:
    """
    Sort of like tqdm but using log lines and not as real time.
    Args:
        - logger: logger obtained from `logging.getLogger`,
        - iterable: iterable object to wrap
        - updates (int): number of lines that will be printed, e.g.
            if `updates=5`, log every 1/5th of the total length.
        - total (int): length of the iterable, in case it does not support
            `len`.
        - name (str): prefix to use in the log.
        - level: logging level (like `logging.INFO`).
    """
    def __init__(self,
                 logger,
                 iterable,
                 updates=5,
                 total=None,
                 name="LogProgress",
                 level=logging.INFO):
        self.iterable = iterable
        self.total = total or len(iterable)
        self.updates = updates
        self.name = name
        self.logger = logger
        self.level = level

    def update(self, **infos):
        self._infos = infos

    def __iter__(self):
        self._iterator = iter(self.iterable)
        self._index = -1
        self._infos = {}
        self._begin = time.time()
        return self

    def __next__(self):
        self._index += 1
        try:
            value = next(self._iterator)
        except StopIteration:
            raise
        else:
            return value
        finally:
            log_every = max(1, self.total // self.updates)
            # logging is delayed by 1 it, in order to have the metrics from update
            if self._index >= 1 and self._index % log_every == 0:
                self._log()

    def _log(self):
        self._speed = (1 + self._index) / (time.time() - self._begin)
        infos = " | ".join(f"{k.capitalize()} {v}" for k, v in self._infos.items())
        if self._speed < 1e-4:
            speed = "oo sec/it"
        elif self._speed < 0.1:
            speed = f"{1/self._speed:.1f} sec/it"
        else:
            speed = f"{self._speed:.1f} it/sec"
        out = f"{self.name} | {self._index}/{self.total} | {speed}"
        if infos:
            out += " | " + infos
        self.logger.log(self.level, out)


def colorize(text, color):
    """
    Display text with some ANSI color in the terminal.
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def bold(text):
    """
    Display text in bold in the terminal.
    """
    return colorize(text, "1")

def load_model_state_dict(model, state_dict):
    current_model_dict = model.state_dict()
    new_state_dict={k:v for k,v in state_dict.items() if list(v.size())==list(current_model_dict[k].size())}
    model.load_state_dict(new_state_dict, strict=False)
    return model

def apply_output_transform(rec_sources_wavs, input_mix_std,
                            input_mix_mean, input_mom, args):
    if args.rescale_to_input_mixture:
        rec_sources_wavs = (rec_sources_wavs * input_mix_std) + input_mix_mean
    if args.apply_mixture_consistency:
        rec_sources_wavs = apply_mixture_consistency(rec_sources_wavs, input_mom)
    return rec_sources_wavs


def apply_mixture_consistency(pr_batch, input_mixture, mix_weights_type="uniform"):
    """Apply mixture consistency
    :param pr_batch: Torch Tensors of size:
                    batch_size x self.n_sources x length_of_wavs
    :param input_mixture: Torch Tensors of size:
                    batch_size x 1 x length_of_wavs
    :param mix_weights_type: type of wights applied
    """
    num_sources = pr_batch.shape[1]
    pr_mixture = torch.sum(pr_batch, 1, keepdim=True)

    if mix_weights_type == "magsq":
        mix_weights = torch.mean(pr_batch ** 2, -1, keepdim=True)
        mix_weights /= torch.sum(mix_weights, 1, keepdim=True) + 1e-8
    elif mix_weights_type == "uniform":
        mix_weights = 1.0 / num_sources
    else:
        raise ValueError(
            "Invalid mixture consistency weight type: {}" "".format(mix_weights_type)
        )

    source_correction = mix_weights * (input_mixture - pr_mixture)
    return pr_batch + source_correction
