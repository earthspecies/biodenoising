#!/usr/bin/env python3
# Copyright (c) Earth Species Project. This work is based on Facebook's denoiser.


from pathlib import Path

from setuptools import setup, find_packages

NAME = 'biodenoising'
DESCRIPTION = (
    'Animal vocalization denoising')

URL = 'https://github.com/earthspecies/biodenoising'
EMAIL = 'info@mariusmiron.com'
AUTHOR = 'Marius Miron'
REQUIRES_PYTHON = '>=3.8.0'
VERSION = "0.3.3"

HERE = Path(__file__).parent

REQUIRED = [
    'julius',
    'numpy>=1.19',
    'six',
    'sounddevice>=0.4',
    'torch>=1.5',
    'torchaudio>=0.5',
    'openunmix',
    'einops',
    'omegaconf==1.4.1',
    'noisereduce',
    'scikit-fuzzy',
    'prosemble',
    'librosa',
    'norbert'
]

REQUIRED_LINKS = [
]

try:
    with open(HERE / "README.md", encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
    dependency_link=REQUIRED_LINKS,
    include_package_data=True,
    license='Creative Commons Attribution-NonCommercial 4.0 International',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Telecommunications Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
