---
layout: page
#
# Content
#
subheadline: "Unsupervised animal vocalization denoising"
sidebar: right
title: "Biodenoising"
teaser: "how to denoise animal vocalizations without clean data?"
permalink: "/research/biodenoising/"
categories:
  -
tags:
  - bioacoustics, denoising
header:
   image_fullwidth: "whale.png"
mediaplayer: true

---

## Introduction 

![biodenoising]({{ site.url }}/images/biodenoising.jpg)

We present Biodenoising, a new method for animal vocalization denoising that does not require access to clean data. There are two core ideas behind Biodenoising:
- We leverage existing speech enhancement models: there is no need to start from scratch. Speech enhancement models have seen a lot of data and have some knowledge about patterns in audio time series. 
- Following the same logic, there is no need to train a separate model for each animal dataset. Since lots of signal characteristics are replicated at different scales and frequencies, a model that has seen more diverse data can be more robust to unseen conditions and generalize better.

There is a eloquent video about how these audio patterns work for whales and birds.

<video width="640" height="360" id="player1" preload="none">
  <source type="video/youtube" src="http://www.youtube.com/watch?v=M5OCCuCIMbA" />
</video>

We publish the pre-print on [arXiv](https://arxiv.org/abs/2410.03427).

{% include alert text='Marius Miron, Sara Keen, Jen-Yu Liu, Benjamin Hoffman, Masato Hagiwara, Olivier Pietquin, Felix Effenberger, Maddie Cusimano, "Biodenoising: animal vocalization denoising without access to clean data"' %}

Along with the pre-print, we publish two Python pip-installable libraries 'biodenoising', 'biodenoising-inference', and 'biodenoising-datasets' that can be used to denoise animal vocalizations and download the datasets. 

| [Github](https://github.com/earthspecies/biodenoising) | [Github inference](https://github.com/earthspecies/biodenoising-inference) | [Github Datasets](https://github.com/earthspecies/biodenoising-datasets) | [Colab](https://colab.research.google.com/drive/1Gc1tCe0MqAabViIgA8zGWm5KLVrEbRzg?usp=sharing) |

We base our work on the speech enhancement models [demucs dns 48](https://github.com/facebookresearch/denoiser) and [CleanUNet](https://github.com/NVIDIA/CleanUNet) because they were small models and fast to train. Demucs worked particularly well. The performance may improve by training newer architectures.  

#### Abstract
{% include alert text='
Animal vocalization denoising is a task similar to human speech enhancement, a well-studied field of research. In contrast to the latter, it is applied to a higher diversity of sound production mechanisms and recording environments, and this higher diversity is a challenge for existing models. Adding to the challenge and in contrast to speech, we lack large and diverse datasets comprising clean vocalizations. As a solution we use as training data pseudo-clean targets, i.e. pre-denoised vocalizations, and segments of background noise without a vocalization. We propose a train set derived from bioacoustics datasets and repositories representing diverse species, acoustic environments, geographic regions. Additionally, we introduce a non-overlapping benchmark set comprising clean vocalizations from different taxa and noise samples. We show that that denoising models (demucs, CleanUNet) trained on pseudo-clean targets obtained with speech enhancement models achieve competitive results on the benchmarking set. We publish data, code, libraries, and demos https://mariusmiron.com/research/biodenoising.'%}

## Benchmarking dataset
We introduce a benchmarking dataset for animal vocalization denoising, [Biodenoising_validation][27]. It contains 62 pairs of clean animal vocalizations and noise excerpts. We list some audio demos from this dataset below. Details about the training data can be found at the end of this page.

## Audio demos

Here we look at zero-shot performance of the methods on the benchmarking dataset, i.e. generalization to unseen taxa and noise. None of the methods has been adapted/seen to the tested datasets. So the performance may improve when doing self-training to those data. We are actually working on such a method.

First, we compare the original noisy file with our denoising trained on pseudo-clean targets(biodenoising) and two state of the art methods noisereduce and noisy target training. 

| Original | Biodenoising | Noisereduce | Noisy target |
|----------|--------------|-------------|--------------|
| <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/12_terrestrial_original.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/12_terrestrial_biodenoising.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/12_terrestrial_noisereduce.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/12_terrestrial_noisy_target.wav" type="audio/wav" controls="controls"></audio> |
| <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/14_terrestrial_original.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/14_terrestrial_biodenoising.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/14_terrestrial_noisereduce.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/14_terrestrial_noisy_target.wav" type="audio/wav" controls="controls"></audio> |
| <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/21_terrestrial_original.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/21_terrestrial_biodenoising.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/21_terrestrial_noisereduce.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/21_terrestrial_noisy_target.wav" type="audio/wav" controls="controls"></audio> |
| <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/36_underwater_original.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/36_underwater_biodenoising.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/36_underwater_noisereduce.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/36_underwater_noisy_target.wav" type="audio/wav" controls="controls"></audio> |
| <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/30_terrestrial_original.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/30_terrestrial_biodenoising.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/30_terrestrial_noisereduce.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/30_terrestrial_noisy_target.wav" type="audio/wav" controls="controls"></audio> |
| <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/34_terrestrial_original.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/34_terrestrial_biodenoising.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/34_terrestrial_noisereduce.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/34_terrestrial_noisy_target.wav" type="audio/wav" controls="controls"></audio> |
| <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/3_terrestrial_original.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/3_terrestrial_biodenoising.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/3_terrestrial_noisereduce.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/3_terrestrial_noisy_target.wav" type="audio/wav" controls="controls"></audio> |

How well does it do on longer recordings? 

| Original | Biodenoising | Noisereduce |
|----------|--------------|-------------|
| <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/22_terrestrial_original.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/22_terrestrial_biodenoising.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/22_terrestrial_noisereduce.wav" type="audio/wav" controls="controls"></audio> 
| <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/52_underwater_original.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/52_underwater_biodenoising.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/52_underwater_noisereduce.wav" type="audio/wav" controls="controls"></audio> |
| <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/24_underwater_original.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/24_underwater_biodenoising.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/benchmark/24_underwater_noisereduce.wav" type="audio/wav" controls="controls"></audio> |

Recording animals in the lab does not always yield clean vocalizations. In fact these zebra finch recorded with a close-mic are noisy because you can hear the fan and the wings and hopping. And noisereduce while it works great for the fan noise it can not do a good job for the wings and hopping.

| Original | Biodenoising | Noisereduce |
|----------|--------------|-------------|
| <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/bl22gr2_bl39gr19_July_07_2023_32957426_original.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/bl22gr2_bl39gr19_July_07_2023_32957426_biodenoising.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/bl22gr2_bl39gr19_July_07_2023_32957426_noisereduce.wav" type="audio/wav" controls="controls"></audio> |

The most difficult condition is when we try to denoise biologger recordings, like this carrion crow. Again the wind and the self-noise are very loud.

| Original | Biodenoising | Noisereduce |
|----------|--------------|-------------|
| <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/19_BUCK_Azul_1008_16_original.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/19_BUCK_Azul_1008_16_biodenoising.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/19_BUCK_Azul_1008_16_noisereduce.wav" type="audio/wav" controls="controls"></audio> |
| <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/19_N7_Morado_1007_610_original.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/19_N7_Morado_1007_610_biodenoising.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/19_N7_Morado_1007_610_noisereduce.wav" type="audio/wav" controls="controls"></audio> |

Underwater conditions tend to be noisier than terrestrial conditions. These models were not trained to operate below -5dB SNR but they can still perform reasonably well. Here you can find recordings of orcas from Orcasound and South-Alaska humpback whale recorded by Michelle Fournet. 

| Original | Biodenoising | Noisereduce |
|----------|--------------|-------------|
| <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/orcasound_original.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/orcasound_biodenoising.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/orcasound_noisereduce.wav" type="audio/wav" controls="controls"></audio> |
| <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/sahw_190806153551_1_original.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/sahw_190806153551_1_biodenoising.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/sahw_190806153551_1_noisereduce.wav" type="audio/wav" controls="controls"></audio> |

My favorite recording is the one of a bowhead whale from the Watkins Marine Mammals dataset. Note that in contrast to the examples above this noisy recording was pre-cleaned using speech enhancement models and then used in training. This recording motivated me to start this project. 

| Original | Biodenoising | Noisereduce |
|----------|--------------|-------------|
| <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/bhw_78018002_original.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/bhw_78018002_biodenoising.wav" type="audio/wav" controls="controls"></audio> | <audio src="https://storage.googleapis.com/esp-public-files/biodenoising/demo/others/bhw_78018002_noisereduce.wav" type="audio/wav" controls="controls"></audio> |

## Training dataset description 

| Noisy datasets              | Hours  | Medium     | Private | Direct | Link       | Type        |
|-----------------------------|--------|------------|---------|--------|------------|-------------|
| Dolphin signature whistles  | 0.23   | underwater |   yes   |  no    | [link][1]  | dolphins    |
| Hanaian Gibbons             | 1.11   | terrestrial|   no    |  yes   | [link][2]  | gibbons     |
| Geladas                     | 2.23   | terrestrial|   yes   |  no    | [link][3]  | geladas     |
| Orcasound Aldev             | 0.25   | underwater |   no    |  yes   | [link][4]  |  orcas      |
| Thyolo                      | 0.61   | terrestrial|   no    |  yes   | [link][1]  |  birds      |
| Anuran                      | 1.13   | terrestrial|   no    |  no    | [link][5]  |  frogs      |
| South-Alaska humpback whale | 14.13  | underwater |   yes   |  no    | [link][6]  |  cetaceans  |
| Orcasound SKRW              | 2.41   | underwater |   no    |  yes   | [link][4]  |  orcas      |
| Black and white ruffed lemur| 1.06   | terrestrial|   no    |  yes   | [link][7]  |  lemurs     |
| Orcasound humpback whale    | 0.8    | underwater |   no    |  yes   | [link][4]  |  orcas      |
| Orchive                     | 0.03   | underwater |   no    |  yes   | [link][13] |  orcas      |
| Whydah                      | 0.57   | terrestrial|   no    |  yes   | [link][8]  |  birds      |
| Sabiod NIPS4B               | 0.55   | underwater |   no    |  yes   | [link][9]  |  cetaceans  |
| Xeno canto labeled subset   | 6.82   | terrestrial|   no    |  yes   | [link][26] |  birds      |
| ASA Berlin                  | 4.69   | terrestrial|   no    |  no    | [link][10] |  various    |
| Watkins                     | 5.33   | underwater |   no    |  yes   | [link][11] |  various    |
| Macaques coo calls          | 0.7    | terrestrial|   no    |  yes   | [link][15] |  macaques   |

| Noise datasets              | Hours  | Medium     | Private | Direct | Link       | Type        |
|-----------------------------|--------|------------|---------|--------|------------|-------------|
| FSD50k subset               | 26.34  | terrestrial|   no    |  yes   | [link][19] |  various    |
| IDMT Traffic                | 9.72   | terrestrial|   no    |  yes   | [link][20] |  streets    |
| ShipsEar                    | 3.55   | underwater |   yes   |  no    | [link][21] |  ships      |
| DeepShip subset             | 1.78   | underwater |   no    |  yes   | [link][22] |  ships      |
| Orcasound ship noise        | 7.23   | underwater |   no    |  yes   | [link][4]  |  ships      |
| TUT 2016 subset             | 0.33   | terrestrial|   no    |  yes   | [link][23] |  home       |


| Extracted noise             | Hours  | Medium     | Private | Direct | Link       | Type        |
|-----------------------------|--------|------------|---------|--------|------------|-------------|
| MARS MBARI                  | 0.5    | underwater |   no    |  yes   | [link][24] |  various    |
| NOAA Sanctsound             | 47.48  | underwater |   no    |  yes   | [link][25] |  various    |
| Orcasound best os           | 1.6    | underwater |   no    |  yes   | [link][4]  |  various    |
| South-Alaska humpback whale | 114.85 | underwater |   yes   |  no    | [link][6]  |  various    |

## Bibtex 
{% include alert text='
@misc\{miron2024biodenoisinganimalvocalizationdenoising,
      title={Biodenoising: animal vocalization denoising without access to clean data}, 
      author={Marius Miron and Sara Keen and Jen-Yu Liu and Benjamin Hoffman and Masato Hagiwara and Olivier Pietquin and Felix Effenberger and Maddie Cusimano\},
      year=\{2024\},
      eprint=\{2410.03427\},
      archivePrefix=\{arXiv\},
      primaryClass=\{cs.SD\},
      url=\{https://arxiv.org/abs/2410.03427\}, 
\}
"' %}

 [1]: https://www.sciencedirect.com/science/article/abs/pii/S0003347207002722#:~:text=In%201965%2C%20Melba%20and%20David,et%20al.%2C%201990
 [2]: https://zenodo.org/record/7997739
 [3]: https://link.springer.com/article/10.1007/s00265-018-2612-5
 [4]: https://github.com/awslabs/open-data-registry/blob/main/datasets/orcasound.yaml
 [5]: https://www.kaggle.com/datasets/mehmetbayin/anuran-sound-frogs-or-toads-dataset
 [6]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6071617/
 [7]: https://zenodo.org/record/6331594/
 [8]: https://zenodo.org/record/6330711
 [9]: https://sabiod.univ-tln.fr/nips4b
 [10]: https://www.museumfuernaturkunde.berlin/en/science/animal-sound-archive
 [11]: https://whoicf2.whoi.edu/science/B/whalesounds/about.cfm
 [12]: "http://download.magenta.tensorflow.org/datasets/nsynth
 [13]: https://github.com/earthspecies/library/tree/main/orcas
 [14]: https://zenodo.org/record/1442513
 [15]: https://archive.org/download/macaque_coo_calls
 [16]: https://archive.org/download/giant_otters
 [17]: https://archive.org/download/egyptian_fruit_bats_10k
 [18]: https://zenodo.org/record/1206938/
 [19]: https://zenodo.org/record/4060432/
 [20]: https://zenodo.org/record/7551553
 [21]: https://www.sciencedirect.com/science/article/abs/pii/S0003682X16301566
 [22]: https://github.com/irfankamboh/DeepShip
 [23]: https://zenodo.org/record/996424
 [24]: https://www.mbari.org/project/open-acoustic-data/ 
 [25]: https://sanctuaries.noaa.gov/news/feb21/sanctsound-overview.html
 [26]: https://doi.org/10.5281/zenodo.7828148
 [27]: https://zenodo.org/records/13736465
 [28]: #
 [29]: #
 [30]: #
 [31]: #