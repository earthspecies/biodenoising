# BIODENOISING: Animal vocalization denoising 

Here we provide the inference and training code. If you solely plan to do inference go to the following [github repo](https://github.com/earthspecies/biodenoising-inference)

Check the [biodenoising web page](https://earthspecies.github.io/biodenoising/) for demos and more info. 

The proposed model is based on the Demucs architecture, originally proposed for music source-separation and [real-time audio enhancement](https://github.com/facebook/denoiser). 

We publish the pre-print on [arXiv](https://arxiv.org/abs/2410.03427).

## Colab

If you want to play with the pretrained model inside colab for instance, start from this [Colab Example for Biodenoising](https://colab.research.google.com/drive/1Gc1tCe0MqAabViIgA8zGWm5KLVrEbRzg?usp=sharing).

## Installation

First, install Python >= 3.8 (recommended with miniconda).

#### Through pip (you just want to use pre-trained model out of the box)

Just run
```bash
pip install biodenoising
```

#### Development

Clone this repository and install the dependencies. We recommend using
a fresh virtualenv or Conda environment.

```bash
git clone https://github.com/earthspecies/biodenoising
cd biodenoising
pip install -r requirements.txt  
```

## Live Denoising

If you want to use `biodenoising` live, you will
need a specific loopback audio interface.

### Mac OS X

On Mac OS X, this is provided by [Soundflower][soundflower].
First install Soundflower, and then you can just run

```bash
python -m biodenoising.denoiser.live
```

In your favorite video conference call application, just select "Soundflower (2ch)"
as input to enjoy your denoised audio.

Watch our live demo presentation in the following link: [Demo][demo].

### Linux (tested on Ubuntu 20.04)

You can use the `pacmd` command and the `pavucontrol` tool:
- run the following commands:
```bash
pacmd load-module module-null-sink sink_name=denoiser
pacmd update-sink-proplist denoiser device.description=denoiser
```
This will add a `Monitor of Null Output` to the list of microphones to use. Select it as input in your software. 
- Launch the `pavucontrol` tool. In the _Playback_ tab, after launching 
`python -m biodenoising.denoiser.live --out INDEX_OR_NAME_OF_LOOPBACK_IFACE` and the software you want to denoise for (here an in-browser call), you should see both applications. For *denoiser* interface as Playback destination which will output the processed audio stream on the sink we previously created.

### Other platforms

At the moment, we do not provide official support for other OSes. However, if you
have a a soundcard that supports loopback (for instance Steinberg products), you can try
to make it work. You can list the available audio interfaces with `python -m sounddevice`.
Then once you have spotted your loopback interface, just run
```bash
python -m biodenoising.denoiser.live --out INDEX_OR_NAME_OF_LOOPBACK_IFACE
```
By default, `biodenoising` will use the default audio input. You can change that with the `--in` flag.

Note that on Windows you will need to replace `python` by `python.exe`.


### Troubleshooting bad quality in separation
Biodenoising inherits the drawbacks of the denoiser implementation: 

`denoiser` can introduce distortions for very high level of noises.
Audio can become crunchy if your computer is not fast enough to process audio in real time.
In that case, you will see an error message in your terminal warning you that `denoiser`
is not processing audio fast enough. You can try exiting all non required applications.

`denoiser` was tested on a Mac Book Pro with an 2GHz quadcore Intel i5 with DDR4 memory.
You might experience issues with DDR3 memory. In that case you can trade overall latency for speed by processing multiple frames at once. To do so, run
```
python -m biodenoising.denoiser.live -f 2
```
You can increase to `-f 3` or more if needed, but each increase will add 16ms of extra latency.


### Denoising received audio

You can also denoise received audio, but you won't be able to both denoise your own audio
and the received audio (unless you have a really beefy computer and enough loopback
audio interfaces). This can be achieved by selecting the loopback interface as
the audio output of your VC software and then running
```bash
python -m biodenoising.denoiser.live --in "Soundflower (2ch)" --out "NAME OF OUT IFACE"
```
The way experiments are automatically named, as explained hereafter.

## Usage

Generating the denoised files can be done by:

```
python -m biodenoising.denoiser.denoise --input=<path to the dir with the noisy files> --output=<path to store enhanced files>
```
Notice, you can either provide `noisy_dir` or `noisy_json` for the test data.
Note that the path given to `--model_path` should be obtained from one of the `best.th` file, not `checkpoint.th`.
It is also possible to use pre-trained model, using  `--biodenoising16k_dns48`.
 For more details regarding possible arguments, please see:
```
usage: python -m biodenoising.denoiser.denoise [-h] [-m MODEL_PATH | --biodenoising16k_dns48 ]
                        [--device DEVICE] [--dry DRY]
                        [--num_workers NUM_WORKERS] [--streaming]
                        [--output OUT_DIR] [--batch_size BATCH_SIZE] [-v]
                        [--input NOISY_DIR]

Animal vocalization denoising using biodenoising - Generate enhanced files

optional arguments:
  -h, --help                  show this help message and exit
  -m MODEL_PATH, --model_path MODEL_PATH
                              Path to local trained model.
  --biodenoising16k_dns48     Use pre-trained real time H=48 model trained on biodenoising-datasets.
  --device DEVICE
  --dry DRY                   dry/wet knob coefficient. 0 is only input signal, 1
                              only denoised.
  --num_workers NUM_WORKERS
  --streaming                 true streaming evaluation for biodenoising
  --output OUT_DIR            directory putting enhanced wav files
  --batch_size BATCH_SIZE
                              batch size
  -v, --verbose               more loggging
  --input NOISY_DIR
                              directory including noisy wav files
```

## Online Evaluation
This is from the original denoiser implementation: 

Our online implementation is based on pure python code with some optimization of the streaming convolutions and transposed convolutions.
We benchmark this implementation on a quad-core Intel i5 CPU at 2 GHz.
The Real-Time Factor (RTF) of the proposed models are:

| Model | Threads | RTF  |
|-------|---------|------|
| H=48  | 1       | 0.8  |
| H=48  | 4       | 0.6  |

In order to compute the RTF on your own CPU launch the following command:
```
python -m biodenoising.denoiser.demucs --hidden=48 --num_threads=1
```
The output should be something like this:
```
total lag: 41.3ms, stride: 16.0ms, time per frame: 12.2ms, delta: 0.21%, RTF: 0.8
```
Feel free to explore different settings, i.e. bigger models and more CPU-cores.

### Training

Training is done in three steps:
First we need to obtain the pseudo-clean training data:
```
python generate_training.py --out_dir /home/$USER/data/biodenoising16k/ --noisy_dir /home/$USER/data/biodenoising16k/dev/noisy/ --rir_dir /home/$USER/data/biodenoising16k/rir/ --method biodenoising16k_dns48 --transform none --device cuda
```

Then we need to prepare the csv files needed for training:
```
python prepare_experiments.py --data_dir /home/$USER/data/biodenoising16k/ --transform none --method biodenoising16k_dns48
```

Then we can train the model:
```
python train.py dset=biodenoising16k_biodenoising16k_dns48_none_step0 seed=0
```
### Domain Adaptation

Biodenoising is a generic tool that may fail in some cases. In order to improve the performance of the model in a specific domain, we can leverage domain adaptation. The adaptation process involves multiple steps of training on pseudo-clean targets to fine-tune the model for your specific audio domain.

#### Basic Usage

```bash
python adapt.py --method biodenoising16k_dns48 --noisy_dir /path/to/noisy/audio/ --out_dir /path/to/output/directory/
```

#### Advanced Options

The adaptation script supports numerous parameters to fine-tune the adaptation process:

```
usage: python adapt.py [-h] [--steps STEPS] [--noisy_dir NOISY_DIR] [--noise_dir NOISE_DIR]
                      [--test_dir TEST_DIR] [--out_dir OUT_DIR] [--noisy_estimate]
                      [-v] [--method {biodenoising16k_dns48}] [--segment SEGMENT]
                      [--highpass HIGHPASS] [--peak_height PEAK_HEIGHT]
                      [--transform {none,time_scale}] [--revecho REVECHO]
                      [--use_top USE_TOP] [--num_valid NUM_VALID] [--antialiasing]
                      [--force_sample_rate FORCE_SAMPLE_RATE]
                      [--time_scale_factor TIME_SCALE_FACTOR] [--noise_reduce]
                      [--amp_scale] [--interactive] [--window_size WINDOW_SIZE]
                      [--device DEVICE] [--dry DRY] [--num_workers NUM_WORKERS]
                      [-c CONFIG]

Adaptation parameters:
  --steps STEPS          Number of steps to use for adaptation (default: 5)
  --noisy_dir NOISY_DIR  Path to the directory with noisy wav files
  --noise_dir NOISE_DIR  Path to the directory with noise wav files
  --test_dir TEST_DIR    For evaluation: path to directory containing clean.json and noise.json files
  --out_dir OUT_DIR      Directory for enhanced wav files (default: "enhanced")
  --noisy_estimate       Compute noise as the difference between noisy and estimated signal
  
Model parameters:
  --method {biodenoising16k_dns48}
                        Method to use for denoising (default: "biodenoising16k_dns48")
  --device DEVICE        Device to use (default: "cuda")
  --dry DRY              Dry/wet knob coefficient. 0 is only denoised, 1 only input signal (default: 0)

Audio processing:
  --segment SEGMENT      Minimum segment size in seconds (default: 4)
  --highpass HIGHPASS    Apply a highpass filter with this cutoff before separating (default: 20)
  --peak_height PEAK_HEIGHT
                        Filter segments with rms lower than this value (default: 0.008)
  --transform {none,time_scale}
                        Transform input by pitch shifting or time scaling (default: "none")
  --revecho REVECHO      Revecho probability (default: 0)
  --antialiasing         Use an antialiasing filter when using time scaling (default: False)
  --force_sample_rate FORCE_SAMPLE_RATE
                        Force the model to take samples of this sample rate
  --time_scale_factor TIME_SCALE_FACTOR
                        If model has different sample rate, play audio slower/faster with this factor before resampling to the model sample rate
  --noise_reduce         Use noisereduce preprocessing
  --amp_scale            Scale to the amplitude of the input
  --window_size WINDOW_SIZE
                        Size of the window for continuous processing (default: 0)

Training options:
  --use_top USE_TOP      Use the top ratio of files for training, sorted by rms (default: 1.0)
  --num_valid NUM_VALID  Number of files to use for validation (default: 0)
  --interactive          Pause at each step to allow deleting files and continue
  --num_workers NUM_WORKERS
                        Number of workers (default: 5)

Configuration:
  -c CONFIG, --config CONFIG
                        Path to YAML configuration file (default: "biodenoising/conf/config_adapt.yaml")
  -v, --verbose          Enable verbose logging
```

The option `--interactive` allows for a manual inspection of the generated files and deletion of files for which the model is not performing well i.e. active learning.

#### Example Workflow

1. **Collect domain-specific noisy audio**: Gather audio samples from your target domain
2. **Run adaptation**:
   ```bash
   python adapt.py --method biodenoising16k_dns48 --noisy_dir /path/to/domain/audio/ --out_dir ./adapted_model/ --steps 3 --segment 2 --highpass 100
   ```
3. **Use your adapted model**: The adaptation process creates a fine-tuned model in the output directory

#### Tips for Effective Adaptation

- Use at least 5-10 minutes of audio from your target domain
- For wildlife recordings with specific frequency ranges, adjust the `--highpass` parameter
- If your recordings have specific noise characteristics, consider providing examples in `--noise_dir`
- The adaptation process works best with audio that has a good signal-to-noise ratio
- Use `--interactive` mode to inspect and manually filter generated files during adaptation

## Citation
If you use the code in your research, then please cite it as:
```
@misc{miron2024biodenoisinganimalvocalizationdenoising,
      title={Biodenoising: animal vocalization denoising without access to clean data}, 
      author={Marius Miron and Sara Keen and Jen-Yu Liu and Benjamin Hoffman and Masato Hagiwara and Olivier Pietquin and Felix Effenberger and Maddie Cusimano},
      year={2024},
      eprint={2410.03427},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2410.03427}, 
}
```

## License
This model is released under the CC-BY-NC 4.0. license as found in the [LICENSE](LICENSE) file.
