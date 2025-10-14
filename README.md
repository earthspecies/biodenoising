# BIODENOISING: Animal vocalization denoising 

Here we provide the inference and training code. If you solely plan to do inference go to the following [github repo](https://github.com/earthspecies/biodenoising-inference)

Check the [biodenoising web page](https://earthspecies.github.io/biodenoising/) for demos and more info. 

The proposed model is based on the Demucs architecture, originally proposed for music source-separation and [real-time audio enhancement](https://github.com/facebook/denoiser). 

We publish the pre-print on [arXiv](https://arxiv.org/abs/2410.03427).

## Quick start

- **Install (from PyPI)**

```bash
pip install biodenoising
```

- **Install (from source, editable)**

```bash
git clone https://github.com/earthspecies/biodenoising
cd biodenoising
pip install -r requirements.txt
pip install -e .
```

- **Denoise a folder (writes enhanced WAVs)**

```bash
biodenoise \
  --method biodenoising16k_dns48 \
  --noisy_dir /path/to/noisy_audio \
  --out_dir   /path/to/output_dir \
  --device cuda
```

- **Adapt the model to your domain/dataset (multi-step fine-tuning)**

```bash
biodenoise-adapt \
  --method biodenoising16k_dns48 \
  --noisy_dir /path/to/noisy_audio \
  --out_dir   /path/to/output_dir \
  --steps 3 \
  --epochs 10 \
  --device cuda
```

Notes:
- **--noisy_dir**: directory with your input audio files
- **--out_dir**: destination directory for outputs
- **--steps / --epochs (adapt)**: control adaptation passes and training epochs per step
 - **--keep_original_sr**: keep the original audio sample rate instead of resampling to the model rate (for high frequency vocalizations e.g. bats, belugas)
 - **--selection_table**: enable event-based masking using selection tables (csv/tsv/txt) next to audio files

## New features

- **Domain adaptation with `adapt.py`**: Fine-tune the pretrained `biodenoising16k_dns48` model on your own recordings using pseudo-clean targets generated from your data. This multi-step procedure (configure with `--steps` and `--epochs`) adapts the model to your target domain/dataset and can improve performance when the target acoustics differ from the original training data.

- **Event-aware processing with `--selection_table`**: When annotations (selection tables) are available next to your audio files, enabling `--selection_table` will restrict processing to annotated events. This can:
  - Improve denoising quality by removing the background outside the vocalizations.
  - Improve adaptation quality by using event-restricted targets and extracting the noise between events.

## Colab and Notebooks

Notebooks in `scripts/`:

- `scripts/Biodenoising_demo.ipynb` — [Open in Colab](https://colab.research.google.com/drive/1Gc1tCe0MqAabViIgA8zGWm5KLVrEbRzg?usp=sharing)
- `scripts/biodenoising_demo_long_audio.ipynb` — [Open in Colab](https://colab.research.google.com/drive/1DYzIC43s2f2-jeH3Qn29lFhSqqBnJjVi?usp=sharing)
- `scripts/Biodenoising_denoise_zip_demo.ipynb` — [Open in Colab](https://colab.research.google.com/drive/19kT5JBUvXucYHdUpL0_nGKRvD2QlGgif?usp=sharing)
- `scripts/Biodenoising_adapt_zip_demo.ipynb` — [Open in Colab](https://colab.research.google.com/drive/1Ypii2no6BtWgPvH7JikcaoNNQO48WN8v?usp=sharing)

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


## Usage

Once the package is installed generating the denoised files can be done by:

```
biodenoise \
  --method biodenoising16k_dns48 \
  --noisy_dir <path to the dir with the noisy files> \
  --out_dir   <path to store enhanced files>
```
Notes:
- You can either provide `--noisy_dir` (directory) or extend the tool to accept JSONs as in legacy flows.
- The path given to `--model_path` (when overriding the pretrained) should point to a `best.th` file, not `checkpoint.th`.
- Use `--selection_table` to restrict processing to annotated events; use `--keep_original_sr` to keep the input sampling rate.
For more details regarding possible arguments, see the CLI help:

```
biodenoise --help
```

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
python adapt.py --method biodenoising16k_dns48 --noisy_dir /path/to/noisy/audio/ --out_dir /path/to/output/directory/ --steps 3 --epochs 10
```

#### Advanced Options

The adaptation script supports numerous parameters to fine-tune the adaptation process:

```
usage: python adapt.py [-h] [--steps STEPS] [--noisy_dir NOISY_DIR] [--noise_dir NOISE_DIR]
                      [--test_dir TEST_DIR] [--out_dir OUT_DIR] [--noisy_estimate]
                      [--cfg CONFIG] [--epochs EPOCHS] [-v] [--method {biodenoising16k_dns48}] 
                      [--segment SEGMENT] [--highpass HIGHPASS] [--peak_height PEAK_HEIGHT]
                      [--transform {none,time_scale}] [--revecho REVECHO]
                      [--use_top USE_TOP] [--num_valid NUM_VALID] [--antialiasing]
                      [--force_sample_rate FORCE_SAMPLE_RATE]
                      [--time_scale_factor TIME_SCALE_FACTOR] [--noise_reduce]
                      [--amp_scale] [--interactive] [--window_size WINDOW_SIZE]
                      [--device DEVICE] [--dry DRY] [--num_workers NUM_WORKERS]
                      [--annotations] [--annotations_begin_column ANNOTATIONS_BEGIN_COLUMN]
                      [--annotations_end_column ANNOTATIONS_END_COLUMN]
                      [--annotations_label_column ANNOTATIONS_LABEL_COLUMN]
                      [--annotations_label_value ANNOTATIONS_LABEL_VALUE]
                      [--annotations_extension ANNOTATIONS_EXTENSION]
                      [--processed_dir PROCESSED_DIR] [--selection_table] [--keep_original_sr]

Adaptation parameters:
  --steps STEPS          Number of steps to use for adaptation (default: 5)
  --epochs EPOCHS        Number of epochs per step (default: 5)
  --noisy_dir NOISY_DIR  Path to the directory with noisy wav files
  --noise_dir NOISE_DIR  Path to the directory with noise wav files
  --test_dir TEST_DIR    For evaluation: path to directory containing clean.json and noise.json files
  --out_dir OUT_DIR      Directory for enhanced wav files (default: "enhanced")
  --noisy_estimate       Compute noise as the difference between noisy and estimated signal
  --processed_dir PROCESSED_DIR
                        Directory for storing preprocessed audio segments
  
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
  --selection_table      Enable event masking via selection tables (csv/tsv/txt) located next to audio files
  --keep_original_sr     Keep the original sample rate instead of resampling to model's sample rate

Annotation options:
  --annotations          Use annotation files to extract segments from audio files (default: False)
  --annotations_begin_column ANNOTATIONS_BEGIN_COLUMN
                        Column name for segment start time in annotation files (default: "Begin")
  --annotations_end_column ANNOTATIONS_END_COLUMN
                        Column name for segment end time in annotation files (default: "End")
  --annotations_label_column ANNOTATIONS_LABEL_COLUMN
                        Column name for segment label in annotation files (default: None)
  --annotations_label_value ANNOTATIONS_LABEL_VALUE
                        Filter annotations by this label value (default: None)
  --annotations_extension ANNOTATIONS_EXTENSION
                        Extension of annotation files (default: ".csv")

Training options:
  --use_top USE_TOP      Use the top ratio of files for training, sorted by rms (default: 1.0)
  --num_valid NUM_VALID  Number of files to use for validation (default: 0)
  --interactive          Pause at each step to allow deleting files and continue
  --num_workers NUM_WORKERS
                        Number of workers (default: 5)

Configuration:
  --cfg CONFIG           Path to YAML configuration file (default: "biodenoising/conf/config_adapt.yaml")
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

#### Using Annotations for Targeted Adaptation

The adaptation process supports using annotation files to extract specific segments:

```bash
python adapt.py --method biodenoising16k_dns48 --noisy_dir /path/to/audio/ --out_dir ./adapted_model/ --annotations --annotations_label_column "Call_Type" --annotations_label_value "Whistle"
```

This allows you to target adaptation to specific vocalizations or sound events in your recordings.

#### Using Selection Tables for Event-Based Processing

Both the denoising and adaptation processes support using selection tables for event-based processing:

```bash
# For denoising with selection tables
python -m biodenoising.denoiser.denoise --input /path/to/audio/ --output /path/to/output/ --selection_table

# For adaptation with selection tables
python adapt.py --method biodenoising16k_dns48 --noisy_dir /path/to/audio/ --out_dir ./adapted_model/ --selection_table
```

Selection tables are CSV, TSV, or TXT files located next to your audio files with the same base name. They should contain columns with start and end times in seconds. The system automatically detects columns with names like 'start', 'beginning', 'begin time', 'begin' for start times and 'end', 'end time' for end times.

When `--selection_table` is enabled:
- Only audio within the specified event intervals is processed for denoising
- Noise extraction focuses on gaps between events (with 0.2s buffer before and 0.4s after each event)
- The final output is masked to preserve only the denoised events

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
