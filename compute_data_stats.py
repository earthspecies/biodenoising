import argparse
import os

import torchaudio


parser = argparse.ArgumentParser(
        'stats',
        description="Generate denoised files")
parser.add_argument("--biodenoising_dir", type=str, required=True,
                    help="directory where data is")

def stats_dir(indir):
    ### get all subdirs
    subdirs = [ f.name for f in os.scandir(indir) if f.is_dir()]
    for subdir in subdirs:
        ### get all wav files
        files = [ f.name for f in os.scandir(os.path.join(indir,subdir)) if f.is_file() and f.name.endswith('.wav')]
        duration_sum = 0
        for file in files:
            ### get file duration with torchaudio.info 
            info = torchaudio.info(os.path.join(indir,subdir,file))
            duration = info.num_frames / info.sample_rate
            duration_sum += duration
        print(subdir, duration_sum/3600)
        
def stats(indir):
    ### first compute stats for the dev subfolder 
    print('Noisy')
    stats_dir(os.path.join(indir, 'dev', 'noisy'))
    print('Clean')
    stats_dir(os.path.join(indir, 'train', 'clean'))
    print('Noise')
    stats_dir(os.path.join(indir, 'train', 'noise'))


if __name__ == "__main__":
    args = parser.parse_args()
    stats(args.biodenoising_dir)