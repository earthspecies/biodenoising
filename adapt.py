import argparse
import biodenoising
import logging
import os
import sys
import yaml

logger = logging.getLogger(__name__)

class ConfigParser():
    def __init__(self, *pargs, **kwpargs):
        self.options = []
        self.pargs = pargs
        self.kwpargs = kwpargs
        self.conf_parser = argparse.ArgumentParser(add_help=False)
        self.conf_parser.add_argument("-c", "--config",
                                 default="biodenoising/conf/config_adapt.yaml",
                                 help="where to load YAML configuration")
        
    def add_argument(self, *args, **kwargs):
        self.options.append((args, kwargs))

    def parse(self, args=None):
        if args is None:
            args = sys.argv[1:]

        res, remaining_argv = self.conf_parser.parse_known_args(args)

        config_vars = {}
        if res.config is not None:
            with open(res.config, 'rb') as stream:
                config_vars = yaml.safe_load(stream.read())

        parser = argparse.ArgumentParser(
            *self.pargs,
            # Inherit options from config_parser
            parents=[self.conf_parser],
            # Don't mess with format of description
            formatter_class=argparse.RawDescriptionHelpFormatter,
            **self.kwpargs,
        )

        for opt_args, opt_kwargs in self.options:
            parser_arg = parser.add_argument(*opt_args, **opt_kwargs)
            if parser_arg.dest in config_vars:
                config_default = config_vars.pop(parser_arg.dest)
                expected_type = str
                if parser_arg.type is not None:
                    expected_type = parser_arg.type
                # import pdb; pdb.set_trace()
                # if not isinstance(config_default, expected_type):
                if not issubclass(type(config_default),expected_type):
                    parser.error('YAML configuration entry {} '
                                 'does not have type {}'.format(
                                     parser_arg.dest,
                                     expected_type))

                parser_arg.default = config_default
        
        
        for k,v in config_vars.items():
            parser.set_defaults(**{k:v})
            if k=='dset': 
                for k1,v1 in v.items():
                    parser.set_defaults(**{k1:v1})
            
        return parser.parse_args(remaining_argv)

# parser = argparse.ArgumentParser(
#         'adapt',
#         description="Adapt model to the noisy dataset by training on pseudo-clean targets")
parser = ConfigParser()
parser.add_argument("--steps", default=5, type=int, help="Number of steps to use for adaptation")
parser.add_argument("--noisy_dir", type=str, default=None,
                    help="path to the directory with noisy wav files")
parser.add_argument("--noise_dir", type=str, default=None,
                    help="path to the directory with noise wav files")
parser.add_argument("--out_dir", type=str, default="enhanced",
                    help="directory putting enhanced wav files")
# parser.add_argument("--cfg", type=str, default="biodenoising/conf/config_adapt.yaml",
#                     help="path to the directory with noise wav files")
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
parser.add_argument("--force_sample_rate", default=0, type=int, help="Force the model to take samples of this sample rate")
parser.add_argument("--time_scale_factor", default=0, type=int, help="If the model has a different sample rate, play the audio slower or faster with this factor. If force_sample_rate this automatically changes.")
parser.add_argument('--noise_reduce', action="store_true",help="use noisereduce preprocessing")
parser.add_argument('--amp_scale', action="store_true",help="scale to the amplitude of the input")
parser.add_argument('--interactive', action="store_true",help="pause at each step to allow the user to delete some files and continue")
parser.add_argument("--window_size", type=int, default=0,
                    help="size of the window for continuous processing")
parser.add_argument('--device', default="cuda")
parser.add_argument('--dry', type=float, default=0,
                    help='dry/wet knob coefficient. 0 is only denoised, 1 only input signal.')
parser.add_argument('--num_workers', type=int, default=5)

def main(args):
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    model = None
    os.makedirs(os.path.join(args.out_dir, 'checkpoints'), exist_ok=True)

    for step in range(args.steps):
        model = biodenoising.adapt.denoise(args, step=step)
        biodenoising.adapt.generate_json(args, step=step)
        if step>0:
            args.continue_from = os.path.join(args.out_dir, 'checkpoints', args.checkpoint_file)
            args.checkpoint_file = os.path.basename(args.checkpoint_file).replace('_step'+str(step-1)+'.th', '_step'+str(step)+'.th')
        else:
            args.continue_from = ''
            args.checkpoint_file = args.checkpoint_file.replace('.th', '_step0.th')
        args.checkpoint_file = os.path.join(args.out_dir, 'checkpoints', args.checkpoint_file)
        args.history_file = os.path.join(args.out_dir, 'checkpoints', args.history_file)
        biodenoising.adapt.train(args,step=step)
        args.model_path = args.checkpoint_file
        args.lr = args.lr*0.5
    
    model = biodenoising.adapt.denoise(args, step=step+1)


if __name__ == "__main__":
    args = parser.parse()
    if args.method == 'biodenoising16k_dns48':
        args.biodenoising16k_dns48 = True
    
    main(args)