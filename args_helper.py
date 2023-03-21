import argparse
import sys
import yaml

from configs import parser as _parser


global parser_args

class ArgsHelper:
    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="GAN Compression")


        parser.add_argument(
            "--config",
            help="Config file to use"
        )
        # parser.add_argument(
        #     "--rank",
        #     type = int,
        #     default = 0,
        #     metavar = "G",
        #     help = "Override the default choice for a CUDA-enabled GPU by specifying the GPU\"s integer index (i.e. \"0\" for \"cuda:0\")"
        # )
        parser.add_argument(
            "--random_seed",
            type = int,
            default = 0,
            metavar = "S",
            help = "Random seed (default: 0, benchmark = True, deterministic = False)"
        )
        parser.add_argument(
            "--w_dim",
            type = int,
            default = 512,
            metavar = "S",
            help = "Random seed (default: 0, benchmark = True, deterministic = False)"
        )


        args = parser.parse_args()
        self.get_config(args)

        return args


    def get_config(self, parser_args):
        # get commands from command line
        override_args = _parser.argv_to_vars(sys.argv)

        # load yaml file
        yaml_txt = open(parser_args.config).read()

        # override args
        loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)

        for v in override_args:
            loaded_yaml[v] = getattr(parser_args, v)

        print(f"=> Reading YAML config from {parser_args.config}")
        parser_args.__dict__.update(loaded_yaml)


    def get_args(self):
        global parser_args
        parser_args = self.parse_arguments()

argshelper = ArgsHelper()
argshelper.get_args()
