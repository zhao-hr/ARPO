from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import os
from coms_cleaned import coms_cleaned
from gradient_ascent import gradient_ascent
from romo import romo
from iom import iom
from gdr import gdr
from adr import adr
from arpo import arpo

def main(args):
    config = os.path.join('configs', args.config_filename + '.yaml')
    with open(config) as f:
        kwargs = yaml.load(f, Loader=yaml.Loader)
    exp = kwargs.get('exp')
    globals()[exp](**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config_filename",
                        default="arpo",
                        type=str,
                        help="Configuration filename for restoring the model."
                        )
    args = parser.parse_args()
    main(args)