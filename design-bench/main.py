from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml
import os

from gradient import gradient_ascent
from coms import coms
from iom import iom
from romo import romo
from gdr import gdr
from adr import adr
from arpo import arpo

def main(args):
    method = args.config.strip().split('-')[0]
    config = os.path.join('configs', method, args.config + '.yaml')
    with open(config) as f:
        kwargs = yaml.load(f, Loader=yaml.Loader)
    exp = kwargs.get('exp')
    globals()[exp](**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config",
                        default="arpo-tfbind8",
                        type=str,
                        help="Configuration filename for restoring the model."
                        )
    args = parser.parse_args()
    main(args)