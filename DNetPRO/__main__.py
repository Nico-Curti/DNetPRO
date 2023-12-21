#!/usr/bin/env python
# -*- coding: utf-8 -*-

# parse command line arguments
import argparse

# default classifier
from sklearn.naive_bayes import GaussianNB
# default cross-validation
from sklearn.model_selection import LeaveOneOut

# package version
from .__version__ import __version__
# DNetPRO algorithm
from DNetPRO import DNetPRO

__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']

GREEN_COLOR_CODE = '\033[38;5;40m'
ORANGE_COLOR_CODE = '\033[38;5;208m'
VIOLET_COLOR_CODE = '\033[38;5;141m'
RED_COLOR_CODE = '\033[38;5;196m'
RESET_COLOR_CODE = '\033[0m'

def parse_args ():

  description = ('DNetPRO algorithm - '
    'Discriminant Analysis with Network Processing'
  )

  # global sofware information
  parser = argparse.ArgumentParser(
    prog='dnetpro',
    argument_default=None,
    add_help=True,
    prefix_chars='-',
    allow_abbrev=True,
    description=description,
    epilog=f'DNetPRO Python package v{__version__}'
  )

  # dnetpro --version
  parser.add_argument(
    '--version', '-v',
    dest='version',
    required=False,
    action='store_true',
    default=False,
    help='Get the current version installed',
  )

  # input data filename
  parser.add_argument(
    '--input', '-i',
    dest='filepath',
    required=False,
    action='store',
    default=None,
    type=str,
    help='Input filename or path on which load the data.',
  )

  # number of threads to use
  parser.add_argument(
    '--nth', '-n',
    dest='nth',
    required=False,
    action='store',
    default=None,
    type=int,
    help='Number of threads to use for the computation.',
  )

  # dnetpro --verbose
  parser.add_argument(
    '--verbose', '-w',
    dest='verbose',
    required=False,
    action='store_true',
    default=True,
    help='Enable/Disable the code logging',
  )

  args = parser.parse_args()

  return args




def main ():

  # get the cmd parameters
  args = parse_args()

  if args.verbose:
    print(fr'''{GREEN_COLOR_CODE}
______ _   _      _  ____________ _____
|  _  \ \ | |    | | | ___ \ ___ \  _  |
| | | |  \| | ___| |_| |_/ / |_/ / | | |
| | | | . ` |/ _ \ __|  __/|    /| | | |
| |/ /| |\  |  __/ |_| |   | |\ \\ \_/ /
|___/ \_| \_/\___|\__\_|   \_| \_|\___/
      {RESET_COLOR_CODE}''',
      end='\n',
      flush=True,
    )

  # results if version is required
  if args.version:
    # print it to stdout
    print(f'DNetPRO package v{__version__}',
      end='\n', flush=True
    )
    # exit success
    exit(0)

  # # TODO: add data loading for X, y def

  # # define the feature selection algorithm
  # dnet = DNetPRO(
  #   estimator=GaussianNB(),
  #   cv=LeaveOneOut(),
  #   n_jobs=args.nth,
  #   verbose=args.verbose
  # )
  # # fit the model - aka extract the couples and
  # # evaluate the list of putative signatures
  # dnet.fit(X, y)
  # # get the best signature as the first element
  # # of the sorted list of signatures
  # best_signature = dnet.get_signature()[0]
  # # get the corresponding score of the features
  # # evaluated on the same dataset (aka overfitting)
  # best_score = dnet.score(X, y)

  # # log the results
  # print(f'{GREEN_COLOR_CODE}Best Signature: {best_signature}{RESET_COLOR_CODE}',
  #   end='\n', flush=True
  # )
  # print(f'{GREEN_COLOR_CODE}Score: {score:.3f}{RESET_COLOR_CODE}',
  #   end='\n', flush=True
  # )



if __name__ == '__main__':

  main ()
