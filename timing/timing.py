#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

import itertools
import multiprocessing
from functools import partial

import timeit
from time import time as now

__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']

NUM_REPEATS = 3
NUMBER = 10


def couple_evaluation (couple, data, labels):
  f1, f2 = couple

  samples = data[:, [f1, f2]]
  score = cross_val_score(GaussianNB(), samples, labels,
                          cv=LeaveOneOut(), n_jobs=1).mean()
  return (f1, f2, score * 100.)

def couple_pooling (data, labels, nth):

  _, Nfeature = data.shape

  couples = itertools.combinations(range(0, Nfeature), 2)

  couple_eval = partial(couple_evaluation, data=data, labels=labels)

  with multiprocessing.Pool(nth) as pool:
    scores = zip(*pool.map(couple_eval, couples))

  scores = sorted(scores, key=lambda x : x[2], reverse=True)

  return scores



def python_version (Nsample, Nfeature, seed, nth):

  SETUP_CODE = '''
from __main__ import couple_pooling
from sklearn.datasets import make_classification

data, label = make_classification(n_samples={Nsample},
                                n_features={Nfeature},
                                n_classes=2,
                                n_clusters_per_class=1,
                                class_sep=1.,
                                scale=1.,
                                shuffle=False,
                                random_state={seed}
                                )
  '''.format(**{'Nsample':Nsample, 'Nfeature':Nfeature, 'seed':seed})

  TEST_CODE = '''
couple_pooling (data, label, {nth})
  '''.format(**{'nth':nth})

  times = timeit.repeat(setup=SETUP_CODE,
                        stmt=TEST_CODE,
                        repeat=NUM_REPEATS,
                        number=NUMBER)

  return times


def cpp_version (Nsample, Nfeature, seed, nth):

  SETUP_CODE = '''
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from DNetPRO import DNetPRO
from sklearn.datasets import make_classification

data, label = make_classification(n_samples={Nsample},
                                n_features={Nfeature},
                                n_classes=2,
                                n_clusters_per_class=1,
                                class_sep=1.,
                                scale=1.,
                                shuffle=False,
                                random_state={seed}
                                )

dnet = DNetPRO(estimator=GaussianNB(), cv=LeaveOneOut(), n_jobs={nth}, verbose=False)
  '''.format(**{'Nsample':Nsample, 'Nfeature':Nfeature, 'seed':seed, 'nth':nth})

  TEST_CODE = '''
dnet._evaluate_couples(data, label)
  '''

  times = timeit.repeat(setup=SETUP_CODE,
                        stmt=TEST_CODE,
                        repeat=NUM_REPEATS,
                        number=NUMBER)

  return times


if __name__ == '__main__':

  import numpy as np

  available_cores = multiprocessing.cpu_count()

  nths = range(2, available_cores + 2, 2)
  Nsamples = range(10, 100, 10)
  Nfeatures = range(10, 100, 10)

  parameters = itertools.product(nths, Nsamples, Nfeatures)

  with open('DNetPRO_timing.dat', 'w') as fp:

    row = 'num_repeats,number,samples,features,nth,py_mean,py_max,py_min,py_std,cpp_mean,cpp_max,cpp_min,cpp_std'
    fp.write(row + '\n')

    for seed, (nth, Nsample, Nfeature) in enumerate(parameters):

      print('Evaluating (sample={}, feature={}, nth={}) ...'.format(Nsample, Nfeature, nth),
           end='', flush=True)

      py_tic = now()

      times = python_version(Nsample, Nfeature, seed, nth)

      py_mean = np.mean(times)
      py_max  = np.max(times)
      py_min  = np.min(times)
      py_std  = np.std(times)

      py_toc = now()

      cpp_tic = now()

      times = cpp_version(Nsample, Nfeature, seed, nth)

      cpp_mean = np.mean(times)
      cpp_max  = np.max(times)
      cpp_min  = np.min(times)
      cpp_std  = np.std(times)

      cpp_toc = now()

      save = '{NUM_REPEATS},{NUMBER},{Nsample},{Nfeature},{NTH},\
{PY_MEAN:.3f},{PY_MAX:.3f},{PY_MIN:.3f},{PY_STD:.3f},\
{CPP_MEAN:.3f},{CPP_MAX:.3f},{CPP_MIN:.3f},{CPP_STD:.3f}'.format(
              **{'NUM_REPEATS':NUM_REPEATS, 'NUMBER':NUMBER,
                 'Nsample':Nsample, 'Nfeature':Nfeature, 'NTH':nth,
                 'PY_MEAN':py_mean, 'PY_MAX':py_max, 'PY_MIN':py_min, 'PY_STD':py_std,
                 'CPP_MEAN':cpp_mean, 'CPP_MAX':cpp_max, 'CPP_MIN':cpp_min, 'CPP_STD':cpp_std
                 })

      fp.write(save + '\n')

      print('  took {:.3f} seconds'.format(cpp_toc - py_tic), flush=True)

