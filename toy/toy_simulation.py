#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from DNetPRO import DNetPRO

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import train_test_split

import itertools
import numpy as np
import multiprocessing
from time import time as now

__package__ = 'DNetPRO toy model simulation'
__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']


if __name__ == '__main__':

  available_cores = multiprocessing.cpu_count()

  Nsamples = [100]#range(10, 50, 10)
  Nfeatures = [5000]#range(1000, 10000, 1000)
  Ninformatives = [2]#[2, 4, 6]#range(2, 10, 2)

  parameters = itertools.product(Ninformatives, Nsamples, Nfeatures)

  classifier = GaussianNB()

  with open('DNetPRO_toy.dat', 'w') as fp:

    row = 'samples,features,informative,dnet_score,dnet_informative,dnet_size,kbest_score,kbest_informative,common_features,time_dnet,time_kbest'
    fp.write(row + '\n')

    for seed, (Ninformative, Nsample, Nfeature) in enumerate(parameters):

      print('Evaluating (sample={}, feature={}, informative={}) ...'.format(Nsample, Nfeature, Ninformative),
            end='', flush=True)


      data, label = make_classification(n_samples=Nsample,
                                        n_features=Nfeature,
                                        n_informative=Ninformative,
                                        n_redundant=0,
                                        n_repeated=0,
                                        n_classes=2,
                                        n_clusters_per_class=1,
                                        class_sep=1.,
                                        scale=1.,
                                        shuffle=False,
                                        random_state=seed
                                        )

      X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42)

      dnet_tic = now()

      # DNetPRO feature selection

      dnet = DNetPRO(estimator=classifier, scoring='accuracy', n_jobs=available_cores, verbose=False, max_chunk=4)

      Dnet_data = dnet.fit_transform(X_train, y_train)
      #Dnet_data, dnet_score = dnet.predict_transform(X_train, y_train, X_test, y_test)
      new_sample, new_probe = Dnet_data.shape

      dnet_signature = dnet.selected_signature

      # DNetPRO parameters to save
      dnet_score = dnet.score(X_test, y_test)
      dnet_informative = len([x for x in dnet_signature if x < Ninformative])
      dnet_size = len(dnet_signature)

      dnet_toc = now()

      kbest_tic = now()

      # K-best feature selection

      filter_kbest = SelectKBest(k=new_probe)
      Kbest_data = filter_kbest.fit_transform(X_train, y_train)
      kbest_score = classifier.fit(Kbest_data, y_train).score(filter_kbest.transform(X_test), y_test)
      Kbest_filtered = filter_kbest.inverse_transform(Kbest_data)
      Kbest_signature = set(np.nonzero(Kbest_filtered)[1])

      assert (len(Kbest_signature) == len(dnet_signature))

      # K-best parameters to save
      kbest_informative = len([x for x in Kbest_signature if x < Ninformative])

      kbest_toc = now()

      common_features = len(set(Kbest_signature) & set(dnet_signature))

      # join string
      save = '{},{},{},{},{},{},{},{},{},{:.3f},{:.3f}'.format(Nsample, Nfeature, Ninformative,
                                                               dnet_score, dnet_informative, dnet_size,
                                                               kbest_score, kbest_informative, common_features,
                                                               dnet_toc - dnet_tic, kbest_toc - kbest_tic)

      fp.write(save + '\n')

      print('  took {:.3f} seconds (DNetPRO: {:.3f}, Kbest: {:.3f}, size: {:d})'.format(kbest_toc - dnet_tic, dnet_score, kbest_score, dnet_size), flush=True)
