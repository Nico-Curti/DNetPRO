#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from DNetPRO import DNetPRO

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import itertools
# import numpy as np
import pandas as pd
import multiprocessing
from time import time as now

__package__ = 'DNetPRO toy model simulation'
__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']


if __name__ == '__main__':

  available_cores = multiprocessing.cpu_count()

  Nsamples = range(100, 1000, 100)
  Nfeatures = range(1000, 10000, 1000)
  Ninformatives = range(2, 10, 2)

  parameters = itertools.product(Ninformatives, Nsamples, Nfeatures)

  classifier = GaussianNB()

  with open('DNetPRO_weights.dat', 'w') as fp:

    row = 'samples,features,informative,dnet_score,dnet_informative,dnet_size,kbest_score,kbest_informative,common_features'#,time_dnet,time_kbest'
    fp.write(row + '\n')

    for seed, (Ninformative, Nsample, Nfeature) in enumerate(parameters):

      print('Evaluating (sample={}, feature={}, informative={}) ...'.format(Nsample, Nfeature, Ninformative),
            end='\n', flush=True)


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

      # dnet_tic = now()

      # DNetPRO feature selection

      dnet = DNetPRO(estimator=classifier, scoring='accuracy', n_jobs=available_cores, verbose=False, max_chunk=10)

      # Dnet_data = dnet.fit_transform(X_train, y_train)
      # new_sample, new_probe = Dnet_data.shape

      # dnet_signature = dnet.selected_signature

      # # DNetPRO parameters to save
      # dnet_score = dnet.score(X_test, y_test)
      # dnet_informative = len([x for x in dnet_signature if x < Ninformative])
      # dnet_size = len(dnet_signature)

      # dnet_toc = now()

      # kbest_tic = now()

      # K-best feature selection

      single_perf = [accuracy_score(classifier.fit(X_train[:, i].reshape(-1, 1), y_train).predict(X_train[:, i].reshape(-1, 1)), y_train)
                     for i in range(X_train.shape[1])]
      # filter_kbest = SelectKBest(k=new_probe)
      # Kbest_data = filter_kbest.fit_transform(X_train, y_train)
      # Kbest_filtered = filter_kbest.inverse_transform(Kbest_data)
      # Kbest_signature = set(np.nonzero(Kbest_filtered)[1])

      # assert (len(Kbest_signature) == len(dnet_signature))

      ## K-best parameters to save
      # kbest_score = classifier.fit(Kbest_data, y_train).score(filter_kbest.transform(X_test), y_test)
      # kbest_informative = len([x for x in Kbest_signature if x < Ninformative])

      # kbest_toc = now()

      # common_features = len(set(Kbest_signature) & set(dnet_signature))

      # single_perf = filter_kbest.scores_
      single_perf = pd.DataFrame(data=single_perf, columns=['feature_1_ct'])
      single_perf['feature_1'] = single_perf.index

      dnet_perf = dnet._evaluate_couples(X_train, y_train)
      dnet_perf = pd.DataFrame(data=dnet_perf, columns=['feature_1', 'feature_2', 'ct'])
      dnet_perf.ct /= X_train.shape[0]

      dnet_perf = pd.merge(dnet_perf, single_perf, on='feature_1', how='inner')
      single_perf.rename(columns={'feature_1': 'feature_2'}, inplace=True)
      single_perf.rename(columns={'feature_1_ct': 'feature_2_ct'}, inplace=True)

      dnet_perf = pd.merge(dnet_perf, single_perf, on='feature_2', how='inner')

      dnet_perf['inv_ct_feature_1'] = 1. - dnet_perf.feature_1_ct
      dnet_perf['inv_ct_feature_2'] = 1. - dnet_perf.feature_2_ct
      dnet_perf['inv_ct_couples']   = 1. - dnet_perf.ct
      dnet_perf['weights'] = (dnet_perf[['inv_ct_feature_1', 'inv_ct_feature_2']].min(axis=1) / dnet_perf.inv_ct_couples) * dnet_perf.ct
      dnet_perf.sort_values(by='weights', inplace=True, ascending=False)

      # print('  took {:.3f} seconds'.format(kbest_toc - dnet_tic), flush=True)

      # print(dnet_perf[['feature_1', 'feature_2', 'ct', 'feature_1_ct', 'feature_2_ct', 'weights']].head(n=20))

      single_perf.sort_values(by='feature_2_ct', inplace=True, ascending=False)
      # print(single_perf.head(n=20))



      Dnet_data = dnet.fit_transform(dnet_perf, X_train, y_train)
      new_sample, new_probe = Dnet_data.shape

      dnet_signature = dnet.selected_signature

      # DNetPRO parameters to save
      dnet_score = dnet.score(X_test, y_test)
      dnet_informative = len([x for x in dnet_signature if x < Ninformative])
      dnet_size = len(dnet_signature)

      Kbest_data = single_perf.iloc[:dnet_size].feature_2.values.tolist()
      kbest_score = classifier.fit(X_train[:, Kbest_data], y_train).score(X_test[:, Kbest_data], y_test)

      kbest_informative = len([x for x in Kbest_data if x < Ninformative])

      common_features = len(set(Kbest_data) & set(dnet_signature))

      # join string
      save = '{},{},{},{},{},{},{},{},{}'.format(Nsample, Nfeature, Ninformative,
                                                               dnet_score, dnet_informative, dnet_size,
                                                               kbest_score, kbest_informative, common_features)#,
                                                               # dnet_toc - dnet_tic, kbest_toc - kbest_tic)

      fp.write(save + '\n')

