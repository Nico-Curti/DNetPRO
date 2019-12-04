#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from DNetPRO import DNetPRO

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import train_test_split

import numpy as np

__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']


if __name__ == '__main__':

  Nsample = 100
  Nfeature = 10000
  Ninformative = 10
  view = False

  np.random.seed(123)

  # generate synthetic data according to two classes and a fixed number of informative features
  # The first goal is to obtain better performances over this synthetic data
  # The second goal is to identify as much as possible informative feature in the best signature of DNetPRO
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
                                    random_state=123
                                    )
  classifier = GaussianNB()

  # Split the dataset in training and test for performances evaluation
  X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=42)

  # Create the DNetPRO feature selection object with GaussianNB classifier
  dnet = DNetPRO(estimator=classifier, scoring='accuracy', n_jobs=4, verbose=False)
  # extract the filtered dataset as the signature with highest score in the training set
  Dnet_data = dnet.fit_transform(X_train, y_train)
  new_sample, new_probe = Dnet_data.shape

  # Best DNetPRO signature
  dnet_signature = dnet.selected_signature

  # print some informations
  print('Signature DNetPRO: {}'.format(sorted(dnet_signature)))
  print('DNetPRO score: {:.3f}'.format(dnet.score(X_test, y_test)))
  print('Informative found: {:d} / {:d}'.format(len([x for x in dnet_signature if x < Ninformative]), Ninformative))

  # Compare the obtained results against the Kbest features with K=number of feature in the DNetPRO signature
  filter_kbest = SelectKBest(k=new_probe)
  # extract the filtered datasets
  Kbest_data = filter_kbest.fit_transform(X_train, y_train)
  # set to zero the other features
  Kbest_filtered = filter_kbest.inverse_transform(Kbest_data)
  # now it is easy to extract the selected features as non-zero columns
  Kbest_signature = set(np.nonzero(Kbest_filtered)[1])

  # Just to be sure that everything goes right...
  if not (len(Kbest_signature) == len(dnet_signature)):
    raise ValueError('Inconsistent length of features between the two methods')

  # print some informations
  print('Signature Kbest: {}'.format(sorted(Kbest_signature)))
  print('Kbest score: {:.3f}'.format(classifier.fit(Kbest_data, y_train).score(filter_kbest.transform(X_test), y_test)))
  print('Informative found: {:d} / {:d}'.format(len([x for x in Kbest_signature if x < Ninformative]), Ninformative))

  # Number of common features identified by the two methods
  print('Common features: {}'.format(len(set(Kbest_signature) & set(dnet_signature))))


  if view:

    # Visualize single feature scores

    import pylab as plt

    # use select kbest to evaluate the full set of features
    selector = SelectKBest(k=Nfeature)
    x_new = selector.fit_transform(data, label)
    # extract private score values of each feature
    scores = selector.scores_

    # sort in ascending order
    idx = np.argsort(scores)[::-1]
    # the informative features are placed on the top so the index has to be
    # less than the number of (fixed) informative features
    info = np.where(idx < Ninformative)[0]

    # start the plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    # plot the scores as concatenated dot line (blue)
    ax.plot(range(0, Nfeature), scores[idx],
            '-o', color='blue', alpha=.5,
            label='Single feature score')
    # highlight the informative features
    ax.scatter(info,
               scores[idx][info],
               marker='o', s=200,
               edgecolors='r', facecolors='none',
               linewidth=2, alpha=.5,
               label='Informative feature')

    # set some labels and legend
    ax.set_xlabel('Features', fontsize=24)
    ax.set_ylabel('Score', fontsize=24)
    ax.set_title('Single feature scores', fontsize=24)

    fig.legend(fontsize=14,
               loc='upper right',
               prop={'weight' : 'semibold',
                     'size':14},
               bbox_to_anchor=(0.8, 0.8)
               )

    # enlarge the tick font size
    for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(16)
    for tick in ax.yaxis.get_major_ticks():
      tick.label.set_fontsize(16)

    # add a informative text-box with some parameters
    textstr = '# features: {:d}\n# samples: {:d}\n# informative: {:d}'.format(
              Nfeature, Nsample, Ninformative)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax.text(0.5, 0.6, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)


