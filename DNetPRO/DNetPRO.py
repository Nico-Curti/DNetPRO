#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import pandas as pd
import networkx as nx
from functools import partial
from operator import itemgetter

from sklearn.utils import check_X_y
from sklearn.utils import check_array
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted

from sklearn.model_selection import check_cv
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score

from sklearn.base import is_classifier
from sklearn.base import clone

from sklearn.metrics.scorer import check_scoring

from sklearn.preprocessing import LabelEncoder

from lib.DNetPRO.DNetPRO import _score
from lib.DNetPRO.DNetPRO import _DNetPRO_couples

__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']

class DNetPRO (_score):
  '''
  DNetPRO feature selection algorithm

  Parameters
  ----------
  estimator : object
      A supervised learning estimator with a ``fit`` method that provides
      information about feature importance either through a ``coef_``
      attribute or through a ``feature_importances_`` attribute.

  cv : int, cross-validation generator or an iterable, optional
      Determines the cross-validation splitting strategy.
      Possible inputs for cv are:

      - None, to use the default 3-fold cross-validation,
      - integer, to specify the number of folds.
      - An object to be used as a cross-validation generator.
      - An iterable yielding train/test splits.

      For integer/None inputs, if ``y`` is binary or multiclass,
      :class:`sklearn.model_selection.StratifiedKFold` is used. If the
      estimator is a classifier or if ``y`` is neither binary nor multiclass,
      :class:`sklearn.model_selection.KFold` is used.

      Refer :ref:`User Guide <cross_validation>` for the various
      cross-validation strategies that can be used here.

  scoring : string, callable or None, optional, default: None
      A string (see model evaluation documentation) or
      a scorer callable object / function with signature
      ``scorer(estimator, X, y)``.

  max_chunk : int, default=100
      Max number of features allowed in performances-chunk. If the size of chunk is greater than max_chunk
      and it is not the first one, features selection iteration is stopped.

  percentage : float, default=0.1
      Percentage of couples to save after sorting

  verbose : int, default=0
      Controls verbosity of couples evaluation.

  n_jobs : int, default 1
      Number of cores to run in parallel while fitting across folds.
      Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
      to number of cores.

  Notes
  -----
  TODO
  '''

  def __init__ (self, estimator, cv=LeaveOneOut(), scoring=None, max_chunk=100, percentage=.1, verbose=False, n_jobs=1):
    self.estimator = estimator
    self.cv = cv
    self.scoring = scoring

    self._max_chunk = max_chunk
    self._percentage = percentage
    self._verbose = verbose
    self._nth = n_jobs

    self._scorers = None

  def _pendrem (self, graph):
    '''
    Remove pendant node iterativelly
    '''
    deg = graph.degree()
    while min(map(itemgetter(1), deg)) < 2:
      graph.remove_nodes_from( [n for n, d in deg if d < 2] )
      deg = graph.degree()
      if len(deg) == 0: break
    return graph

  @property
  def _estimator_type (self):
    return self.estimator._estimator_type

  def _check_params (self, X, y):
    '''
    Check input parameters

    Parameters
    ----------
    X : array of shape [n_samples, n_features]
        The input samples.

    y : array of shape [n_samples]
        The target values.
    '''
    Nsample, Nprobe = np.shape(X)
    Ncomb = Nprobe * (Nprobe - 1) >> 1
    if not (self._max_chunk <= Ncomb or self._max_chunk > 0):
      raise ValueError('max_chunk must be >= 0 and <= possible features combinations. Given: {}'.format(self._max_chunk))

    if not (0. < self._percentage <= 1.):
      raise ValueError('percentage must be > 0 and <= 1. Given {}'.format(self._percentage))

    if not (0. < self._nth):
      raise ValueError('n_jobs must be a positive integer. Given {}'.format(self._percentage))


  def _label2numbers (self, arr):
    '''
    Convert labels to numerical values

    Parameters
    ----------
      arr : array_like
        The array of labels

    Returns
    -------
      numeric_labels : np.ndarray
        Array of numerical labels obtained by the LabelEncoder transform

    Notes
    -----
      The C++ function allows only numerical (integer) values as labels in input.
      For more general support refers to the C++ example.
    '''
    self.le = LabelEncoder()
    self.le.fit(arr)
    numeric_labels = self.le.transform(arr)
    return numeric_labels


  def _evaluate_couples (self, X, y):
    '''
    Evaluate feature couples using GaussianNB classifier using the C++ support function.

    Parameters
    ----------
      X : array-like of shape (Nsample, Nprobe)
          The training input samples

      y : array-like, shape (Nsample, )
          The target values (converted to unique integers)

    Returns
    -------
      performances : pandas dataframe
          Dataframe of ordered results with columns (feature_1, feature_2, performances).
          The variable pairs are sorted in ascending order according to the performance values.
    '''
    Nsample, _ = np.shape(X)

    if not isinstance(y[0], int):
      y = self._label2numbers(y)

    self.X = check_array(X)
    # set contiguous order memory for c++ compatibility
    self.y = np.ascontiguousarray(y)
    X = np.ascontiguousarray(self.X.T)

    # pay attention to the transposition
    score = _DNetPRO_couples(X, y, self._percentage, self._verbose, self._nth)
    performances = score.score

    return performances

  def _couple_evaluation (self, couple, data, labels):
    '''
    Evaluate couples of features with a LOOCV
    '''
    f1, f2 = couple

    samples = data[:, [f1, f2]]
    score = cross_val_score(self.estimator, samples, labels,
                            cv=LeaveOneOut(), n_jobs=1).mean()
    return (f1, f2, score * 100.)

  def _couple_pooling (self, data, labels):
    '''
    Compute the DNetPRO couples in pure Python
    '''

    Nsample, Nfeature = data.shape
    couples = itertools.combinations(range(0, Nfeature), 2)

    couple_eval = partial(self._couple_evaluation, data=data, labels=labels)

    scores = list(map(couple_eval, couples))
    scores = sorted(scores, key=lambda x : x[2], reverse=True)

    return scores

  def fit (self, X, y=None, **fit_params):
    '''
    Fit the DNetPRO model meta-transformer

    Parameters
    ----------
      X : array-like of shape (n_samples, n_features)
          The training input samples.

      y : array-like, shape (n_samples,)
          The target values (integers that correspond to classes in
          classification, real numbers in regression).

      **fit_params : Other estimator specific parameters

    Returns
    -------
      self : object
        Returns self.
    '''

    X, y = check_X_y(X, y)
    self._check_params(X, y)

    # Initialization
    cv = check_cv(self.cv, y, is_classifier(self.estimator))
    scorer = check_scoring(self.estimator, scoring=self.scoring)

    couples = self._evaluate_couples(X, y, **fit_params)
    couples = pd.DataFrame(data=couples, columns=['feature_1', 'feature_2', 'performances'])

    # compute the max of performances
    max_perf = couples.performances.max()

    # group-by variables by the same performance
    chunks = (couples.groupby('performances', sort=False))

    # resulting signature
    graph = nx.Graph()

    # results
    self.signatures = []

    for perf, chunk in chunks:

      if len(chunk) >= self._max_chunk and perf != max_perf:
        break

      graph.add_weighted_edges_from(chunk.values)
      sub_graphs = iter(nx.connected_component_subgraphs(graph))

      for comp in sub_graphs:

        g = self._pendrem(comp.copy())
        if len(g.nodes) != 0:
          comp = g
        sub_data = X[:, comp.nodes]
        score = cross_val_score(self.estimator, sub_data,
                                y=self.y, cv=cv,
                                scoring=scorer).mean()

        self.signatures.append({
                                 'number_of_genes'    : len(comp.nodes()),
                                 'performace_couples' : perf,
                                 'features'           : list(comp.nodes()),
                                 'signature'          : comp,
                                 'score'              : score
                               })

    return self

  def get_signature (self):
    '''
    Return the computed signature in ascending order (training score value)
    '''
    return sorted(self.signatures, key=lambda x : x['score'], reverse=True)

  def set_signature (self, index):
    '''
    Set the signature as selected features and fit the model

    Parameters
    ----------
      index : int
        Index of the signatures array
    '''
    if not 0 <= index < len(self.signatures):
      raise ValueError('Signature index out of range')

    self.selected_signature = self.get_signature()[index]['features']
    self.estimator_ = clone(self.estimator)
    self.estimator_.fit(self.transform(self.X), self.y)


  @if_delegate_has_method(delegate='estimator')
  def predict (self, X):
    '''
    Reduce X to the selected features and then predict using the
    underlying estimator.

    Parameters
    ----------
    X : array of shape [n_samples, n_features]
        The input samples.

    Returns
    -------
    y : array of shape [n_samples]
        The predicted target values.
    '''
    check_is_fitted(self, 'estimator_')
    return self.estimator.predict(self.transform(X))

  def predict_transform (self, X_train, y_train, X_test, y_test):
    '''
    Fit the DNetPRO model meta-transformer and apply the data transformation,
    i.e feature selection, and then compute the score on test set

    Parameters
    ----------
      X : array-like of shape (n_samples, n_features)
          The training input samples.

      y : array-like, shape (n_samples,)
          The target values (integers that correspond to classes in
          classification, real numbers in regression).

      **fit_params : Other estimator specific parameters

    Returns
    -------
      Xnew : array-like of shape (n_sample, n_signature_features)
             The data filtered according to the best features found by the model

      score : float
              Accuracy score over the test set (X_test)

    Notes
    -----
    The signature is selected as the signature with highest score on test (X_test) data.
    '''
    self.fit(X_train, y_train)

    scores = [self.estimator.fit(X_train[:, sign['features']], y_train).score(X_test[:, sign['features']], y_test)
              for sign in self.signatures]
    index = np.argmax(scores)
    self.set_signature(index)
    self.estimator_ = clone(self.estimator)
    Xnew = self.transform(self.X)
    self.estimator_.fit(Xnew, self.y)
    return (Xnew, scores[index])

  def fit_transform (self, X, y):
    '''
    Fit the DNetPRO model meta-transformer and apply the data transformation,
    i.e feature selection

    Parameters
    ----------
      X : array-like of shape (n_samples, n_features)
          The training input samples.

      y : array-like, shape (n_samples,)
          The target values (integers that correspond to classes in
          classification, real numbers in regression).

      **fit_params : Other estimator specific parameters

    Returns
    -------
      Xnew : array-like of shape (n_sample, n_signature_features)
             The data filtered according to the best features found by the model

    Notes
    -----
    The signature is selected as the signature with highest score on training (X) data.
    '''

    self.fit(X, y)
    self.set_signature(0)
    self.estimator_ = clone(self.estimator)
    Xnew = self.transform(self.X)
    self.estimator_.fit(Xnew, self.y)
    return Xnew

  def transform (self, X):
    '''
    Apply the data reduction according to the features in the best
    signature found.

    Parameters
    ----------
    X : array of shape [n_samples, n_features]
        The input samples.
    '''

    check_is_fitted(self, 'estimator_')
    X = X[:, self.selected_signature]
    return X

  @if_delegate_has_method(delegate='estimator')
  def score (self, X, y):
    '''
    Reduce X to the selected features and then return the score of the
    underlying estimator.

    Parameters
    ----------
    X : array of shape [n_samples, n_features]
        The input samples.

    y : array of shape [n_samples]
        The target values.
    '''

    check_is_fitted(self, 'estimator_')
    return self.estimator_.score(self.transform(X), y)

  @if_delegate_has_method(delegate='estimator')
  def decision_function (self, X):
    check_is_fitted(self, 'estimator_')
    return self.estimator_.decision_function(self.transform(X))

  @if_delegate_has_method(delegate='estimator')
  def predict_proba (self, X):
    check_is_fitted(self, 'estimator_')
    return self.estimator_.predict_proba(self.transform(X))

  @if_delegate_has_method(delegate='estimator')
  def predict_log_proba (self, X):
    check_is_fitted(self, 'estimator_')
    return self.estimator_.predict_log_proba(self.transform(X))



if __name__ == '__main__':

  from sklearn.naive_bayes import GaussianNB
  from sklearn.model_selection import train_test_split

  X = pd.read_csv('../bin/data.txt', sep='\t', index_col=0, header=0)
  y = np.asarray(X.columns.astype(float).astype(int))
  X = X.transpose()

  X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.33, random_state=42)

  dnet = DNetPRO(estimator=GaussianNB(), n_jobs=4, verbose=True)

  Xnew = dnet.fit_transform(X_train, y_train)
  print('Best Signature: {}'.format(dnet.get_signature()[0]))
  print('Score: {:.3f}'.format(dnet.score(X_test, y_test)))
