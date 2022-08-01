#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import os
import pytest
import numpy as np
import networkx as nx
from DNetPRO import DNetPRO
from sklearn.naive_bayes import GaussianNB
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import LeaveOneOut

__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']


class TestDNetPRO:
  '''
  Test DNetPRO model
  '''

  def test_constructor (self):

    params = {'estimator' : GaussianNB(priors=None, var_smoothing=1e-09),
              'cv' : LeaveOneOut(),
              'scoring' : None,
              'max_chunk' : 100,
              'percentage' : .1,
              'verbose' : False,
              'n_jobs' : 1}

    dnet = DNetPRO(**params)
    print(dnet)


  def test_max_chunk (self):

    params = {'estimator' : GaussianNB(priors=None, var_smoothing=1e-09),
              'cv' : LeaveOneOut(),
              'scoring' : None,
              'max_chunk' : -1,
              'percentage' : .1,
              'verbose' : False,
              'n_jobs' : 1}

    with pytest.raises(ValueError):
      dnet = DNetPRO(**params)


  def test_percentage (self):

    params = {'estimator' : GaussianNB(priors=None, var_smoothing=1e-09),
              'cv' : LeaveOneOut(),
              'scoring' : None,
              'max_chunk' : 1,
              'percentage' : 100,
              'verbose' : False,
              'n_jobs' : 1}

    with pytest.raises(ValueError):
      dnet = DNetPRO(**params)

  def test_n_jobs (self):

    params = {'estimator' : GaussianNB(priors=None, var_smoothing=1e-09),
              'cv' : LeaveOneOut(),
              'scoring' : None,
              'max_chunk' : 10,
              'percentage' : .1,
              'verbose' : False,
              'n_jobs' : 0}

    with pytest.raises(ValueError):
      dnet = DNetPRO(**params)

  def test_estimator (self):

    params = {'estimator' : None,
              'cv' : LeaveOneOut(),
              'scoring' : None,
              'max_chunk' : 1,
              'percentage' : .1,
              'verbose' : False,
              'n_jobs' : 1}

    with pytest.raises(ValueError):
      dnet = DNetPRO(**params)


  def test_label2numbers (self):

    y = ('A', 'A', 'B', 'B')
    num_y = DNetPRO.label2numbers(y)

    assert all(x == y for x, y in zip(num_y, (0, 0, 1, 1)))

  def test_pendrem_star (self):
    graph = nx.star_graph(5)

    g = DNetPRO.pendrem(graph)

    assert all(x == 0 for x in g.nodes())

  def test_pendrem_empty (self):
    graph = nx.empty_graph(5)

    g = DNetPRO.pendrem(graph)

    assert all(x == y for x, y in zip(g.nodes(), graph.nodes))

  def test_pendrem_path (self):
    graph = nx.path_graph(5)

    g = DNetPRO.pendrem(graph)

    assert all(x == y for x, y in zip(g.nodes(), graph.nodes))

  def test_pendrem_full (self):
    graph = nx.complete_graph(5)

    g = DNetPRO.pendrem(graph)

    assert all(x == y for x, y in zip(g.nodes(), graph.nodes))

  def test_estimator_type (self):
    dnet = DNetPRO(estimator=GaussianNB(), max_chunk=4)

    assert dnet._estimator_type == 'classifier'

  def test_is_fitted (self):
    dnet = DNetPRO(estimator=GaussianNB(), max_chunk=4)

    with pytest.raises(NotFittedError):
      pred_label = dnet.predict([])
      assert pred_label is not None

  def test_fit_max_chunk (self):
    dnet = DNetPRO(estimator=GaussianNB(), max_chunk=20)

    Nprobe, Nsample = (5, 4)

    X = np.arange(Nprobe * Nsample).reshape((Nsample, Nprobe))
    y = np.array(['A', 'A', 'B', 'B'])

    with pytest.raises(ValueError):
      dnet.fit(X, y)

  def test_random_fit (self):
    dnet = DNetPRO(estimator=GaussianNB(), max_chunk=4)

    Nprobe, Nsample = (5, 4)

    X = np.arange(Nprobe * Nsample).reshape((Nsample, Nprobe))
    y = np.array(['A', 'A', 'B', 'B'])

    dnet.fit(X, y)

    signature = dnet.get_signature()[0]

    assert all(x in ('number_of_genes', 'performace_couples', 'features', 'signature', 'score') for x in signature.keys())
    assert signature['number_of_genes'] == 2
    assert signature['score'] == 0
    assert signature['performace_couples'] == 0
    assert all(x == y for x, y in zip(signature['features'], (0, 1)))

  def test_equal_fit (self):
    dnet = DNetPRO(estimator=GaussianNB(), max_chunk=4)

    Nprobe, Nsample = (5, 4)

    X = np.full(shape=(Nsample, Nprobe), fill_value=42.)
    # y = np.zeros(shape=(Nsample), dtype=int)
    y = np.array(['A', 'A', 'B', 'B'])

    with pytest.warns(RuntimeWarning):
      dnet.fit(X, y)

    signature = dnet.signatures[0]

    assert all(x in ('number_of_genes', 'performace_couples', 'features', 'signature', 'score') for x in signature.keys())
    assert signature['number_of_genes'] == 2
    assert signature['score'] == 0.5
    assert signature['performace_couples'] == 2
    assert all(x == y for x, y in zip(signature['features'], (0, 1)))

  def test_set_signature (self):
    dnet = DNetPRO(estimator=GaussianNB(), max_chunk=4)

    Nprobe, Nsample = (5, 4)

    X = np.arange(Nprobe * Nsample).reshape((Nsample, Nprobe))
    y = np.array(['A', 'A', 'B', 'B'])

    dnet.fit(X, y)

    with pytest.raises(ValueError):
      signature = dnet.set_signature(-1)


  def test_fit_transform (self):
    dnet = DNetPRO(estimator=GaussianNB(), max_chunk=4)

    Nprobe, Nsample = (5, 4)

    X = np.arange(Nprobe * Nsample).reshape((Nsample, Nprobe))
    y = np.array(['A', 'A', 'B', 'B'])

    Xnew = dnet.fit_transform(X, y)
    assert Xnew.shape == (Nsample, 2)
    assert np.allclose(Xnew, X[:, :2])

  def test_score (self):
    dnet = DNetPRO(estimator=GaussianNB(), max_chunk=4)

    Nprobe, Nsample = (5, 4)

    X = np.arange(Nprobe * Nsample).reshape((Nsample, Nprobe))
    y = np.array(['A', 'A', 'B', 'B'])

    dnet.fit(X, y)

    score = dnet.score(X, y)
    assert np.isclose(score, 0.)

  def test_fit_fit_transform (self):
    dnet = DNetPRO(estimator=GaussianNB(), max_chunk=4)

    Nprobe, Nsample = (5, 4)

    X = np.arange(Nprobe * Nsample).reshape((Nsample, Nprobe))
    y = np.array(['A', 'A', 'B', 'B'])

    Xnew = dnet.fit_transform(X, y)

    Xnew2 = dnet.fit(X, y).transform(X)
    assert np.allclose(Xnew, Xnew2)

  def test_predict (self):
    dnet = DNetPRO(estimator=GaussianNB(), max_chunk=4)

    Nprobe, Nsample = (5, 4)

    X = np.full(shape=(Nsample, Nprobe), fill_value=42.)
    # y = np.zeros(shape=(Nsample), dtype=int)
    y = np.array(['A', 'A', 'B', 'B'])

    with pytest.warns(RuntimeWarning):
      dnet.fit(X, y)
      y_pred = dnet.predict(X)
      assert all(x == 0 for x in y_pred)

  def test_predict_proba (self):
    dnet = DNetPRO(estimator=GaussianNB(), max_chunk=4)

    Nprobe, Nsample = (5, 4)

    X = np.full(shape=(Nsample, Nprobe), fill_value=42.)
    # y = np.zeros(shape=(Nsample), dtype=int)
    y = np.array(['A', 'A', 'B', 'B'])

    with pytest.warns(RuntimeWarning):
      dnet.fit(X, y)
      proba = dnet.predict_proba(dnet.transform(X))
      assert np.isnan(proba).sum() == np.prod(proba.shape)

  def test_predict_logproba (self):
    dnet = DNetPRO(estimator=GaussianNB(), max_chunk=4)

    Nprobe, Nsample = (5, 4)

    X = np.full(shape=(Nsample, Nprobe), fill_value=42.)
    # y = np.zeros(shape=(Nsample), dtype=int)
    y = np.array(['A', 'A', 'B', 'B'])

    with pytest.warns(RuntimeWarning):
      dnet.fit(X, y)
      proba = dnet.predict_log_proba(dnet.transform(X))
      assert np.isnan(proba).sum() == np.prod(proba.shape)
