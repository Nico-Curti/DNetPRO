#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from DNetPRO import DNetPRO
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':

  Nprobe, Nsample = (5, 4)

  X = np.arange(Nprobe * Nsample).reshape((Nprobe, Nsample))
  y = np.array(['A', 'A', 'B', 'B'])

  dnet = DNetPRO(estimator=GaussianNB())
  dnet.fit(X, y)

  print(dnet.signatures)
