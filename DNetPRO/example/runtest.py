#!/usr/bin/env python
# -*- coding: utf-8 -*-

from DNetPRO import DNetPRO
import numpy as np

if __name__ == '__main__':

  Nprobe, Nsample = (5, 4)

  X = np.arange(Nprobe * Nsample).reshape((Nprobe, Nsample))
  y = np.array(['A', 'A', 'B', 'B'])

  dnet = DNetPRO()
  dnet.fit(X, y)

  print(dnet.signatures)
