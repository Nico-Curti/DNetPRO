#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import pickle

__package__ = 'DNetPRO extract Kbest'
__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']


def extract_kbest (local, cancer, num_genes, percent=.1):
  '''

  '''
  if cancer not in ['KIRC', 'GBM', 'LUSC', 'OV']:
    raise ValueError('Cancer type not found')

  couples_directory = os.path.join(local, cancer, 'Train_' + cancer + '_mRNA', 'couples_mRNA')

  data_type = [('genea', np.uint16), ('geneb', np.uint16), ('c0', np.uint16), ('c1', np.uint16), ('ct', np.uint16)]
  columns = list(x[0] for x in data_type)

  couple_files = (os.path.join(couples_directory, f) for f in os.listdir( couples_directory + os.sep ) if os.path.isfile(os.path.join( couples_directory, f)))

  kbest = []

  for f in couple_files:

    couples = pd.DataFrame(np.fromfile(f, dtype=data_type).tolist(), columns=columns)[: int(num_genes * percent) + 1] # only single couples!

    #assert np.allclose(couples.genea, couples.geneb)
    genes = couples.genea.values
    genes = map(str, genes)
    kbest.append(genes)

  return kbest



if __name__ == '__main__':

    cancers = {'KIRC' : 20530,
               'GBM'  : 17814,
               'LUSC' : 20530,
               'OV'   : 17814
               }
    local = os.path.abspath('.')

    for cancer, num_genes in cancers.items():
      print('Processing {}'.format(cancer))
      kbest = extract_kbest(local, cancer, num_genes)

      with open(cancer + '_kbest.pickle', 'wb') as fp:
        pickle.dump(kbest, fp, pickle.HIGHEST_PROTOCOL)


    # KIRC
#    import pickle
#    with open('KIRC_kbest.pickle', 'rb') as fp:
#      data = pickle.load(fp)
#    d = [x[:-1, 0] for _, x in data]
#    d2 = []
#    for i in range(100):
#      d2.append([])
#      for j in range(10):
#        idx = i * 10 + j
#        length = lengths[i][j]
#        d2[i].append(d[idx][:length])
##    d2[i] = np.concatenate(d2[i]) # not allow reduction
#    del d
#    d2 = np.asarray(d2)
#
#    # for each fold I have to filter the Kbest according to the Dnet
#    # signature dimension! Then compute the overlap
