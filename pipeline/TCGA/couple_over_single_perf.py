#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

__package__ = 'DNetPRO signature couples-perf over single'
__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']


if __name__ == '__main__':

  filename = 'binary_train_0_0'
  number_of_genes = 20530

  data_type = [('genea', np.uint16), ('geneb', np.uint16), ('c0', np.uint16), ('c1', np.uint16), ('ct', np.uint16)]
  columns = list(x[0] for x in data_type)

  couples = np.fromfile(filename, dtype=data_type).tolist()
  couples = pd.DataFrame(couples, columns=columns)

  singles = couples.loc[:2053-1]
  couples = couples.loc[2053:]

  nsamples = 135
  singles.ct /= nsamples
  couples.ct /= nsamples

  #%%


  merged = pd.merge(couples, singles[['genea', 'c0', 'c1', 'ct']], how='inner', on='genea')
  merged.columns = ['genea', 'geneb', 'c0_couple', 'c1_couple', 'ct_couple', 'c0_single1', 'c1_single1', 'ct_single1']
  merged = pd.merge(merged,  singles[['geneb', 'c0', 'c1', 'ct']], how='inner', on='geneb')
  merged.columns = ['genea', 'geneb', 'c0_couple', 'c1_couple', 'ct_couple', 'c0_single1', 'c1_single1', 'ct_single1', 'c0_single2', 'c1_single2', 'ct_single2']

#  merged.fillna(value=0.5*135, inplace=True)

#%%
  merged['inv_ct_couples'] = 1. - merged.ct_couple
  merged['inv_ct_single1'] = 1. - merged.ct_single1
  merged['inv_ct_single2'] = 1. - merged.ct_single2

#%%

  merged['weights'] = (merged[['inv_ct_single1', 'inv_ct_single2']].min(axis=1) / merged.inv_ct_couples
                      ) * merged.ct_couple#merged[['inv_ct_single1', 'inv_ct_single2']].max(axis=1)
  merged.sort_values(by='weights', inplace=True, ascending=False)


#%%
#
#  import networkx as nx
#  from operator import itemgetter
#
#  def _pendrem (graph):
#    '''
#    Remove pendant node iterativelly
#    '''
#    deg = graph.degree()
#    while min(map(itemgetter(1), deg)) < 2:
#      graph.remove_nodes_from( [n for n, d in deg if d < 2] )
#      deg = graph.degree()
#      if len(deg) == 0: break
#    return graph
#
#  size = 50
#  len_couple = []
#  len_merged = []
#  overlap = []
#  for size in range(10, 1000, 20):
#    couples_best = couples[['genea', 'geneb']].iloc[:size].values
#    merged_best  = merged[['genea', 'geneb']].iloc[:size].values
#
#    G_couples = nx.from_edgelist(couples_best)
#    G_merged  = nx.from_edgelist(merged_best)
#
#    G_couples = _pendrem(G_couples)
#    G_merged  = _pendrem(G_merged)
#
#    len_couple.append(len(G_couples.nodes()))
#    len_merged.append(len(G_merged.nodes()))
#
#    overlap.append(len(set(G_couples) & set(G_merged)))#/max(len(G_couples), len(G_merged) )
#
#  plt.plot(len_couple)
#  plt.plot(len_merged)
#  plt.plot(overlap)

#  import pylab as plt
#
#  fig, (ax1, ax2) = plt.subplots(1, 2)
#  nx.draw_networkx(G_couples, ax=ax1)
#  nx.draw_networkx(G_merged, ax=ax2)
#  pos = nx.spring_layout(G_merged)
#  nx.draw_networkx_nodes(G_merged, pos=pos, node_color='blue')
#  nx.draw_networkx_edges(G_merged, pos=pos)
