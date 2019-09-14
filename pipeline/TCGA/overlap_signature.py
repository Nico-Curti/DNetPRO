#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import argparse
import pickle
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import matplotlib.patches as mpatches # pretty labels

sns.set_context('paper', font_scale=2)

__package__ = 'DNetPRO signature overlap'
__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']


def parse_args ():

  description = 'DNetPRO signature overlap over TCGA datasets'

  parser = argparse.ArgumentParser(description=description)
  parser.add_argument('--filename',
                      dest='filename',
                      required=True,
                      type=str,
                      action='store',
                      help='Signature filename (output of TCGA dataset processing)'
                      )
  parser.add_argument('--dim',
                      dest='dimension',
                      required=False,
                      type=int,
                      action='store',
                      help='Number of genes in the original dataset',
                      default=0
                      )

  args = parser.parse_args()

  if args.dimension == 0:
    if 'KIRC' in args.filename:
      args.dimension = 20530
    elif 'GBM' in args.filename:
      args.dimension = 17814
    elif 'LUSC' in args.filename:
      args.dimension = 20530
    elif 'OV' in args.filename:
      args.dimension = 17814
    else:
      raise ValueError('Total number of genes not found in default template. Please give it in command line')

  return args


def get_cancer (filename):
  if 'KIRC' in filename:
    return 'KIRC'
  elif 'GBM' in filename:
    return 'GBM'
  elif 'LUSC' in filename:
    return 'LUSC'
  elif 'OV' in filename:
    return 'OV'
  else:
    raise ValueError('Cancer type not in default template.')


def extract_kbest (cancer, lengths):

  with open(cancer + '_kbest.pickle', 'rb') as fp:
    data = pickle.load(fp)

  genes = [x[:-1] for _, x in data]
  signatures = []
  for i in range(100):
    signatures.append([])
    for j in range(10):
      idx = i * 10 + j
      length = lengths[i][j]
      signatures[i].append(genes[idx][:length])

  return signatures


if __name__ == '__main__':

  #args = parse_args()

  filename = 'KIRC_new_mRNA.best'#args.filename
  number_of_genes = 20530#args.dimension

  # read the data but focused only on interesting columns
  data = pd.read_csv(filename, sep=',', header=0,
                     usecols=['signatures_nodes',
                              'lengths_signatures'
                              ])

  # split the string in list
  data['signatures_nodes']   = data.signatures_nodes.apply(lambda x : [list(map(int, y.split(';'))) for y in x.split('|')])
  data['lengths_signatures'] = data.lengths_signatures.apply(lambda x : list(map(int, x.split(';'))))

  lengths = np.asarray(data.lengths_signatures.values.tolist())
  signature_genes = np.asarray(data.signatures_nodes.values.tolist())

  # data is useless now
  del data

  number_of_iteration, number_of_fold = np.shape(lengths)

  # global mean and median over the full signatures list
  mean_length_signature = np.mean(lengths)
  median_length_signature = np.median(lengths)

  # print some statistics
  print('Minimum Size Signature: {:d}'.format(np.min(lengths)))
  print('Maximum Size Signature: {:d}'.format(np.max(lengths)))
  print('Average Size Signature: {:.3f}'.format(mean_length_signature))
  print('Median  Size Signature: {:.3f}'.format(median_length_signature))

  # Read and process Kbest data
  cancer = get_cancer(filename)
  kbest_genes = extract_kbest(cancer, lengths)

  # Occurrences of genes in each signature over the number of iterations
  genes = np.zeros(shape=(number_of_iteration, number_of_genes), dtype=float)

  # Nested for loops for add-reduction
  for i, signatures in enumerate(signature_genes):
    for signature in signatures:
      genes[i][signature] += 1
  # remove all zeros columns
  genes = genes[:, ~np.all(genes == 0, axis=0)] * .1
  genes_occ = genes.sum(axis=0)
  size_genes = len(set(genes_occ))

  # "Null-model" with random gene extraction of the same lengths as extracted signatures
  random = np.zeros(shape=(number_of_iteration, number_of_genes), dtype=float)

  # Nested for loops for add-reduction
  for i, signatures in enumerate(signature_genes):
    for signature in signatures:
      random_genes = np.random.randint(low=0, high=number_of_genes, size=(len(signature),))
      random[i][random_genes] += 1
  # remove all zeros columns
  random = random[:, ~np.all(random == 0, axis=0)] * .1
  random_occ = random.sum(axis=0)
  size_rng = len(set(random_occ))

  # "Kbest" signatures of the same lengths as extracted signatures
  KBest = np.zeros(shape=(number_of_iteration, number_of_genes), dtype=float)

  # Nested for loops for add-reduction
  for i, signatures in enumerate(kbest_genes):
    for signature in signatures:
      KBest[i][signature] += 1
  # remove all zeros columns
  KBest = KBest[:, ~np.all(KBest == 0, axis=0)] * .1
  KBest_occ = KBest.sum(axis=0)
  size_kbest = len(set(KBest_occ))

  bins = min(size_genes, size_rng)
  bins = min(bins, size_kbest)

  colors = {'DNetPRO' : sns.xkcd_rgb['denim blue'],
            'Random'  : sns.xkcd_rgb['lime'],
            'KBest'   : sns.xkcd_rgb['pale red']
           }
#%%
  cum =0# -1

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
  ax.hist(random_occ, bins=size_rng,
          cumulative=cum, alpha=.5, histtype='stepfilled',
          color=colors['Random'])
  ax.hist(KBest_occ, bins=size_kbest,
          cumulative=cum, alpha=.5, histtype='stepfilled',
          color=colors['KBest'])
  ax.hist(genes_occ, bins=size_genes,
          cumulative=cum, alpha=.5, histtype='stepfilled',
          color=colors['DNetPRO'])

  # set x limits
  ax.set_xlim(100, 1)
  ax.semilogy()

  # set axes labels
  ax.set_xlabel('# iterations', fontsize=24)
  ax.set_ylabel('Cumulative occurrences', fontsize=24)

  # pretty labels
  labels = [ mpatches.Patch(facecolor=colors['DNetPRO'], label='DNetPRO', edgecolor='k', linewidth=2),
             mpatches.Patch(facecolor=colors['Random'],  label='Random', edgecolor='k', linewidth=2),
             mpatches.Patch(facecolor=colors['KBest'],   label='KBest', edgecolor='k', linewidth=2)
           ]

  # add legend
  ax.legend(handles=labels,
            fontsize=24,
            loc='upper right',
            prop={'weight' : 'semibold',
                  'size':24}
            )

  # despine axis
  sns.despine(ax=ax, offset=10, top=True, right=True, bottom=False, left=False)

  fig.tight_layout()
