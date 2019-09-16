#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import pandas as pd
import pylab as plt
import seaborn as sns
import matplotlib.patches as mpatches

__package__ = 'DNetPRO toy analysis'
__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']


if __name__ == '__main__':

  simulation = pd.read_csv('DNetPRO_toy_test.dat', sep=',', header=0)

  if 0:
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    ax1.scatter(simulation.dnet_score, simulation.kbest_score,
                marker='o', s=200,
                edgecolors='k', facecolors='none',
                linewidth=2, alpha=.5)

    ax1.plot([0, 1], [0, 1], 'r', linewidth=2, linestyle='dashed', alpha=.5)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax1.set_xlabel('DNetPRO score', fontsize=24)
    ax1.set_ylabel('K-best score', fontsize=24)


  #%%

  N = len(simulation)
  db = pd.concat([simulation, simulation])
  db['algorithm'] = 'kbest'
  db.algorithm[:N] = 'DNetPRO'
  db['score'] = db.kbest_score
  db.score[:N] = db.dnet_score[:N]

  #%%

  sns.set_context('paper', font_scale=2)

  palette = sns.color_palette(['forestgreen', 'gold'])

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
  fig.subplots_adjust(left=0.15, right=0.9, top=0.8,  bottom=0.2)

  box = sns.boxplot(x='samples',
                    y='score',
                    hue='algorithm',
                    data=db[(db.features == 40000)],
                    palette=palette,
                    ax=ax,
                    notch=False,
                    saturation=.75,
                    linewidth=3,
                    # split=True
                    )
  for i,artist in enumerate(box.artists):
    line = box.lines[i*6 + 4]
    line.set_color('k')
    line.set_linewidth(8)

  ax.hlines(.5, -0.5, len(db), colors='r', linestyle='dashed',
            alpha=.5, linewidth=4)

  ax.hlines(.5, -0.5, len(db), colors='r', linestyle='dashed',
            alpha=.5, linewidth=4)
  labels = [ mpatches.Patch(facecolor='forestgreen', label='DNetPRO', edgecolor='k', linewidth=2),
             mpatches.Patch(facecolor='gold',        label='K-best',  edgecolor='k', linewidth=2)
           ]

  # add legend
  ax.legend(handles=labels,
            fontsize=24,
            loc='best',
            prop={'weight' : 'semibold',
                  'size':24},
            # bbox_to_anchor=(1.05, 1.25)
            )

  # set axes labels
  ax.set_ylabel('Accuracy', fontsize=24)
  ax.set_xlabel('# of Samples', fontsize=24)

  ax.set_ylim(.6, 1)

  sns.despine(ax=ax, offset=10, top=True, right=True, bottom=False, left=False)

  # fig.tight_layout()

#  fig.savefig('./samples_toy.svg', bbox_inches='tight')

  #%%

  sns.set_context('paper', font_scale=2)

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(14, 8))
  fig.subplots_adjust(left=0.15, right=0.9, top=0.8,  bottom=0.2)

  # violinplot goes over 1
  box = sns.boxplot(x='features',
                    y='score',
                    hue='algorithm',
                    data=db[(db.samples  == 500) & (
                            (db.features == 10000) |
                            (db.features == 20000) |
                            (db.features == 30000) |
                            (db.features == 40000) #|
#                            (db.features == 5000) |
#                            (db.features == 6000) |
#                            (db.features == 7000) |
#                            (db.features == 8000) |
#                            (db.features == 9000)
                            )
                            ],
                    palette=palette,
                    ax=ax,
                    notch=False,
                    saturation=.75,
                    linewidth=3,
                    # split=True,
                    # inner='quartile'
                    )
  for i,artist in enumerate(box.artists):
    line = box.lines[i*6 + 4]
    line.set_color('k')
    line.set_linewidth(8)

  ax.hlines(.5, -0.5, len(db), colors='r', linestyle='dashed',
            alpha=.5, linewidth=4)
  labels = [ mpatches.Patch(facecolor='forestgreen', label='DNetPRO', edgecolor='k', linewidth=2),
             mpatches.Patch(facecolor='gold',        label='K-best',  edgecolor='k', linewidth=2)
           ]

  # add legend
  ax.legend(handles=labels,
            fontsize=24,
            loc='upper right',
            prop={'weight' : 'semibold',
                  'size':24},
            bbox_to_anchor=(1.05, 1.25)
            )

  # set axes labels
  ax.set_ylabel('Accuracy', fontsize=24)
  ax.set_xlabel('# of Features', fontsize=24)

  ax.set_ylim(.6, 1)

  sns.despine(ax=ax, offset=10, top=True, right=True, bottom=False, left=False)

  # fig.tight_layout()

#  fig.savefig('./features_toy.svg', bbox_inches='tight')

  #%%

  if 0:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))

    ax.scatter(simulation.dnet_informative  / simulation.informative,
               simulation.kbest_informative / simulation.informative,
               marker='o', s=200,
               edgecolors='k', facecolors='none',
               linewidth=2, alpha=.5)

    ax.plot([0, 1], [0, 1], 'r', linewidth=2, linestyle='dashed', alpha=.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel('DNetPRO informative', fontsize=24)
    ax.set_ylabel('K-best informative', fontsize=24)

