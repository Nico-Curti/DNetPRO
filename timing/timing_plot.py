#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import pandas as pd
import seaborn as sns
import pylab as plt

__package__ = 'DNetPRO Timing plots'
__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']


if __name__ == '__main__':

  filename = 'DNetPRO_timing.dat'
  data = pd.read_csv(filename, sep=',', header=0)
  N = len(data)
  data = pd.concat([data, data])
  data['types'] = 'python'
  data.types[:N] = 'C++'
  data['times'] = data.py_min
  data.times[:N] = data.cpp_min[:N]

  #%%

  fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

  # hide the spines between ax and ax2
  ax1.spines['bottom'].set_visible(False)
  ax2.spines['top'].set_visible(False)
  ax1.xaxis.tick_top()
  ax1.tick_params(labeltop='off')  # don't put tick labels at the top
  ax2.xaxis.tick_bottom()
  d = .005  # how big to make the diagonal lines in axes coordinates
  # arguments to pass to plot, just so we don't keep repeating them
  kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
  ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
  kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
  ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

  sns.despine(ax=ax1, offset=1, top=True, right=True, bottom=True,  left=False)
  sns.despine(ax=ax2, offset=1, top=True, right=True, bottom=False, left=False)


  down = sns.barplot(x='nth',
                     y='times',
                     hue='types',
                     data=data[(data.samples == 90) & (data.features == 90)],
                     ax=ax2,
                     )

  ax2.set(yscale='log')
  down.set(ylabel='', xlabel='')

  for i, bar in enumerate(down.patches):
    bar.set_edgecolor((0., 0., 0., 0.25))
  l = ax2.legend()
  l.set_visible(False)

  up = sns.barplot(x='nth',
                   y='times',
                   hue='types',
                   data=data[(data.samples == 90) & (data.features == 90)],
                   ax=ax1,
                   )
  up.set(xlabel='', ylabel='')
  ax1.set(yscale='log')

  ax2.set_ylabel('Time (sec.)', multialignment='center', fontsize=24)
  ax2.yaxis.set_label_coords(-0.075, 1.3)
  ax2.set_xticklabels(labels=data.nth.unique())#, rotation=45)

  ax2.set_xlabel('Number of Threads', multialignment='center', fontsize=24)
  ax1.axes.get_xaxis().set_visible(False)

  l = ax1.legend(fontsize='x-large')
  l.set_title('Language', prop={'size':'xx-large'})

  for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(16)

  for tick in ax2.yaxis.get_minor_ticks():
    tick.label.set_fontsize(16)
  for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(16)
  for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(16)

  ax1.set_ylim(100, 10000)
  ax2.set_ylim(1e-2, 0.1)

  fig.savefig('./nth_timing.svg', bbox_inches='tight')

  #%%

  fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

  # hide the spines between ax and ax2
  ax1.spines['bottom'].set_visible(False)
  ax2.spines['top'].set_visible(False)
  ax1.xaxis.tick_top()
  ax1.tick_params(labeltop='off')  # don't put tick labels at the top
  ax2.xaxis.tick_bottom()
  d = .005  # how big to make the diagonal lines in axes coordinates
  # arguments to pass to plot, just so we don't keep repeating them
  kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
  ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
  kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
  ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

  sns.despine(ax=ax1, offset=1, top=True, right=True, bottom=True,  left=False)
  sns.despine(ax=ax2, offset=1, top=True, right=True, bottom=False, left=False)


  down = sns.barplot(x='samples',
                     y='times',
                     hue='types',
                     data=data[(data.nth == 32) & (data.features == 90)],
                     ax=ax2,
                     )

  ax2.set(yscale='log')
  down.set(ylabel='', xlabel='')

  for i, bar in enumerate(down.patches):
    bar.set_edgecolor((0., 0., 0., 0.25))
  l = ax2.legend()
  l.set_visible(False)

  up = sns.barplot(x='samples',
                   y='times',
                   hue='types',
                   data=data[(data.nth == 32) & (data.features == 90)],
                   ax=ax1,
                   )
  up.set(xlabel='', ylabel='')
  ax1.set(yscale='log')

  ax2.set_ylabel('Time (sec.)', multialignment='center', fontsize=24)
  ax2.yaxis.set_label_coords(-0.075, 1.)
  ax2.set_xticklabels(labels=data.samples.unique())#, rotation=45)

  ax2.set_xlabel('Number of Samples', multialignment='center', fontsize=24)
  ax1.axes.get_xaxis().set_visible(False)

  l = ax1.legend(fontsize='x-large')
  l.set_title('Language', prop={'size':'xx-large'})

  for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(16)

  for tick in ax2.yaxis.get_minor_ticks():
    tick.label.set_fontsize(16)
  for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(16)
  for tick in ax1.yaxis.get_minor_ticks():
    tick.label.set_fontsize(16)
  for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(16)

  ax1.set_ylim(10, 1000)
  ax2.set_ylim(0, 0.1)

  fig.savefig('./samples_timing.svg', bbox_inches='tight')


  #%%

  fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

  # hide the spines between ax and ax2
  ax1.spines['bottom'].set_visible(False)
  ax2.spines['top'].set_visible(False)
  ax1.xaxis.tick_top()
  ax1.tick_params(labeltop='off')  # don't put tick labels at the top
  ax2.xaxis.tick_bottom()
  d = .005  # how big to make the diagonal lines in axes coordinates
  # arguments to pass to plot, just so we don't keep repeating them
  kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
  ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
  kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
  ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

  sns.despine(ax=ax1, offset=1, top=True, right=True, bottom=True,  left=False)
  sns.despine(ax=ax2, offset=1, top=True, right=True, bottom=False, left=False)


  down = sns.barplot(x='features',
                     y='times',
                     hue='types',
                     data=data[(data.nth == 32) & (data.samples == 90)],
                     ax=ax2,
                     )

  ax2.set(yscale='log')
  down.set(ylabel='', xlabel='')

  for i, bar in enumerate(down.patches):
    bar.set_edgecolor((0., 0., 0., 0.25))
  l = ax2.legend()
  l.set_visible(False)

  up = sns.barplot(x='features',
                   y='times',
                   hue='types',
                   data=data[(data.nth == 32) & (data.samples == 90)],
                   ax=ax1,
                   )
  up.set(xlabel='', ylabel='')
  ax1.set(yscale='log')

  ax2.set_ylabel('Time (sec.)', multialignment='center', fontsize=24)
  ax2.yaxis.set_label_coords(-0.075, 1.)
  ax2.set_xticklabels(labels=data.features.unique())#, rotation=45)

  ax2.set_xlabel('Number of Features', multialignment='center', fontsize=24)
  ax1.axes.get_xaxis().set_visible(False)

  l = ax1.legend(fontsize='x-large')
  l.set_title('Language', prop={'size':'xx-large'})

  for tick in ax2.xaxis.get_major_ticks():
    tick.label.set_fontsize(16)

  for tick in ax2.yaxis.get_minor_ticks():
    tick.label.set_fontsize(16)
  for tick in ax2.yaxis.get_major_ticks():
    tick.label.set_fontsize(16)
  for tick in ax1.yaxis.get_minor_ticks():
    tick.label.set_fontsize(16)
  for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(16)

  ax1.set_ylim(5, 1000)
  ax2.set_ylim(0, 0.1)

  fig.savefig('./features_timing.svg', bbox_inches='tight')
