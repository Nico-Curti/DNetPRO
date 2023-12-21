#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import utilities

import multiprocessing
from .DNetPRO import DNetPRO
from .__version__ import __version__

NTH = multiprocessing.cpu_count()

__author__  = ['Nico Curti', 'Enrico Giampieri', 'Daniel Remondini']
__email__ = ['nico.curti2@unibo.it', 'enrico.giampieri@unibo.it', 'daniel.remondini@unibo.it']

__all__ = ['NTH', 'DNetPRO', '__version__']
