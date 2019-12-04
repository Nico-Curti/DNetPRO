#!/usr/bin/env python
# -*- coding: utf-8 -*-

from DNetPRO.build import read_version

__author__  = ['Nico Curti', 'Enrico Giampieri', 'Daniel Remondini']
__email__ = ['nico.curit2@unibo.it', 'enrico.giampieri@unibo.it', 'daniel.remondini@unibo.it']

VERSION = read_version('./CMakeLists.txt')

__version__ = '.'.join(map(str, VERSION))
