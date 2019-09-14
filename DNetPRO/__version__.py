#!/usr/bin/env python
# -*- coding: utf-8 -*-

__package__ = 'version'
__author__  = ['Nico Curti',
               'Enrico Giampieri',
               'Daniel Remondini'
               ]

__email__ = ['nico.curit2@unibo.it',
             'enrico.giampieri@unibo.it',
             'daniel.remondini@unibo.it'
             ]

from DNetPRO.build import read_version

VERSION = read_version('./CMakeLists.txt')

__version__ = '.'.join(map(str, VERSION))
