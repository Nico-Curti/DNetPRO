#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import platform
import numpy as np
import multiprocessing

try:
  from setuptools import setup
  from setuptools import Extension
  from setuptools import find_packages

except ImportError:
  from distutils.core import setup
  from distutils.core import Extension
  from distutils.core import find_packages

from Cython.Distutils import build_ext
from distutils.sysconfig import customize_compiler

def get_requires (requirements_filename):
  '''
  What packages are required for this module to be executed?
  '''
  with open(requirements_filename, 'r') as fp:
    requirements = fp.read()

  return list(filter(lambda x: x != '', requirements.split()))



class dnetpro_build_ext (build_ext):
  '''
  Custom build type
  '''

  def build_extensions (self):

    customize_compiler(self.compiler)

    try:
      self.compiler.compiler_so.remove('-Wstrict-prototypes')

    except (AttributeError, ValueError):
      pass

    build_ext.build_extensions(self)


def read_description (readme_filename):
  '''
  Description package from filename
  '''

  try:

    with open(readme_filename, 'r') as fp:
      description = '\n'
      description += fp.read()

  except Exception:
    return ''




NTH = multiprocessing.cpu_count()

here = os.path.abspath(os.path.dirname(__file__))

# Package meta-data.
NAME = 'DNetPRO'
DESCRIPTION = 'Discriminant Network Processing'
URL = 'https://github.com/Nico-Curti/DNetPRO'
EMAIL = ['nico.curit2@unibo.it', 'enrico.giampieri@unibo.it', 'daniel.remondini@unibo.it']
AUTHOR = ['Nico Curti', 'Enrico Giampieri', 'Daniel Remondini']
REQUIRES_PYTHON = '>=3.4'
VERSION = None
KEYWORDS = 'feature-selection machine-learning-algorithm classification-algorithm genomics rna mirna'

CPP_COMPILER = platform.python_compiler()
README_FILENAME = os.path.join(here, 'README.md')
REQUIREMENTS_FILENAME = os.path.join(here, 'requirements.txt')
VERSION_FILENAME = os.path.join(here, 'DNetPRO', '__version__.py')

ENABLE_OMP = False

current_python = sys.executable.split('/bin')[0]
numpy_dir = current_python + '/lib/python{}.{}/site-packages/numpy/core/include'.format(sys.version_info.major, sys.version_info.minor)
if os.path.isdir(numpy_dir):
  os.environ['CFLAGS'] = '-I' + numpy_dir

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
  LONG_DESCRIPTION = read_description(README_FILENAME)

except IOError:
  LONG_DESCRIPTION = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
  with open(VERSION_FILENAME) as fp:
    exec(fp.read(), about)

else:
  about['__version__'] = VERSION

# parse version variables and add them to command line as definitions
Version = about['__version__'].split('.')

# Set compiler variables

define_args = [ '-DMAJOR={}'.format(Version[0]),
                '-DMINOR={}'.format(Version[1]),
                '-DREVISION={}'.format(Version[2]),
                '-DNUM_AVAILABLE_THREADS={}'.format(NTH),
                '-D__dnet__', # enable misc utilities
                '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'
              ]

if 'GCC' in CPP_COMPILER or 'Clang' in CPP_COMPILER:
  compile_args = ['-std=c++14', '-g0',
                  '-Wno-unused-function', # disable unused-function warnings
                  '-Wno-narrowing', # disable narrowing conversion warnings
                   # enable common warnings flags
                  '-Wall',
                  '-Wextra',
                  '-Wno-unused-result',
                  '-Wno-unknown-pragmas',
                  '-Wfatal-errors',
                  '-Wpedantic',
                  '-march=native'
                  ]

  try:

    compiler, compiler_version = CPP_COMPILER.split()

  except ValueError:

    compiler, compiler_version = (CPP_COMPILER, '0')

  if compiler == 'Clang':
    ENABLE_OMP = False

  if ENABLE_OMP:
    linker_args = ['-fopenmp']

  else:
    linker_args = []

elif 'MSC' in CPP_COMPILER:
  cpp_compiler_args = ['/std:c++latest']
  BUILD_SCORER = True

  if ENABLE_OMP:
    linker_args = ['/openmp']
  else:
    linker_args = []

else:
  raise ValueError('Unknown c++ compiler arg')

whole_compiler_args = sum([compile_args, define_args, linker_args], [])


setup(
  name                          = NAME,
  version                       = about['__version__'],
  description                   = DESCRIPTION,
  long_description              = LONG_DESCRIPTION,
  long_description_content_type = 'text/markdown',
  author                        = AUTHOR,
  author_email                  = EMAIL,
  maintainer                    = AUTHOR,
  maintainer_email              = EMAIL,
  python_requires               = REQUIRES_PYTHON,
  install_requires              = get_requires(REQUIREMENTS_FILENAME),
  url                           = URL,
  download_url                  = URL,
  keywords                      = KEYWORDS,
  packages                      = find_packages(include=['DNetPRO', 'DNetPOR.*'], exclude=('test', 'testing', 'example')),
  #include_package_data          = True, # no absolute paths are allowed
  platforms                     = 'any',
  classifiers                   =[
                                   #'License :: OSI Approved :: GPL License',
                                   'Programming Language :: Python',
                                   'Programming Language :: Python :: 3',
                                   'Programming Language :: Python :: 3.6',
                                   'Programming Language :: Python :: Implementation :: CPython',
                                   'Programming Language :: Python :: Implementation :: PyPy'
                                 ],
  license                       = 'MIT',
  cmdclass                      = {'build_ext': dnetpro_build_ext},
  ext_modules                   = [Extension( name='.'.join(['lib', 'DNetPRO', 'DNetPRO']),
                                              sources=['./DNetPRO/source/DNet_couples.pyx',
                                                       './src/dnetpro_couples.cpp',
                                                       './src/score.cpp',
                                                       './src/utility.cpp'
                                                       ],
                                              include_dirs=[ './DNetPRO/include','./hpp/', './include/'],
                                              libraries=[],
                                              library_dirs=[
                                                            os.path.join(here, 'lib'),
                                                            os.path.join('usr', 'lib'),
                                                            os.path.join('usr', 'local', 'lib'),
                                                            np.get_include(),
                                              ],  # path to .a or .so file(s)
                                              extra_compile_args = whole_compiler_args,
                                              extra_link_args = [],
                                              language='c++'
                                              )
  ],
)

