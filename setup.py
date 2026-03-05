#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import sys
from pathlib import Path

from skbuild import setup
from setuptools import find_packages


def _load_pyproject() -> dict:
  path = Path(__file__).resolve().parent / 'pyproject.toml'
  data = path.read_bytes()
  try:
    import tomllib  # py>=3.11
    return tomllib.loads(data.decode('utf-8'))
  except ModuleNotFoundError:
    import tomli  # py<3.11 (build-system requires)
    return tomli.loads(data.decode('utf-8'))


pp = _load_pyproject()
proj = pp.get('project', {}) or {}

PACKAGE_VERSION = str(proj.get('version', '0.0.0'))
DESCRIPTION = str(proj.get('description', 'DNetPRO'))

# scripts declared in pyproject.toml: [project.scripts]
scripts = proj.get('scripts', {}) or {}
console_scripts = [f'{name}={target}' for name, target in scripts.items()]

# user-controlled OpenMP
USE_OMP = os.environ.get('DNETPRO_OMP', '0').lower() in ('1', 'true', 'on', 'yes')

# Forward core metadata to CMake (single source = pyproject)
cmake_args = [
  '-DSKBUILD:BOOL=ON',
  f"-DOMP:BOOL={'ON' if USE_OMP else 'OFF'}",
  '-DBUILD_DOCS:BOOL=OFF',
  '-DFORCE_BUILD_SUBMODULES:BOOL=OFF',
  f'-DDNetPRO_VERSION:STRING={PACKAGE_VERSION}',
  f'-DCMAKE_PROJECT_DESCRIPTION:STRING={DESCRIPTION}',
  f'-DPython3_EXECUTABLE:FILEPATH={sys.executable}',
]

# optional split fields (if you still generate version.h from them)
parts = (PACKAGE_VERSION.split('.') + ['0', '0', '0'])[:3]
cmake_args += [
  f'-DDNetPRO_MAJOR:STRING={parts[0]}',
  f'-DDNetPRO_MINOR:STRING={parts[1]}',
  f'-DDNetPRO_REVISION:STRING={parts[2]}',
]

setup(
  version=PACKAGE_VERSION,
  description=DESCRIPTION,
  packages=find_packages(include=['DNetPRO', 'DNetPRO.*']),
  include_package_data=True,
  platforms='any',
  entry_points={'console_scripts': console_scripts} if console_scripts else {},
  cmake_install_dir='',
  cmake_args=cmake_args,
)
