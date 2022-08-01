# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys

sys.path.insert(0, '/home/ncurti/Code/DNetPRO')

# -- Project information -----------------------------------------------------

project = 'DNetPRO - Discriminant Network Processing - Feature Selection'
copyright = '2022, Nico Curti'
author = 'Nico Curti'

# The full version, including alpha/beta/rc tags
release = '1.0.0'

master_doc = 'index'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.intersphinx',
              'sphinx_rtd_theme',
              #'rst2pdf.pdfbuilder',
              'breathe',
              'nbsphinx',
              'IPython.sphinxext.ipython_console_highlighting',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# -- Options for PDF output --------------------------------------------------

# Grouping the document tree into LaTeX files. List of tuples# (source start file, target name, title, author, documentclass [howto/manual]).
latex_engine = 'xelatex'
latex_documents = [('index', 'DNetPRO.tex', u'DNetPRO - Discriminant Network Processing - Feature Selection', u'Nico Curti', 'manual'),]
latex_show_pagerefs = True
latex_domain_indices = False

pdf_documents = [('index', u'DNetPRO', u'DNetPRO - Discriminant Network Processing - Feature Selection', u'Nico Curti', 'DNetPRO - Discriminant Network Processing - Feature Selection'),]


nbsphinx_input_prompt = 'In [%s]:'
nbsphinx_kernel_name = 'python3'
nbsphinx_output_prompt = 'Out[%s]:'


breathe_projects = {
  'score' : '/home/ncurti/Code/DNetPRO/docs/source//',
  'DNetPRO' : '/home/ncurti/Code/DNetPRO/docs/source//',
  'utils' : '/home/ncurti/Code/DNetPRO/docs/source//',
  }
