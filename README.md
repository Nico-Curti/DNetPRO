| **Authors**  | **Project** | **Build Status** | **Code Quality** | **Coverage** |
|:------------:|:-----------:|:----------------:|:----------------:|:------------:|
| [**N. Curti**](https://github.com/Nico-Curti) |  **DNetPRO**<br/>[![arxiv.org](http://img.shields.io/badge/bioRxiv-0.1101/773622-B31B1B.svg)](https://www.biorxiv.org/content/10.1101/773622v1)  | **Linux/MacOS** : [![travis](https://travis-ci.com/Nico-Curti/DNetPRO.svg?branch=master)](https://travis-ci.com/Nico-Curti/DNetPRO) <br/> **Windows** : [![appveyor](https://ci.appveyor.com/api/projects/status/tbjbqnr3am9xms1a?svg=true)](https://ci.appveyor.com/project/Nico-Curti/dnetpro) | **Codacy** : [![Codacy](https://api.codacy.com/project/badge/Grade/f3ff6cf583d4474e988d33923137e184)](https://www.codacy.com/manual/Nico-Curti/DNetPRO?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Nico-Curti/DNetPRO&amp;utm_campaign=Badge_Grade) <br/> **Codebeat** : [![Codebeat](https://codebeat.co/badges/0c5129c1-4537-4545-a8e2-07807c6303f5)](https://codebeat.co/projects/github-com-nico-curti-dnetpro-master) | [![codecov](https://codecov.io/gh/Nico-Curti/DNetPRO/branch/master/graph/badge.svg)](https://codecov.io/gh/Nico-Curti/DNetPRO) |

[![C++ version CI](https://github.com/Nico-Curti/DNetPRO/workflows/C++%20version%20CI/badge.svg)](https://github.com/Nico-Curti/DNetPRO/actions?query=workflow%3A%22C%2B%2B+version+CI%22)

[![Python version CI](https://github.com/Nico-Curti/DNetPRO/workflows/Python%20version%20CI/badge.svg)](https://github.com/Nico-Curti/DNetPRO/actions?query=workflow%3A%22Python+version+CI%22)

[![Docs CI](https://github.com/Nico-Curti/DNetPRO/workflows/DNetPRO%20Docs%20CI/badge.svg)](https://github.com/Nico-Curti/DNetPRO/actions?query=workflow%3A%22rFBP+Docs+CI%22)

[![docs](https://readthedocs.org/projects/dnetpro/badge/?version=latest)](https://dnetpro.readthedocs.io/en/latest/?badge=latest)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/Nico-Curti/DNetPRO.svg?style=plastic)](https://github.com/Nico-Curti/DNetPRO/pulls)
[![GitHub issues](https://img.shields.io/github/issues/Nico-Curti/DNetPRO.svg?style=plastic)](https://github.com/Nico-Curti/DNetPRO/issues)

[![GitHub stars](https://img.shields.io/github/stars/Nico-Curti/DNetPRO.svg?label=Stars&style=social)](https://github.com/Nico-Curti/DNetPRO/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/Nico-Curti/DNetPRO.svg?label=Watch&style=social)](https://github.com/Nico-Curti/DNetPRO/watchers)

<a href="https://github.com/UniboDIFABiophysics">
  <div class="image">
    <img src="https://cdn.rawgit.com/physycom/templates/697b327d/logo_unibo.png" width="90" height="90">
  </div>
</a>

# DNetPRO

## Discriminant Analysis with Network Processing

Official implementation of the DNetPRO algorithm published on [BioRXiv](https://www.biorxiv.org/content/early/2019/09/19/773622) by Curti et al.

* [Overview](#overview)
* [Theory](#theory)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Efficiency](#efficiency)
* [Usage](#usage)
* [Testing](#testing)
* [Table of contents](#table-of-contents)
* [Contribution](#contribution)
* [References](#references)
* [Authors](#authors)
* [License](#license)
* [Acknowledgments](#acknowledgments)
* [Citation](#citation)

## Overview

![(a) An example in which single-parameter classification fails in predicting higher-dimension classification performance. Both parameters (*feature1* and *feature2*) badly classify in 1-D, but have a very good performance in 2D. Moreover, classification can be easily interpreted in terms of relative higher/lower expression of both probes. (b)Activity of a biological feature (e.g. a gene) as a function of its expression level: top) monotonically increasing, often also discretized to an on/off state; center, bottom) "windowed" behavior, in which there are two or more activity states that do not depend monotonically on expression level. X axis: expression level, Y axis, biological state (arbitrary scales).](https://github.com/Nico-Curti/DNetPRO/blob/master/img/examples.png)

Methods that select variables for multi-dimensional signatures based on single-variable performance can have limits in predicting
higher-dimensional signature performance.
As shown in Fig. [1](https://github.com/Nico-Curti/DNetPRO/blob/master/img/examples.png)(a), in which both variables taken singularly perform poorly, but their performance becomes optimal in a 2-dimensional combination, in terms of linear separation of the two classes.

It is known that complex separation surfaces characterize classification tasks associated to image and speech recognition, for which Deep Networks are used successfully in recent times, but in many cases biological data, such as gene or protein expression, are more likely characterized by a up/down-regulation behavior (as shown in Fig. [1](https://github.com/Nico-Curti/DNetPRO/blob/master/img/examples.png)(b) top), while more complex behaviors (e.g. a optimal range of activity, Fig. [1](https://github.com/Nico-Curti/DNetPRO/blob/master/img/examples.png)(b) bottom) are much less likely.
Thus, discriminant-based methods (and logistic regression methods alike) can very likely provide good classification performances in these cases (as demonstrated by our results with DNetPRO) if applied in at least
two-dimensional spaces.
Moreover, the of these methods (that generate very simple class separation surfaces, i.e. linear or quadratic) guarantee that a of a signature based on lower-dimensional signatures is feasible.

This consideration are relevant in particular for microarray data where we face on a small number of samples compared to a huge amount of variables (gene probes).
This kind of problem, often called problem (where `N` is the number of features, i.e variables, and `S` is the number of samples), tend to be prone to overfitting and they are classified to ill-posed.
The difficulty on the feature extraction can also increase due to noisy variables that can drastically affect the machine learning algorithms.
Often is difficult to discriminate between noise and significant variables and even more as the number of variables rises.

In this project we propose a new method of features selection - DNetPRO, *Discriminant Analysis with Network PROcessing* - developed to outperform the mentioned above problems.
The method is particularly designed to gene-expression data analysis and it was tested against the most common feature selection techniques.
The method was already applied on gene-expression datasets but my work focused on the benchmark of it and on its optimization for Big Data applications.
The pipeline algorithm is made by many different steps and only a part of it was designed to biological application: this allow me to apply (part of) the same techniques also in different kind of problems with good results (see [[Mizzi2018](https://doi.org/10.1140/epjds/s13688-018-0168-2)]).

## Theory

The `DNetPRO` algorithm produces multivariate signatures starting from all the couples of variables analyzed by a Discriminant Analysis.
For this reason, it can be classified as a combinatorial method and the computational time for the exploration of variable space is proportional to the square of the number of underlying variables (ranging from `10^3` to `10^5` in a typical high-throughput omics study).
This behavior allows to overcome some of the limits of single-feature selection methods and it provides a hard-thresholding approach compared to projection-based variable selection methods.
The combinatorial evaluation is the most time-expensive step of the algorithm and it needs an accurate algorithmic implementation for Big Data applications.
A summary of the algorithm is shown in the following pseudo-code.

```
Data matrix (N, S) -> List of putative signatures
  Divide the data into training and test by an Hold-Out method;
  For {couple <- (feature_1, feature_2) in Couples}{
    Leave-One-Out cross validation;
    Score estimation using a Classifier;
  }
  Sorting of the couples in ascending order according to their score;
  Threshold over the couples score (K-best couples);
  For {component in connected_components}{
    If {reduction}{
      Iteratively pendant node remotion;
    }
    Signature evaluation using a Classifier;
  }
```

So, given an initial dataset, with `S` *samples* (e.g. cells or patients) each one described by $N$ observations (our *variables*, e.g. gene or protein expression profiles), the signature identification can be summarized with the following steps:

- separation of available data into a training and test sets (e.g. 33/66, or 20/80);

- estimation of the classification performance on the training set of all `S(S-1)/2` variable couples through a computationally fast and reproducible cross-validation procedure (leave-one-out cross validation was chosen);

- selection of top-performing couples through a hard-thresholding procedure.
  The performance of each couple constitutes a *weighted link* of a network, where the nodes are the variables connected at least through one link;

- every *connected component* which composes the network identifies a putative signature.

- (optional) in order to reduce the size of an identified signature, the pendant nodes of the network (i.e. nodes with degree equal to one) can be removed, in a single step or recursively up to the core network (i.e. a network with all nodes with at least two links).

- all signatures are evaluated onto the test set to estimate their performances.

- a further cross validation step is performed (with a further dataset splitting into test and validation sets) to identify the best performing signature.

We would stress that this method is completely independent to the choose of the classification algorithm, but, from a biological point-of-view, a simple one is preferable to keep an easy interpretability of the results.
The geometrical simplicity of the resulting class-separation surfaces, in fact, allows an easier interpretation of the results, as compared to very powerful but black-box methods like nonlinear-kernel SVM or Neural Networks.
These are the reasons which lead us to use very simple classifier methods in our biological applications as diag-quadratic Discriminant Analysis or Quadratic Discriminant Analysis.
Both these methods allow fast computation and an easy interpretation of the results.
A linear separation might not be common in some classification problems (e.g. image classification), but it is very likely in biological systems, where many responses to perturbation consist in an increase or decrease of variable values (e.g. expression of genes or proteins, see Fig.[1](https://github.com/Nico-Curti/DNetPRO/blob/master/img/expression.svg)(b)).

A second direct gain by the couples evaluation is related to the network structure: the `DNetPRO` network signatures allow a hierarchical ranking of features according to their centrality compared to other methods.
The underlying network structure of the signature could suggests further methods to improve its dimensionality reduction based on network topological properties to fit real application needs, and it could help to evaluate the cooperation of variables for the class identification.

In the end, we remark that our signatures have a purely statistical relevance by being generated with a purpose of maximal classification performance, but sometimes the selected features (e.g. genes, DNA loci, metabolites) can be of clinical and biological interest, helping to improve knowledge on the mechanism associated to the studied phenomenon [[PMrna](https://genome.cshlp.org/content/early/2013/10/02/gr.155192.113.abstract), [Scotlandi2009](https://doi.org/10.1200/JCO.2008.19.2542), [PMgene](https://www.ncbi.nlm.nih.gov/pubmed/26297486), [Terragna](https://www.ncbi.nlm.nih.gov/pubmed/26575327)].

## Prerequisites

C++ supported compilers:

![gcc version](https://img.shields.io/badge/gcc-4.8.5%20|%204.9.*%20|%205.*%20|%206.*%20|%207.*%20|%208.*%20|%209.*-yellow.svg)

![clang version](https://img.shields.io/badge/clang-3.*%20|%204.*%20|%205.*%20|%206.*%20|%207.*%20|-red.svg)

![msvc version](https://img.shields.io/badge/msvc-vs2017%20x86%20|%20vs2017%20x64|%20vs2019%20x86%20|%20vs2019%20x64-blue.svg)

The `DNetPRO` project is written in `C++` and it supports also older standard versions (std=c++1+).
The package installation can be performed via [`CMake`](https://github.com/Nico-Curti/DNetPRO/blob/master/CMakeLists.txt).
The `CMake` installer provides also a `DNetPRO.pc`, useful if you want link to the `DNetPRO` using `pkg-config`.

You can also use the `DNetPRO` package in `Python` using the `Cython` wrap provided inside this project.
The only requirements are the following:

* numpy >= 1.14.3
* networkx >= 2.2
* cython >= 0.29
* scikit -learn >= 0.19.1
* pandas >= 0.24.2

The `Cython` version can be built and installed via `CMake` enabling the `-DPYWRAP` variable.
You can use also the `DNetPRO` package in `Python` using the `Cython` wrap provided inside this project.
The `Python` wrap guarantees also a good integration with the other common Machine Learning tools provided by `scikit-learn` `Python` package; in this way you can use the `DNetPRO` algorithm as an equivalent alternative also in other pipelines.
Like other Machine Learning algorithm also the `DNetPRO` one depends on many parameters, i.e its hyper-parameters, which has to be tuned according to the given problem.
The `Python` wrap of the library was written according to `scikit-optimize` `Python` package to allow an easy hyper-parameters optimization using the already implemented classical methods.

## Installation

1) Follow your system prerequisites (below)

2) Clone the `DNetPRO` package from this repository, or download a stable release

```bash
git clone https://github.com/Nico-Curti/DNetPRO.git
cd DNetPRO
```

3) `DNetPRO` could be built with CMake and Make or with the *build* scripts in the project.
Example:

**Unix OS:**
```bash
./build.sh
```

**Windows OS:**
```Powershell
PS \>                 ./build.ps1
```

### Ubuntu

1) Define a work folder, which we will call WORKSPACE in this tutorial: this could be a "Code" folder in our home, a "c++" folder on our desktop, whatever you want. Create it if you don't already have, using your favourite method (mkdir in bash, or from the graphical interface of your distribution). We will now define an environment variable to tell the system where our folder is. Please note down the full path of this folder, which will look like `/home/$(whoami)/code/`

```bash
echo -e "\n export WORKSPACE=/full/path/to/my/folder \n" >> ~/.bashrc
source ~/.bashrc
```

2) Open a Bash terminal and type the following commands to install all the prerequisites.

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install -y gcc-8 g++-8

wget --no-check-certificate https://cmake.org/files/v3.13/cmake-3.13.1-Linux-x86_64.tar.gz
tar -xzf cmake-3.13.1-Linux-x86_64.tar.gz
export PATH=$PWD/cmake-3.13.1-Linux-x86_64/bin:$PATH

sudo apt-get install -y make git dos2unix ninja-build
git config --global core.autocrlf input
git clone https://github.com/physycom/sysconfig
```

3) Build the project with CMake (enable or disable OMP with the define **-DOMP**:

```bash
cd $WORKSPACE
git clone https://github.com/Nico-Curti/DNetPRO
cd DNetPRO

mkdir -p build
cd build

cmake ..
make -j
cmake --build . --target install
cd ..
```

### macOS

1) If not already installed, install the XCode Command Line Tools, typing this command in a terminal:

```bash
xcode-select --install
```

2) If not already installed, install Homebrew following the [official guide](https://brew.sh/index_it.html).

3) Open the terminal and type these commands

```bash
brew update
brew upgrade
brew install gcc@8
brew install cmake make git ninja
```

4) Define a work folder, which we will call WORKSPACE in this tutorial: this could be a "Code" folder in our home, a "c++" folder on our desktop, whatever you want. Create it if you don't already have, using your favourite method (mkdir in bash, or from the graphical interface in Finder). We will now define an environment variable to tell the system where our folder is. Please note down the full path of this folder, which will look like /home/$(whoami)/code/

5) Open a Terminal and type the following command (replace /full/path/to/my/folder with the previous path noted down)

```bash
echo -e "\n export WORKSPACE=/full/path/to/my/folder \n" >> ~/.bash_profile
source ~/.bash_profile
```

6) Build the project with CMake (enable or disable OMP with the define **-DOMP**:

```bash
cd $WORKSPACE
git clone https://github.com/Nico-Curti/DNetPRO
cd DNetPRO

mkdir -p build
cd build

cmake ..
make -j
cmake --build . --target install
cd ..
```

### Windows (7+)

1) Install Visual Studio 2017 from the [official website](https://www.visualstudio.com/)

2) Open your Powershell with Administrator privileges, type the following command and confirm it:

```PowerShell
PS \>                 Set-ExecutionPolicy unrestricted
```

3) If not already installed, please install chocolatey using the [official guide](http://chocolatey.org)

4) If you are not sure about having them updated, or even installed, please install `git`, `cmake` and an updated `Powershell`. To do so, open your Powershell with Administrator privileges and type

```PowerShell
PS \>                 cinst -y git cmake powershell
```

5) Restart the PC if required by chocolatey after the latest step

6) Install PGI 18.10 from the [official website](https://www.pgroup.com/products/community.htm) (the community edition is enough and is free; NOTE: install included MS-MPI, but avoid JRE and Cygwin)

7) Activate license for PGI 18.10 Community Edition (rename the file `%PROGRAMFILES%\PGI\license.dat-COMMUNITY-18.10` to `%PROGRAMFILES%\PGI\license.dat`) if necessary, otherwise enable a Professional License if available

8) Define a work folder, which we will call `WORKSPACE` in this tutorial: this could be a "Code" folder in our home, a "cpp" folder on our desktop, whatever you want. Create it if you don't already have, using your favourite method (mkdir in Powershell, or from the graphical interface in explorer). We will now define an environment variable to tell the system where our folder is. Please note down its full path. Open a Powershell (as a standard user) and type

```PowerShell
PS \>                 rundll32 sysdm.cpl,EditEnvironmentVariables
```

9) In the upper part of the window that pops-up, we have to create a new environment variable, with name `WORKSPACE` and value the full path noted down before.
If it not already in the `PATH` (this is possible only if you did it before), we also need to modify the "Path" variable adding the following string (on Windows 10 you need to add a new line to insert it, on Windows 7/8 it is necessary to append it using a `;` as a separator between other records):

```cmd
                      %PROGRAMFILES%\CMake\bin
```

10) If `vcpkg` is not installed, please follow the next procedure, otherwise please jump to #12

```PowerShell
PS \>                 cd $env:WORKSPACE
PS Code>              git clone https://github.com/Microsoft/vcpkg.git
PS Code>              cd vcpkg
PS Code\vcpkg>        .\bootstrap-vcpkg.bat
```

11) Open a Powershell with Administrator privileges and type

```PowerShell
PS \>                 cd $env:WORKSPACE
PS Code>              cd vcpkg
PS Code\vcpkg>        .\vcpkg integrate install
```

12) Open a Powershell and build `DNetPRO` using the `build.ps1` script

```PowerShell
PS \>                 cd $env:WORKSPACE
PS Code>              git clone https://github.com/Nico-Curti/DNetPRO
PS Code>              cd DNetPRO
PS Code\DNetPRO>      .\build.ps1
```

### CMake C++ installation

We recommend the use of `CMake` for the installation since it is the most automated way to reach your needs.
First of all make sure you have a sufficient version of `CMake` installed (3.9 minimum version required).
If you are working on a machine without root privileges and you need to upgrade your `CMake` version a valid solution to overcome your problems is provided [here](https://github.com/Nico-Curti/Shut).

With a valid `CMake` version installed first of all clone the project as:

```bash
git clone https://github.com/Nico-Curti/DNetPRO
cd DNetPRO
```

The you can build the `DNetPRO` package with

```bash
mkdir -p build
cd build && cmake .. && cmake --build . --target install
```

or more easily

```bash
./build.sh
```

if you are working on a Windows machine the right script to call is the [`build.ps1`](https://Nico-Curti/DNetPRO/blob/master/build.ps1).

**NOTE 1:** if you want enable the OpenMP support compile the library with `-DOMP=ON`.

**NOTE 2:** if you want enable the Cython support compile the library with `-DPYWRAP=ON`. The Cython packages will be compiled and correctly positioned in the `DNetPRO` Python package **BUT** you need to run also the setup before use it.

### Python installation

Python version supported : ![Python version](https://img.shields.io/badge/python-3.5|3.6|3.7|3.8-blue.svg)

The `Python` installation can be performed with or without the `C++` installation.
The `Python` installation is always executed using [`setup.py`](https://github.com/Nico-Curti/blob/master/setup.py) script.

If you have already build the `DNetPRO` `C++` library the installation is performed faster and the `Cython` wrap directly links to the last library installed.
Otherwise the full list of dependencies is build.

In both cases the installation steps are

```bash
python -m pip install -r ./requirements.txt
```

to install the prerequisites and then

```bash
python setup.py install
```

or for installing in development mode:

```bash
python setup.py develop --user
```

## Efficiency

As described in the above sections, the `DNetPRO` is a combinatorial algorithm and thus it requires a particular accuracy in the code implementation to optimize as much as possible the computational performances.
The theoretical optimization strategies, described up to now, have to be proved by quantitative measures.

The time evaluation was performed using the `Cython` (`C++` wrap) implementation against the pure `Python` (naive) implementation showed in the previous snippet.
The time evaluation was performed using the `timing` `Python` package in which we can easily simulate multiple runs of a given algorithm.
In our simulations, we monitored the three main parameters related to the algorithm efficiency: the number of samples, the number of variables and (as we provided a parallel multi-threading implementation) the number of threads used.
For each combination of parameters, we performed 30 runs of both algorithms and we extracted the minimum execution time.
The tests were performed on a classical bioinformatics server (128 GB RAM memory and 2 CPU E5-2620, with 8 cores each).
The obtained results are shown in following Figure.
In each plot, we fixed two variables and we evaluated the remaining one.

![Execution time of `DNetPRO` algorithm. We compare the execution time between pure-`Python` (orange) and `Cython` (blue, `C++` wrap) implementations. Execution time in function of the number of variables (the number of samples and the number of threads are kept fixed).](https://github.com/Nico-Curti/DNetPRO/blob/master/img/features_timing.svg)

![Execution time of `DNetPRO` algorithm. We compare the execution time between pure-`Python` (orange) and `Cython` (blue, `C++` wrap) implementation. Execution time in function of the number of samples (the number of variables and the number of threads are kept fixed).](https://github.com/Nico-Curti/DNetPRO/blob/master/img/samples_timing.svg)

![Execution time of `DNetPRO` algorithm. We compare the execution time between pure-`Python` (orange) and `Cython` (blue, `C++` wrap) implementation. Execution time in function of the number of threads (the number of variables and the number of samples are kept fixed).](https://github.com/Nico-Curti/DNetPRO/blob/master/img/nth_timing.svg)

In all our simulations, the efficiency of the (optimized) `Cython` version is easily visible and the gap between the two implementations reached more than `10^4` seconds.
On the other hand, it is important to highlight the scalability of the codes against the various parameters.
While the code performances scale quite well with the number of features (Fig. [1](https://github.com/Nico-Curti/DNetPRO/blob/master/img/features_timing.svg)) in both the implementations, we have a different trend varying the number of samples (Fig. [2](https://github.com/Nico-Curti/DNetPRO/blob/master/img/samples_timing.svg)): the `Cython` rend starts to saturate almost immediately, while the computational time of the `Python` implementation continues to grow.
This behavior highlights the results of the optimizations performed on the `Cython` version which allows the application of the `DNetPRO` algorithm also to larger datasets without loosing performances.
An opposite behavior is found monitoring the number of threads (ref Fig. [3](https://github.com/Nico-Curti/DNetPRO/blob/master/img/nth_timing.svg)): the `Python` version scales quite well with the number of threads, while `Cython` trend is more unstable.
This behavior is probably due to a non-optimal schedule in the parallel section: the work is not equally distributed along the available threads and it penalizes the code efficiency, creating a bottleneck related to the slowest thread.
The above results are computed considering a number of features equal to 90 and, thus, the parallel section distributes the 8100 (`N x N`) iterations along the available threads: when the number of iterations is proportional to the number of threads used (12, 20 and 30 in our case), we have a maximization of the time performances.
Despite of this, the computational efficiency of the `Cython` implementation is so much better than the `Python` one that its usage is indisputable.

## Usage

You can use the `DNetPRO` algorithm into pure-Python modules or inside your C++ application.

### C++ Version

The easiest usage of `DNetPRO` algorithm is given by the example provided in the [example](https://github.com/Nico-Curti/DNetPRO/blob/master/example) folder.
This script includes an easy-to-use command line to run the `DNetPRO` algorithm on a dataset stored into a file.

```
./bin/DNetPRO_couples
Usage: ./DNetPRO_couples -f <std :: string> -o <std :: string> [-frac <St16remove_referenceIfE> ] [-bin <St16remove_referenceIbE> ] [-verbose <St16remove_referenceIbE> ] [-probeID <St16remove_referenceIbE> ] [-nth <St16remove_referenceIiE> ]

DNetPRO couples evaluation 2.0

optional arguments:
        -f,   --input                   Input filename
        -o,   --output                  Output filename
        -s,   --frac                    Fraction of results to save
        -b,   --bin                     Enable Binary output
        -q,   --verbose                 Enable stream output
        -p,   --probeID                 ProbeID name to skip (true/false)
        -n,   --nth                     Number of threads to use
```

If you are interested in using `DNetPRO` algorithm inside your code you can simply import the [`dnetpro_couples.h`](https://github.com/Nico-Curti/DNetPRO/blob/master/include/dnetpro_couples.h) and call the `dnetpro_couples` function.

Then all the results will be stored into a easy-to-manage [`score`](https://github.com/Nico-Curti/DNetPRO/blob/master/include/score.h) object.

### Python Version

The `DNetPRO` object is totally equivalent to a `scikit-learn` feature-selection method and thus it provides the member functions `fit` (to train your model) and `predict` (to test a trained model on new samples).

First of all you need to import the `DNetPRO` modules and then simply call the training/testing functions.

```python
import pandas as pd
from DNetPRO import DNetPRO
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

X = pd.read_csv('./example/data.txt', sep='\t', index_col=0, header=0)
y = np.asarray(X.columns.astype(float).astype(int))
X = X.transpose()

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.33, random_state=42)

dnet = DNetPRO(estimator=GaussianNB(), n_jobs=4, verbose=True)

Xnew = dnet.fit_transform(X_train, y_train)
print('Best Signature: {}'.format(dnet.get_signature()[0]))
print('Score: {:.3f}'.format(dnet.score(X_test, y_test)))
```

## Testing

The Python version of the package is tested using [`pytest`](https://docs.pytest.org/en/latest/).
To install the package in development mode you need to add also this requirement:

* pytest == 3.0.7

The full list of python test scripts can be found [here](https://github.com/Nico-Curti/DNetPRO/blob/master/DNetPRO/test).

## Table of contents

Description of the folders related to the `C++` version.

| **Directory**  |  **Description** |
|:--------------:|:-----------------|
| [example](https://github.com/Nico-Curti/DNetPRO/blob/master/example) | Example script for the `C++` version of the code. Use the command line helper for a full description of the required parameters. |
| [hpp](https://github.com/Nico-Curti/DNetPRO/blob/master/hpp)         | Implementation of the `C++` template functions and objects used in the `DNetPRO` algorithm.                                       |
| [include](https://github.com/Nico-Curti/DNetPRO/blob/master/include) | Definition of the `C++` function and objects used in the `DNetPRO` algorithm.                                                    |
| [src](https://github.com/Nico-Curti/DNetPRO/blob/master/src)         | Implementation of the `C++` functions and objects used in the `DNetPRO` algorithm.                                               |

Description of the folders related to the `Python` version (base directory `DNetPRO`).

| **Directory**  |  **Description** |
|:--------------:|:-----------------|
| [example](https://github.com/Nico-Curti/DNetPRO/blob/master/DNetPRO/example) | `Python` version of the `C++` example.      |
| [lib](https://github.com/Nico-Curti/DNetPRO/blob/master/DNetPRO/lib)         | List of `Cython` definition files.          |
| [source](https://github.com/Nico-Curti/DNetPRO/blob/master/DNetPRO/source)   | List of `Cython` implementation objects.    |
| [test](https://github.com/Nico-Curti/DNetPRO/blob/master/DNetPRO/test)       | List of test scripts for the `Python` wraps.|

Description of the folders containing the scripts used for the paper simulations.

| **Directory**  |  **Description** |
|:--------------:|:-----------------|
| [toy](https://github.com/Nico-Curti/DNetPRO/blob/master/toy)           | Implementation of the `Python` scripts used for the simulations on synthetic datasets. |
| [pipeline/TCGA](https://github.com/Nico-Curti/DNetPRO/blob/master/toy) | Implementation of the `Python` scripts used for the simulations on the TCGA datasets.  |
| [timing](https://github.com/Nico-Curti/DNetPRO/blob/master/timing)     | Implementation of the `Python` scripts for the performances evaluation.           |

## Contribution

Any contribution is more than welcome :heart:. Just fill an [issue](https://github.com/Nico-Curti/DNetPRO/blob/master/.github/ISSUE_TEMPLATE/ISSUE_TEMPLATE.md) or a [pull request](https://github.com/Nico-Curti/DNetPRO/blob/master/.github/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md) and we will check ASAP!

See [here](https://github.com/Nico-Curti/DNetPRO/blob/master/.github/CONTRIBUTING.md) for further informations about how to contribute with this project.

## References

<blockquote>1- K. Scotlandi, D. Remondini, G. Castellani, M. C. Manara, F. Nardi, L. Cantiani, M. Francesconi, M. Mercuri, A. M. Caccuri, M. Serra, S. Knuutila, P. Picci. "Overcoming Resistance to Conventional Drugs in Ewing Sarcoma and Identification of Molecular Predictors of Outcome", Journal of Clinical Oncology, 2009. </blockquote>

<blockquote>2- Y. Wang, J. GM Klijn, Yi Zhang, A. M Sieuwerts, M. P Look, F. Yang, D. Talantov, M. Timmermans, M. E Meijer-van Gelder, J. Yu, T. Jatkoe, E. MJJ Berns, D. Atkins, J. A Foekens. "Gene-expression profiles to predict distant metastasis of lymph-node-negative primary breast cancer", The Lancet, 2005. </blockquote>

<blockquote>3- Y. Yuan, E. M Van Allen, L. Omberg, N. Wagle, A. Amin-Mansour, A. Sokolov, L. A Byers, Y. Xu, K. R Hess, L. Diao, L. Han, X. Huang, M. S Lawrence, J. N Weinstein, J. M Stuart, G. B Mills, L. A Garraway, A. A Margolin, G. Getz, H. Liang. "Assessing the clinical utility of cancer genomic and proteomic data across tumor types", Nature Biotechnology, 2014. </blockquote>

<blockquote>4- C. Mizzi, A. Fabbri, S. Rambaldi, F. Bertini, N. Curti, S. Sinigardi, R. Luzi, G. Venturi, M. Davide, G. Muratore, A. Vannelli, A. Bazzani. "Unraveling pedestrian mobility on a road network using ICTs data during great tourist events", EPJ Data Science, 2018</blockquote>

<blockquote>5- V. Boccardi, L. Paolacci, D. Remondini, E. Giampieri, N. Curti, A. Villa, G. Poli, R. Cecchetti, S. Brancorsini, P. Mecocci. '"Cytokinome signature" in cognitive impairment: from early stages to Alzheimer’s disease', Journal of Alzheimer Disease, 2019</blockquote>

<blockquote>6- C. Terragna, D. Remondini, M. Martello, E. Zamagni, L. Pantani, F. Patriarca, A. Pezzi, G. Levi, M. Offidani, I. Proserpio, G. De Sabbata, P. Tacchetti, C. Cangialosi, F. Ciambelli, C. Virginia Viganò, F. Angela Dico, B. Santacroce, E. Borsi, A. Brioli, G. Marzocchi, G. Castellani, G. Martinelli, A. Palumbo, M. Cavo. "The genetic and genomic background of multiple myeloma patients achieving complete response after induction therapy with bortezomib, thalidomide and dexamethasone", Oncotarget, 2016</blockquote>

## Authors

* <img src="https://avatars0.githubusercontent.com/u/24650975?s=400&v=4" width="25px"> **Nico Curti** [git](https://github.com/Nico-Curti), [unibo](https://www.unibo.it/sitoweb/nico.curti2)
* <img src="https://avatars2.githubusercontent.com/u/1419337?s=400&v=4" width="25px;"/> **Enrico Giampieri** [git](https://github.com/EnricoGiampieri), [unibo](https://www.unibo.it/sitoweb/enrico.giampieri)
* <img src="https://www.unibo.it/uniboweb/utils/UserImage.aspx?IdAnagrafica=236217&IdFoto=bf094429" width="25px"> **Gastone Castellani** [unibo](https://www.unibo.it/sitoweb/gastone.castellani)
* <img src="https://avatars2.githubusercontent.com/u/25343321?s=400&v=4" width="25px"> **Daniel Remondini** [git](https://github.com/dremondini), [unibo](https://www.unibo.it/sitoweb/daniel.remondini)

See also the list of [contributors](https://github.com/Nico-Curti/DNetPRO/contributors) [![GitHub contributors](https://img.shields.io/github/contributors/Nico-Curti/DNetPRO.svg?style=plastic)](https://github.com/Nico-Curti/DNetPRO/graphs/contributors/) who participated in this project.

## License

The `DNetPRO` package is licensed under the MIT "Expat" License. [![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/Nico-Curti/DNetPRO/blob/master/LICENSE)

### Acknowledgment

Thanks goes to all contributors of this project.

### Citation

If you have found `DNetPRO` helpful in your research, please consider citing the original paper

```BibTex
@article {Curti2019DNetPRO,
  author = {Curti, Nico and Giampieri, Enrico and Levi, Giuseppe and Castellani, Gastone and Remondini, Daniel},
  title = {DNetPRO: A network approach for low-dimensional signatures from high-throughput data},
  elocation-id = {773622},
  year = {2019},
  doi = {10.1101/773622},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {\url{https://www.biorxiv.org/content/early/2019/09/19/773622}},
  eprint = {\url{https://www.biorxiv.org/content/early/2019/09/19/773622.full.pdf}},
  journal = {bioRxiv}
}
```
or just this repository

```BibTex
@misc{DNetPRO,
  author = {Curti, Nico},
  title = {{DNetPRO pipeline}: Implementation of the DNetPRO pipeline for TCGA datasets},
  year = {2019},
  url = {https://github.com/Nico-Curti/DNetPRO},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Nico-Curti/DNetPRO}}
}
```
