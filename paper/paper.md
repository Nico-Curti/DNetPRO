---
title: 'DNetPRO: Discriminant Analysis with Network Processing'
tags:
- feature-selection
- microarray-data
- network-analysis
- mrna
- combinatorial-optimization
- machine-learning-algorithms
- parallel
- python3
- cpp
authors:
- name: Nico Curti
  orcid: 0000-0001-5802-1195
  affiliation: 1
- name: Enrico Giampieri
  orcid: 0000-0003-2269-2338
  affiliation: 1
- name: Daniel Remondini
  orcid: 0000-0003-3185-7456
  affiliation: 2
- name: Gastone Castellani
  orcid: 0000-0003-4892-925X
  affiliation: 1
affiliations:
- name: Department of Experimental, Diagnostic and Specialty Medicine of Bologna University
  index: 1
- name: Department of Physics and Astronomy of Bologna University
  index: 2
date: 02 August 2022
bibliography: paper.bib
---

# Summary

The `DNetPRO` project implements a `scikit-learn` compatible feature selection algorithm based on a bottom-up combinatorial approach that exploits the discriminant power of all feature pairs.
The core of the library is written in `C++` (with multi-threading support via OpenMP) to guarantee a fast computational time of the possible feature pairs.
The wrap, via `Cython`, of the `C++` APIs guarantees an efficient computational time also in the `Python` applications.
The library provides easily extendible APIs and possible integrations with other projects.
The method is easily scalable allowing efficient computing for high number of observable ($10^4$—$10^5$).

We tested our method on synthetic data, showing how its efficiency increases on ill-posed problems (similar to those encountered in omics analysis) in comparison with classical incremental feature selection methods.
We have also applied the algorithm on real high-throughput genomic datasets [@10.1101/773622@, @10.3233/JAD-190480@, @10.1007/BF02951333@], proving how our solution is able to outperform existing results or compare to them but with a smaller number of selected features.
The core of the `DNetPRO` method has been also applied for dimensional reduction of network structures, where sub-modules of the network were identified by studying the correlation between links [@10.1140/epjds/s13688-018-0168-2@]

# Statement of need

One of the prominent problems in Biomedical Big Data Analytics is to extract low-dimensional sets of features – signatures – for a better patients’ stratification and a personalized intervention strategy.
Biological data, such as gene or protein expression, are commonly characterized by an up/down-regulation behavior, for which discriminant-based methods could perform with high accuracy and easy interpretability.
To obtain the most out of these methods features selection is even more critical, but it is known to be a NP-hard problem.

We propose a new method of feature selection – `DNetPRO`, *Discriminant Analysis with Network PROcessing* - developed to overcome the problems mentioned above.
The `DNetPRO` method is an attempt to overcome single feature selection without the computational burden of the full combinatorial exploration, with a computing time for feature space exploration proportional to the square of the number of features (ranging from $10^3$ to $10^5$ in a typical high-throughput omics study).
This method implements a network-based heuristic to generate one or more signatures out of the best performing feature pairs.
Moreover, the geometrical simplicity of the resulting class-separation surfaces allows a clearer interpretation of the obtained signatures in comparison to nonlinear classification models.
The `DNetPRO` method belongs to the category of network-based algorithms, a class of methods recently applied for dimensionality reduction, visualization and clustering tasks [@10.1016/j.cell.2015.05.047@, @10.1038/s41467-021-22266-1@, @10.1023/B:MACH.0000033120.25363.1e@].

# Algorithm description

The pseudo-code of the proposed `DNetPRO` algorithm could be sketched as:

> **Data:** Data Matrix (N, S)\
> **Result:** List of putative signatures
>
> Divide the data into training and test by a Hold-Out method;
>
> **FOR** `couple` &larr; (feature_1, feature_2) &in; `Couples` **DO**
>> &nbsp;&nbsp;&nbsp;&nbsp;Leave-One-Out cross validation;\
>> &nbsp;&nbsp;&nbsp;&nbsp;Score estimation using the Classifier;
>
> **END**
>
> Sorting of the couples in ascending order according to their score;\
> Threshold over the couples score (K-best couples);
>
> **FOR** `component` &in; `connected_components` **DO**
>> &nbsp;&nbsp;&nbsp;&nbsp;**IF** `reduction`\
>> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Iteratively pendant node removal;\
>>  &nbsp;&nbsp;&nbsp;&nbsp;**END**
>>
>> Signature evaluation using the Classifier;
>
> **END**

Given a `dataset`, consisting of $S$ `samples` (e.g., cells, patients) with $N$ observations each (our `features`, e.g., omics measurements such as gene, protein or metabolite expression) the signature identification procedure is summarized with the following pipeline:

1. Separation of available data into a `training` and a `test` set (typically 66/33, or 80/20).
2. Estimation of the classification performance according to the desired metric on the training set of all $S(S-1)/2$ `feature pairs` through a computationally fast and reproducible cross-validation procedure (leave-one-out cross validation was chosen).
  The results are mapped into a completely connected symmetric weighted network, with nodes corresponding to features and link weights corresponding to performance of the node couples.
3. Selection of top-performing pairs through a hard-thresholding procedure, that removes links (and nodes) from the initial completely connected network: every `connected component` obtained is considered as a putative classification signature.
  The threshold value can be tuned according to a desired minimum-performance value or considering a minimum number of nodes/features  in the signature.
  The threshold value can be determined also via cross validation of the entire signature extraction procedure.
4. [**Optional**] In the hypothesis that node degree is associated to the global feature performance in combination with the other features, to reduce the size of an identified signature, the `pendant nodes` of the signature network, *i.e.*, nodes with degree equal to one, can be removed.
  This procedure can be applied once, or recursively until the core network, *i.e.*, a network with all nodes with at least two links, is reached.
  We have tested the efficacy of this empirical approach in some real cases [@10.3233/JAD-190480@, @10.1007/BF02951333@], obtaining a smaller-dimensional signature with comparable performance, even if there is not a solid theoretical basis supporting this procedure.
5. [**Optional**]
6. **(a)** All signatures are applied onto the test set to estimate their performance, producing more than one final signature.

  **OR**

  **(b)** To identify a unique best performing signature, a further cross validation step can be applied, with a further `dataset` splitting into training (to identify the multiple signatures), test (to identify the best signature) and validation set (to evaluate the best signature performance).

To test the performance of all feature pairs, we used a diag-quadratic Discriminant Analysis, a robust classifier that allows fast computation.
We remark that the signatures have a purely statistical relevance, being generated with a purpose of maximal classification performance, but previous applications of a simplified version of the DNetPRO algorithm [@10.1101/gr.155192.113@, @10.1200/JCO.2008.19.2542@, @10.1101/gr.193342.115@, @10.18632/oncotarget.5718@] allowed to gain knowledge on the biological mechanisms associated to the studied phenomena.

# Acknowledgments

The authors acknowledge IMI-2 HARMONY n. 116026 EU Project and IMforFUTURE Horizon 2020 (EU) Project.

# References
