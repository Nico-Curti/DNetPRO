#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import networkx as nx
from subprocess import Popen
from subprocess import PIPE
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']

seed = 123
np.random.seed(seed) # reproducibility

def build_tree_dir (local, cancer, dtype):
  '''
  Function utility to make tree directories.

  Parameters
  ----------
  local : string
    Current path. (ex. os.path.abspath("."))

  cancer : array_like
    List of strings with cancer name. (ex. ["KIRC", "GBM", "LUSC", "OV"])

  dtype : string
    Type of genomic data under analysis.

  Returns
  -------
  None

  Notes
  -----
  For each cancer-name it's assumed that already exist a directory with that name.

  '''
  print ( 'Buildinig tree directories    ', end='' )

  for canc in cancer:
    os.makedirs(os.path.join(local, canc, '_'.join(['Train', canc, dtype])), exist_ok=True)
    os.makedirs(os.path.join(local, canc, '_'.join(['Train', canc, dtype]), '_'.join(['couples', dtype])), exist_ok=True)
    os.makedirs(os.path.join(local, canc, '_'.join(['Train', canc, dtype]), '_'.join(['Trainset', dtype])), exist_ok=True)

  print ( '[done]' )
  return


def generateDB_fromTCGA (cancer, local, datafile='_mRNA_core.txt', lblfile='_binary_survival.txt', out='_data_mRNA.txt'):
  '''
  Create dataset fmt for couples step from TCGA data fmt.
  Import datafile for each cancer and add labels from binary_survival file.
  The transpose of data-core is necessary for couples (genes := rows, samples := columns)

  Parameters
  ----------
  cancer : string
    Cancer name.

  local : string
    Current local path. It's assumed that a directory with cancer name exists in this path.

  datafile : string (default='_mRNA_core.txt')
    Postfix of filename of cancer data. Complete datafile will be cancer + datafile.

  lblfile : string (default='_binary_survival.txt')
    Postfix of filename of cancer labels. Complete lblfile will be cancer + lblfile.

  out : string (default='_data_mRNA.txt')
    Postfix of output results. Complete out will be cancer + out.

  Returns
  -------
  None

  Notes
  -----
  Data used in TCGA-DNetPRO-Pipeline are extracted from https://www.synapse.org/#!Synapse:syn1710282/wiki/27303
  Reference: "Assessing the clinical utility of cancer genomic and proteomic data across tumor types"
  (Yuan et al., 2014, Nature Biotechnology).

  '''

  # read the data file (probes and samples)
  data   = pd.read_csv(os.path.join(local, cancer, cancer + datafile), sep='\t', index_col=0).iloc[:, :-1]
  # read the labels file
  binary = pd.read_table(os.path.join(local, cancer, cancer + lblfile), sep='\t')
  # filter the data according to label values association
  data   = data.filter(items=binary.feature, axis=0).T
  # set the columns as labels in multi-cols order
  data.columns = [binary.is_alive.values, data.columns.values]

  # Add random noise to prevent null values problems
  data = data.replace(to_replace='null', value=0).astype(float)
  data[data == 0] = pd.DataFrame(data=np.random.normal(loc=0., scale=1e-4, size=data.shape), columns=data.columns, index=data.index)[data == 0]
  data.to_csv(out, header=True, index=True, sep='\t')
  return


def proxy_generateDB_fromTCGA (Dict):
  '''
  Proxy fuction for Pool thread
  '''
  generateDB_fromTCGA(**Dict)
  return


def generate_folds(cancer, local, datafile='_data_mRNA.txt', K=10, Nit=10, dtype='mRNA'):
  '''
  Generate indices of iterative K-fold cross validation.

  Parameters
  ----------
  cancer : string
    Cancer name.

  local : string
    Current local path. It's assumed that a directory with cancer name exists in this path.

  datafile : string (default='_data_mRNA.txt')
    Postfix of filename of cancer data. Complete datafile will be cancer + datafile.

  K : int (default=10)
    Number of folds from the K-Fold CrossValidation. Default := 10-folds. The limits of this parameter
    are related to the implementation of StratifiedKFold of sklearn library.

  Nit : int (default=10)
    Number of iteration of K-Fold CV. Nit must be positive.

  dtype : string (default='mRNA')
    Genomic datatype name.

  Returns
  -------
  None

  Notes
  -----
  In TCGA-QDANetPRO-Pipeline dtypes allowed are only 'mRNA', 'miRNA', 'RPPA'.
  Loop over StratifiedKFold is prefered to the single call RepeatedStratifiedKFold because iteraion indices are needed
  in later steps.

  '''
  # TODO: this step and the next one can be optimize using pickle dump and load!

  if Nit <= 0:
    raise ValueError ( 'Nit must be positive. Given : {}'.format(Nit) )

  if K < 2:
    raise ValueError ( 'K must be greater than the number of classes. Given : {}'.format(K) )

  loc = os.path.join(local, cancer, cancer)
  # read the data file
  db  = pd.read_table(loc + datafile, sep='\t', index_col=0, header=[0, 1])
  # extract the label according to the right column order
  lbl = list(db.columns.get_level_values(0).astype(float).astype(int))

  # empty array
  Nsample = np.zeros(shape=(len(lbl),))

  # Generate Nit K-fold indices and save each file as a separate filename
  with open(loc + '_train_idx.txt', 'w') as train, open(loc + '_test_idx.txt', 'w') as test:
    # rskf = RepeatedStratifiedKFold(n_splits=K, n_repeats=Nit, random_state=36851234)
    # for train_index, test_index in rskf.split(Nsample, lbl):
    #     train.write(','.join(train_index) + '\n')
    #     test.write(','.join(test_index) + '\n')
    for i in range(Nit):
      cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=i + 36851234) # K-Fold cross validation
      for train_index, test_index in cv.split(Nsample, lbl):
        train.write('{}\t'.format(i))
        train.write(','.join(map(str, train_index)) + '\n')

        test.write('{}\t'.format(i))
        test.write(','.join(map(str, test_index)) + '\n')

  return

def proxy_generate_folds (Dict):
  '''
  Proxy function for Pool thread
  '''
  generate_folds(**Dict)
  return



def generate_training(cancer, local, datafile='_data_mRNA.txt', cv_file='_train_idx.txt', dtype='mRNA'):
  '''
  Training-set generation. Training indices are obtained by 'generate_folds' function.

  Parameters
  ----------
  cancer : string
    Cancer name.

  local : string
    Current local path. It's assumed that a directory with cancer name exists in this path.

  datafile : string (default='_data_mRNA.txt')
    Postfix of filename of cancer data. Complete datafile will be cancer + datafile.

  cv_file : string (default='_train_idx.txt')
    Postfix of filename of training indices. Complete datafile will be cancer + cv_file.

  dtype : string (default='mRNA')
    Genomic datatype name.

  Returns
  -------
  None

  Notes
  -----
  In TCGA-QDANetPRO-Pipeline dtypes allowed are only 'mRNA', 'miRNA', 'RPPA'.
  Loop over StratifiedKFold is prefered to the single call RepeatedStratifiedKFold because iteraion indices are needed
  in later steps.

  '''
  loc = os.path.join(local, cancer)
  tmp_datafile  = os.path.join(loc, cancer + datafile)
  tmp_trainfile = os.path.join(loc, cancer + cv_file)

  # read the data file
  db  = pd.read_table(tmp_datafile, sep='\t', index_col=0, header=[0, 1])
  # extract the label
  lbl = np.asarray(db.columns.get_level_values(0).astype(float).astype(int))

  # for each indices-file split the data according to that indices and save them
  with open(tmp_trainfile, 'r') as f:
    rows = ( line.split('\t') for line in f )
    d = { (i, int(row[0])): list(map(int, row[1:][0].split(','))) for i, row in enumerate(rows) }
    idx_train = {}
    for key, value in d.items():
      idx_train.setdefault(key[1], []).append(value)

  train_dir     = os.path.join(local, cancer, '_'.join(['Train', cancer, dtype]))
  train_dir_set = os.path.join(train_dir, 'Trainset_' + dtype)

  for fold, fold_idx in idx_train.items():
    for train, train_idx in enumerate(fold_idx):
      tmp = db.iloc[:, train_idx]
      tmp.columns = lbl[train_idx]
      output = os.path.join(train_dir_set, '_'.join([cancer, 'Train', str(fold), str(train) + '.txt']))
      tmp.to_csv(output, header = True, index = True, sep = "\t", mode = "w")

  return

def proxy_generate_training (Dict):
  '''
  Proxy function for Pool thread
  '''
  generate_training(**Dict)
  return


def run_couples(cancer, local, train_dir, percent=.1, dtype='mRNA'):
  '''
  Couples command line building.
  An error equal to 1 is thrown if something goes wrong in the software execution.

  Parameters
  ----------
  cancer : string
    Cancer name.

  local : string
    Current local path. It's assumed that a directory with cancer name exists in this path.

  train_dir : string
    Training files directory. Two sub-directories are required:
      - 'Trainset_' + cancer + dtype : directory of training files.
      - 'couples_' + cancer + dtype : directory of resulting couples.

  percent : float (default=0.1)
    Percentage of couples to save. This parameter will be insert in Couples command line. Couples will
    save this percentage of single-gene-couples (geneA, geneA) followed by gene-couples (geneA, geneB)
    in a binary file format.

  dtype : string (default='mRNA')
    Genetic data type.

  Returns
  -------
  None

  '''
  if not (isinstance(percent, float) or 0 < percent <= 1.):
    raise ValueError('Couples Error: Invalid percent value. percentage must be > 0 and <= 1. Given {}'.format(percent))

  train_dir_couples = os.path.join(train_dir, 'couples_' + dtype)
  # read all the file in the directory
  files = (f for f in os.listdir( os.path.join(train_dir, 'Trainset_' + dtype))
           if os.path.isfile(os.path.join( os.path.join(train_dir, 'Trainset_' + dtype), f)))

  # for each file run the couples evaluation with the maximum number of threads
  for f in files:
    token = f.split('_')
    # extract fold infos by the filename
    r, fold = token[-1], token[-2]

    inpt = os.path.join(train_dir, 'Trainset_' + dtype, cancer + r'_Train_' + fold + '_' + r)

    # run command line process
    process = Popen([os.path.join(local, 'couples'),
                     '-f {}'.format(inpt),
                     '-s {}'.format(percent),
                     '-p 1',
                     '-q 0',
                     '-b 1',
                     '-o {}'.format(os.path.join(train_dir_couples, 'binary_train_' + fold + '_' + r))
                     ], stdout=PIPE, stderr=PIPE)

    process_rc = process.returncode

    if process_rc != 0: # somethig goes wrong
      raise ValueError('Error in couples step.\nCancer: {}\nFile: {}'.format(cancer, inpt) )

  return

def proxy_run_couples (Dict):
  '''
  Proxy function for Pool thread
  '''
  run_couples(**Dict)
  return



def feature_sel_train(cancer, local, train_dirs, train_couples, max_chunk=100, dtype='mRNA', percent=.1):
  '''
  Signatures extraction.
  For each training set, starting from chunk-range-performances of couples, performances over a leave-one-out CV
  of each connected components in the build-graph are computed. Then for each extracted signature
  the betweenness centrality of each gene in the sub-network is saved as score of the gene.

  Parameters
  ----------
  cancer : string
    Cancer name.

  local : string
    Current local path. It's assumed that a directory with cancer name exists in this path.

  train_dirs : string
    Directory of training files. Two sub-directories are required:
      - 'Trainset_' + cancer + dtype : directory of training files.
      - 'couples_' + cancer + dtype : directory of resulting couples.

  train_couples : string
    Directory of couples. Example: os.path.join(train_dirs, 'couples_' + cancer).

  max_chunk : int (default=100)
    Max number of genes allowed in performances-chunk. If the size of chunk is greater than max_chunk
    and it is not the first one, features selection iteration is stopped.

  dtype : string (default='mRNA')
    Genomic data type.

  percent : float (default=0.1)
    Percentage of couples to save. This parameter will be insert in couples command line. Couples will
    save this percentage of single-gene-couples (geneA, geneA) followed by gene-couples (geneA, geneB)
    in a binary file.

  Returns
  -------
  For more info
  #TODO: INSERT HERE PAPER QDANETPRO URL

  '''

  data_type = [('genea', np.uint16), ('geneb', np.uint16), ('c0', np.uint16), ('c1', np.uint16), ('ct', np.uint16)]
  columns = list(x[0] for x in data_type)
  results = []

  couple_files = sorted((os.path.join(train_couples, f)
                        for f in os.listdir( train_couples + os.sep )
                        if os.path.isfile(os.path.join( train_couples, f))) )
  train_files  = sorted((os.path.join(train_dirs, 'Trainset_' + dtype, d)
                        for d in os.listdir( os.path.join(train_dirs, 'Trainset_' + dtype) )
                        if os.path.isfile(os.path.join( train_dirs, 'Trainset_' + dtype, d))) )

  for f, t in zip(couple_files, train_files): # loop over file couples and training
    # extract iteration infos from filename
    token = f.split('_')
    tr, fold = int(token[-1]), int(token[-2])

    # read probe-sample file
    db    = pd.read_table(t, sep='\t', index_col=0, header=0)

    # read corresponding couples file
    couples = np.fromfile(f, dtype=data_type).tolist()
    couples = pd.DataFrame(couples, columns=columns)[int(len(db) * percent) + 1 :] # only mixed couples!

    # extract labels
    lbl     = np.asarray(db.columns.astype(float).astype(int))

    # groupby the same total performance
    chunks = (couples.groupby(couples.ct, sort=False))

    # signatures graph
    G = nx.Graph()
    for perf, chunk in chunks:

      # arrest criteria
      if len(chunk.genea.values) >= max_chunk and perf != couples.iloc[0].ct:
        break

      # add couples to the network
      G.add_weighted_edges_from([(a, b, perf) for a,b in zip(chunk.genea.values, chunk.geneb.values)])

      # evaluate each connected component as putative signature
      graphs = (nx.connected_component_subgraphs(G))

      # for each putative signature
      for comp in graphs:
        # compute the betweenness centrality for ranking nodes
        # bc = nx.betweenness_centrality(comp, weight='weights')

        # evaluate the filtered data
        tmp_data = db.iloc[comp.nodes()]
        score = cross_val_score(QDA(),
                                tmp_data.T,
                                lbl,
                                cv = LeaveOneOut()
                                ).mean()

        # save the results
        results.append({
                        'dtype'       : dtype, # datatype
                        'cancer'      : cancer, # cancer name
                        'n_gene_db'   : len(db), # number of gene in db
                        'n_gene_sign' : len(comp.nodes()), # number of gene in signature
                        'train'       : tr, # number of train
                        'fold'        : fold, # number of fold
                        'perf_couple' : perf, # performance couple binaryfile
                        'nodes'       : ';'.join(map(str, comp.nodes())), # string of nodes
                        #'BC'          : ';'.join(map(str, bc)), # string of bc values
                        'score'       : score * 100. # score of classification in training step with Loo
                        })

  # save the results
  pd.DataFrame(data=results).to_csv(os.path.join(local, cancer + '_signatures_' + dtype + '.csv'), sep=',', index=False)

  return

def proxy_feature_sel_train (Dict):
  '''
  Proxy function for Pool thread
  '''
  feature_sel_train(**Dict)
  return


def import_signatures (local, names):
  '''
  Import database of signatures extracted for each cancer.

  Parameters
  ----------
  local : string
    Current local path. It's assumed that a directory with cancer name exists in this path.

  names : array_like
    Signatures dataset name. The files must be in the same path of 'local'.

  Returns
  -------
  DataFrame of signatures db.

  Notes
  -----
  For each cancer the signature db is imported as csv in a pandas DataFrame. Each db is concatenated with default
  input by pandas concat function.

  '''

  print ( 'Reading Signatures files   ', end='' )
  signatures = pd.concat([pd.read_csv(os.path.join(local, name), sep=',') for name in names])
  print ( '[done]' )
  return signatures


def validation (cancer, local, signature_file, datafile='_data_mRNA.txt', trainfile='_train_idx.txt', testfile='_test_idx.txt', dtype='mRNA'):
  '''
  '''

  print('Validation step for cancer {}'.format(cancer), end='')

  # read signature database
  signatures = pd.read_csv(signature_file, sep=",")

  # read data and labels
  loc  = os.path.join(local, cancer)
  data = pd.read_table(os.path.join(loc, cancer + datafile), sep = "\t", index_col = 0, header=[0, 1])
  lbl  = np.asarray(data.columns.get_level_values(0).astype(float).astype(int))

  # TODO: use default dict and please change this orrible load method!
  with open(os.path.join(loc, cancer + trainfile), 'r') as f:
    rows = ( line.split('\t') for line in f )
    d = { (i, int(row[0])): list(map(int, row[1:][0].split(","))) for i, row in enumerate(rows) }
    train_idx = {}
    for key, value in d.items():
      train_idx.setdefault(key[1], []).append(value)

  with open(os.path.join(loc, cancer + testfile), 'r') as f:
    rows = ( line.split('\t') for line in f )
    d = { (i, int(row[0])): list(map(int, row[1:][0].split(","))) for i, row in enumerate(rows) }
    test_idx = {}
    for key, value in d.items():
      test_idx.setdefault(key[1], []).append(value)

  cls = QDA()

  final_db = []

  # loop over each selected signature
  for i, signature in signatures.iterrows():
    n = np.asarray(signature.nodes.split(';'), dtype = int)
    signature.train = int(signature.train)
    idx_train = train_idx[int(signature.fold)][int(signature.train)]
    idx_test  = test_idx[int(signature.fold)][int(signature.train)]
    Train     = data.iloc[n, idx_train].T

    # take the half of indices for temporary training and the other half for validation
    idx_test1 = idx_test[: len(idx_test)//2]
    idx_test2 = idx_test[len(idx_test) //2 :]

    Test1 = data.iloc[n, idx_test1].T
    Test2 = data.iloc[n, idx_test2].T

    lbl_train = lbl[idx_train]

    lbl_test1  = lbl[idx_test1]
    lbl_test2  = lbl[idx_test2]

    # train over training and predict over test
    cls.fit(Train, lbl_train)
    # test
    lbl_pred_test1_as_test = cls.predict_proba(Test1)
    lbl_pred_test2_as_test = cls.predict_proba(Test2)

    # train over training + test and predict over validation set
    cls.fit(pd.concat([Train, Test1], ignore_index=True), np.concatenate([lbl_train, lbl_test1]))
    lbl_pred_test2_as_val = cls.predict_proba(Test2)

    cls.fit(pd.concat([Train, Test2], ignore_index=True), np.concatenate([lbl_train, lbl_test2]))
    lbl_pred_test1_as_val = cls.predict_proba(Test1)


    for val_scores, test_scores, val_labels, test_labels, val_idx, test_idx_ in zip([lbl_pred_test1_as_val,  lbl_pred_test2_as_val],
                                                                                   [lbl_pred_test2_as_test, lbl_pred_test1_as_test],
                                                                                   [lbl_test1, lbl_test2],
                                                                                   [lbl_test2, lbl_test1],
                                                                                   [idx_test1, idx_test2],
                                                                                   [idx_test2, idx_test1]
                                                                                   ):
      for group_name, scores, labels, idxs in zip(['validation', 'test'],
                                                  [val_scores, test_scores],
                                                  [val_labels, test_labels],
                                                  [val_idx,    test_idx_]
                                                  ):
        for subj_score, subj_label, subj_idx in zip(scores, labels, idxs):
          info = dict()
          info['fold'] = signature.train
          info['repetition'] = signature.fold
          info['cancer'] = signature.cancer
          info['dtype'] = signature['dtype']
          info['signature_nodes'] = signature.nodes
          info['signatureID'] = i
          info['betweenness_centrality'] = signature.bc
          info['performance_chunck_couples'] = signature.perf_couple
          info['training_accuracy_score_signature'] = signature.score
          info['subjectID'] = subj_idx
          info['subject_label'] = subj_label
          prob_cl0, prob_cl1 = subj_score
          info['subject_proba_cl0'] = prob_cl0
          info['subject_proba_cl1'] = prob_cl1
          info['subject_group'] = group_name

          final_db.append(info)

  final_db = pd.DataFrame(data=final_db).to_csv(os.path.join(local, signature_file.split(os.sep)[-1].split(".")[0] +  ".tidy"), sep=",", index=False)

  print ( '[done]' )
  return

def proxy_validation (Dict):
  '''
  Proxy function for Pool thread
  '''
  validation(**Dict)
  return Dict
