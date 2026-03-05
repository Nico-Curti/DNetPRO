#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
  accuracy_score,
  balanced_accuracy_score,
  f1_score,
  classification_report,
  confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Your estimator / feature selector
from DNetPRO.DNetPRO import DNetPRO as DNetPROSelector

__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']

def _get_version() -> str:
  try:
    from . import __version__  # type: ignore
    return str(__version__)
  except Exception:
    try:
      from importlib.metadata import version
      return version('DNetPRO')
    except Exception:
      return 'unknown'


@dataclass
class RunResult:
  selected_features: list[str]
  metrics: dict[str, float]


def _load_csv(path: Path, sep: str, target_col: Optional[str], index_col: Optional[str]) -> tuple[pd.DataFrame, pd.Series]:
  df = pd.read_csv(path, sep=sep)
  if index_col is not None and index_col in df.columns:
    df = df.set_index(index_col)

  if target_col is None:
    target_col = df.columns[-1]

  if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")

  y = df[target_col]
  X = df.drop(columns=[target_col])
  return X, y


def _prepare_X(X: pd.DataFrame, one_hot: bool) -> tuple[np.ndarray, list[str]]:
  '''
  Convert to numeric matrix.
  - If one_hot=True, categorical columns are one-hot encoded.
  - Else, we try to coerce all columns to numeric (will fail if not possible).
  '''
  if one_hot:
    X2 = pd.get_dummies(X, drop_first=False)
  else:
    X2 = X.copy()
    for c in X2.columns:
      X2[c] = pd.to_numeric(X2[c], errors='raise')

  return X2.to_numpy(dtype=np.float32, copy=False), list(X2.columns)


def _make_classifier(name: str, random_state: int) -> object:
  name = name.lower()
  if name in ('logreg', 'logistic', 'logisticregression'):
    return LogisticRegression(max_iter=2000, n_jobs=1, random_state=random_state)
  if name in ('gnb', 'gaussiannb', 'nb'):
    return GaussianNB()
  raise ValueError(f"Unknown classifier '{name}'. Use 'logreg' or 'gnb'.")


def run_pipeline(
  X: np.ndarray,
  y: np.ndarray,
  feature_names: list[str],
  *,
  test_size: float,
  random_state: int,
  scale: bool,
  classifier: str,
  # DNetPRO parameters
  max_chunk: int,
  percentage: float,
  verbose: int,
  n_jobs: int,
) -> RunResult:
  # Feature selector uses an internal estimator for CV scoring;
  # keep it simple and robust with GaussianNB (works well for many cases).
  fs = DNetPROSelector(
    estimator=GaussianNB(),
    max_chunk=max_chunk,
    percentage=percentage,
    verbose=verbose,
    n_jobs=n_jobs,
  )

  clf = _make_classifier(classifier, random_state=random_state)

  steps = [('fs', fs)]
  if scale:
    steps.append(('scaler', StandardScaler()))
  steps.append(('clf', clf))

  pipe = Pipeline(steps)

  # Split (stratify if classification labels are discrete)
  strat = y if (pd.Series(y).nunique() > 1) else None
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    stratify=strat,
  )

  pipe.fit(X_train, y_train)
  y_pred = pipe.predict(X_test)

  metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
    'f1_macro': float(f1_score(y_test, y_pred, average='macro')),
  }

  # Extract selected feature indices from fs (signature 0 is used by transform)
  # DNetPRO internally calls set_signature(0) inside transform().
  fs_fitted = pipe.named_steps['fs']
  selected_idx = list(getattr(fs_fitted, 'selected_signature', []))

  selected_features = [feature_names[i] for i in selected_idx] if selected_idx else []

  print('\n=== Results ===')
  for k, v in metrics.items():
    print(f'{k:>18}: {v:.6f}')

  print('\n=== Confusion matrix ===')
  print(confusion_matrix(y_test, y_pred))

  print('\n=== Classification report ===')
  print(classification_report(y_test, y_pred, digits=4))

  print('\n=== Selected features ===')
  if selected_features:
    print(f'Selected ({len(selected_features)}):')
    for f in selected_features:
      print(f' - {f}')
  else:
    print('No selected features found (unexpected).')

  return RunResult(selected_features=selected_features, metrics=metrics)


def build_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(
    prog='dnetpro',
    description='DNetPRO CLI: CSV -> sklearn-like pipeline (feature selection + classification)',
  )
  p.add_argument('--version', action='version', version=_get_version())

  sub = p.add_subparsers(dest='cmd', required=True)

  run = sub.add_parser('run', help='Run a DNetPRO feature-selection + classification pipeline on a CSV')
  run.add_argument('csv', type=str, help='Path to input CSV')
  run.add_argument('--sep', type=str, default=',', help="CSV separator (default: ',')")
  run.add_argument('--target-col', type=str, default=None, help='Target/label column name (default: last column)')
  run.add_argument('--index-col', type=str, default=None, help='Optional index column name (will be set as index)')

  run.add_argument('--one-hot', action='store_true', help='One-hot encode non-numeric columns')
  run.add_argument('--scale', action='store_true', help='Add StandardScaler after feature selection')

  run.add_argument('--classifier', type=str, default='logreg', choices=['logreg', 'gnb'], help='Classifier step')
  run.add_argument('--test-size', type=float, default=0.25, help='Test split fraction')
  run.add_argument('--random-state', type=int, default=42, help='Random seed')

  # DNetPRO knobs
  run.add_argument('--max-chunk', type=int, default=100, help='DNetPRO max_chunk')
  run.add_argument('--percentage', type=float, default=0.10, help='DNetPRO percentage of couples retained')
  run.add_argument('--verbose', type=int, default=0, help='DNetPRO verbosity (0/1/2...)')
  run.add_argument('--n-jobs', type=int, default=1, help='DNetPRO parallel jobs')

  run.add_argument('--save-features', type=str, default=None, help='Save selected features (one per line) to file')

  run.set_defaults(func=cmd_run)

  return p


def cmd_run(args: argparse.Namespace) -> int:
  csv_path = Path(args.csv)
  if not csv_path.exists():
    print(f'ERROR: file not found: {csv_path}', file=sys.stderr)
    return 2

  try:
    X_df, y_s = _load_csv(csv_path, sep=args.sep, target_col=args.target_col, index_col=args.index_col)
    X, feature_names = _prepare_X(X_df, one_hot=args.one_hot)
    y = y_s.to_numpy()
  except Exception as e:
    print(f'ERROR while loading/preparing CSV: {e}', file=sys.stderr)
    return 2

  try:
    res = run_pipeline(
      X, y, feature_names,
      test_size=args.test_size,
      random_state=args.random_state,
      scale=args.scale,
      classifier=args.classifier,
      max_chunk=args.max_chunk,
      percentage=args.percentage,
      verbose=args.verbose,
      n_jobs=args.n_jobs,
    )
  except Exception as e:
    print(f'ERROR while running pipeline: {e}', file=sys.stderr)
    return 3

  if args.save_features:
    out = Path(args.save_features)
    try:
      out.write_text('\n'.join(res.selected_features) + '\n', encoding='utf-8')
      print(f'\nSaved selected features to: {out}')
    except Exception as e:
      print(f'WARNING: cannot save features to {out}: {e}', file=sys.stderr)

  return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = build_parser()
  ns = parser.parse_args(list(argv) if argv is not None else None)
  return int(ns.func(ns))


if __name__ == '__main__':
  raise SystemExit(main())
