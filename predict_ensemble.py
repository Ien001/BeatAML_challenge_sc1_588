"""A script which executes an drug repurposing model on input files.

Expects:
  /model/: should have pkl_1.csv and pkl_2.csv
  /input/: should have rnaseq.csv and input.csv

Creates:
  /output/aucs.csv: a CSV with columns ['inhibitor', 'lab_id', 'auc']
"""

import util


if __name__ == "__main__":
  ridge_model_dirs = ['/model/Ridge_combine_ian','/model/Ridge_ttest_NPM','/model/LA_1000',]
  lr_model_dirs = ['/model/lr_model']
  util.RunPredictions_ensemble(ridge_model_dirs, lr_model_dirs, '/input', '/output')


