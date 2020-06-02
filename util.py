"""Utilities for training and running the model."""

from itertools import product
import os

import numpy
import pandas
import joblib

def Rnaseq_process(rnaseq_dir):
    rnaseq_path = os.path.join(rnaseq_dir, "rnaseq.csv")
    rnaseq = pandas.read_csv(rnaseq_path)
    rnaseq_T = rnaseq.T.copy()
    gene_list = rnaseq.Gene.copy()
    rnaseq_T = rnaseq_T.drop("Gene").drop("Symbol")
    rnaseq_T.columns = gene_list
    rnaseq_T = rnaseq_T.astype("float32")
    return rnaseq_T

def RunPrediction_lr(model_dir, input_dir, output_dir):
    print("start prediction")
    rnaseq = Rnaseq_process(input_dir)
    gene_dict = joblib.load(os.path.join(model_dir, "selected_genes.pkl"))
    inhibitors = joblib.load(os.path.join(model_dir, "inhibitors.pkl"))
    specimens = rnaseq.index
    aucs = pandas.DataFrame(product(inhibitors, specimens),columns=['inhibitor', 'lab_id'])
    aucs["auc"] = 0
    for i in aucs.index:
        lab_id = aucs.lab_id[i]
        inhibitor = aucs.inhibitor[i]
        model = joblib.load(os.path.join(model_dir, inhibitor+".pkl"))
        aucs.auc[i] = model.predict(numpy.array(rnaseq[gene_dict[inhibitor]].loc[lab_id]).reshape(1, -1))[0]

    return aucs
    #aucs.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)


def TransposeRnaSeqTable(rnaseq):
  """Convert the RnaSeq table, indexed by gene, to be indexed by specimen."""
  rnaseq.index = rnaseq.Gene
  return rnaseq[rnaseq.columns[2:]].T


def GetPickledModelState(pkl_1_path, pkl_2_path):
  """Fetches (pkl_1, pkl_2) from storage."""
  pkl_1 = pandas.read_csv(pkl_1_path).set_index('gene', drop=False)
  pkl_2 = pandas.read_csv(pkl_2_path).set_index('inhibitor')
  return (pkl_1, pkl_2)


def NormSpecimens(specimens):
  normed_specimens = specimens.apply(
      lambda specimen : specimen / numpy.linalg.norm(specimen), axis=1)
  return normed_specimens


# predict
def Predict(inhibitor, normed_specimen, pkl_1, pkl_2):
  """Uses the pickled model to predict the AUC for the specimen."""
  z_scores = (normed_specimen[pkl_1.gene] - pkl_1.gene_mean) / pkl_1.gene_std
  return z_scores.dot(pkl_1[inhibitor]) + pkl_2.loc[inhibitor].intercept


def RunPredictions(model_dir, input_dir, output_dir):
  print('Loading data...')
  (pkl_1, pkl_2) = GetPickledModelState(
      os.path.join(model_dir, 'pkl_1.csv'),
      os.path.join(model_dir, 'pkl_2.csv'))

  specimens = TransposeRnaSeqTable(pandas.read_csv(os.path.join(input_dir, 'rnaseq.csv')))
  normed_specimens = NormSpecimens(specimens)

  print('Getting the cartesian product of inhibitors and specimens...')
  inhibitors = pkl_2.index
  specimens = normed_specimens.index
  aucs = pandas.DataFrame(
      product(inhibitors, specimens),
      columns=['inhibitor', 'lab_id'])

  print('Predicting per-specimen AUC...')
  aucs['auc'] = aucs.apply(lambda r: (
    Predict(r['inhibitor'], normed_specimens.loc[r['lab_id']], pkl_1, pkl_2)),
    axis=1)
  aucs.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)


def RunPredictions_ensemble(ridge_model_dirs, lr_model_dirs, input_dir, output_dir):
  n1 = len(ridge_model_dirs)
  n2 = len(lr_model_dirs)
  # for each ridge model, conduct one prediction 
  for i in range(0,n1):
    model_dir = ridge_model_dirs[i]
    print('Loading data...')
    (pkl_1, pkl_2) = GetPickledModelState(
        os.path.join(model_dir, 'pkl_1.csv'),
        os.path.join(model_dir, 'pkl_2.csv'))

    specimens = TransposeRnaSeqTable(pandas.read_csv(os.path.join(input_dir, 'rnaseq.csv')))
    normed_specimens = NormSpecimens(specimens)

    print('Getting the cartesian product of inhibitors and specimens...')
    inhibitors = pkl_2.index
    specimens = normed_specimens.index
    aucs = pandas.DataFrame(
        product(inhibitors, specimens),
        columns=['inhibitor', 'lab_id'])

    print('Predicting per-specimen AUC...')
    aucs['auc'] = aucs.apply(lambda r: (
      Predict(r['inhibitor'], normed_specimens.loc[r['lab_id']], pkl_1, pkl_2)),
      axis=1)
    aucs.to_csv(os.path.join(output_dir, 'predictions_'+str(i)+'.csv'), index=False)



  # for each lr model
  for i in range(0,n2):
    model_dir = lr_model_dirs[i]
    tmp_auc = RunPrediction_lr(model_dir,input_dir, output_dir)
    tmp_auc.to_csv(os.path.join(output_dir, 'predictions_'+str(n1+i)+'.csv'), index=False)
  
  # avg
  prediction = pandas.read_csv(os.path.join(output_dir,'predictions_0.csv'))
  for i in range(1,n1+n2):
    tmp_prediction = pandas.read_csv(os.path.join(output_dir,'predictions_'+str(i)+'.csv'))
    prediction['auc'] += tmp_prediction['auc']
  prediction['auc'] /= (n1+n2)
  prediction.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)



