import argparse

import numpy as np
import os
import pandas as pd
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, accuracy_score


def in_stream_evaluation(results, known_labels, k=(1, 5,), eps=1e-8):
  results = np.array(results, dtype=[
    ('true_label', np.int32),
    ('predicted_label', np.int32),
    ('real_novelty', np.bool),
    ('detected_novelty', np.bool)
  ])
  print('Stream step, known labels: {}'.format(known_labels))

  ## == Close World Classification Accuracy, CwCA ===
  known_results = results[np.isin(results['true_label'], list(known_labels))]
  cm = confusion_matrix(
    known_results['true_label'],
    known_results['predicted_label'],
    sorted(list(known_labels)+[-1])
  )
  CwCA = accuracy_score(
    known_results['true_label'],
    known_results['predicted_label']
  )  
  # == per class Classification Accuracy ===========
  acc_per_class = cm.diagonal() / cm.sum(axis=1)

  ## == Unknown (Novel) Detection Accuracy (UDA) ====
  real_novelties = results[results['real_novelty']]
  detected_novelties = results[results['detected_novelty']]
  detected_real_novelties = results[results['detected_novelty'] & results['real_novelty']]
  tp = len(detected_real_novelties)
  fp = len(detected_novelties) - len(detected_real_novelties)
  fn = len(real_novelties) - len(detected_real_novelties)
  tn = len(results) - tp - fp - fn
  M_new = fn / (tp + fn + eps)
  F_new = fp / (fp + tn + eps)

  return CwCA, M_new, F_new, cm, acc_per_class
