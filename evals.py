import sys
from sklearn import metrics
import math
import os
from sklearn.metrics import auc
from copy import deepcopy
import numpy as np
import warnings
import time
import tensorflow as tf
import math
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)

import tensorflow as tf
import numpy as np
import math

def ranking_precision_score(Y_true, Y_score, k=10):
    """Precision at rank k
    Parameters
    ----------
    Y_true : tensor, shape = [n_samples]
        Ground truth (true relevance labels).
    Y_score : tensor, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    precision @k : float
    """
    n = len(Y_true)
    unique_Y = tf.unique(Y_true).y
    if len(unique_Y) > 2:
        raise ValueError("Only supported for two relevance levels.")
    
    pos_label = unique_Y[1]
    n_pos = tf.reduce_sum(tf.cast(Y_true == pos_label, tf.float32))
    order = tf.argsort(Y_score, axis=0, direction='DESCENDING')
    Y_true = tf.gather(Y_true, order[:k])

    n_relevant = tf.reduce_sum(tf.cast(Y_true == pos_label, tf.float32))
    prec = n_relevant / float(k)
    return tf.reduce_mean(prec)

def subset_accuracy(true_targets, predictions, per_sample=False, axis=0):
    result = tf.reduce_all(tf.equal(true_targets, predictions), axis=axis)
    if not per_sample:
        result = tf.reduce_mean(tf.cast(result, tf.float32))
    return result

def hamming_loss(true_targets, predictions, per_sample=False, axis=0):
    result = tf.reduce_mean(tf.cast(tf.math.logical_xor(true_targets, predictions), tf.float32), axis=axis)
    if not per_sample:
        result = tf.reduce_mean(result)
    return result

def compute_tp_fp_fn(true_targets, predictions, axis=0):
    tp = tf.reduce_sum(true_targets * predictions, axis=axis)
    fp = tf.reduce_sum(tf.cast(~true_targets, tf.float32) * predictions, axis=axis)
    fn = tf.reduce_sum(true_targets * tf.cast(~predictions, tf.float32), axis=axis)
    return tp, fp, fn

def example_f1_score(true_targets, predictions, per_sample=False, axis=0):
    tp, fp, fn = compute_tp_fp_fn(true_targets, predictions, axis=axis)
    numerator = 2 * tp
    denominator = tf.reduce_sum(true_targets, axis=axis) + tf.reduce_sum(predictions, axis=axis)
    zeros = tf.where(denominator == 0)[0]
    denominator = tf.gather(denominator, zeros, axis=0)
    numerator = tf.gather(numerator, zeros, axis=0)

    example_f1 = numerator / denominator
    if per_sample:
        f1 = example_f1
    else:
        f1 = tf.reduce_mean(example_f1)

    return f1

def f1_score_from_stats(tp, fp, fn, average='micro'):
    assert len(tp) == len(fp)
    assert len(fp) == len(fn)

    if average == 'micro':
        f1 = 2 * tf.reduce_sum(tp) / (2 * tf.reduce_sum(tp) + tf.reduce_sum(fp) + tf.reduce_sum(fn))
    elif average == 'macro':
        def safe_div(a, b):
            return tf.where(b != 0, a / b, tf.zeros_like(a))
        tmp = safe_div(2 * tp, 2 * tp + fp + fn + 1e-6)
        f1 = tf.reduce_mean(tmp)

    return f1

def f1_score(true_targets, predictions, average='micro', axis=0):
    tp, fp, fn = compute_tp_fp_fn(true_targets, predictions, axis=axis)
    f1 = f1_score_from_stats(tp, fp, fn, average=average)
    return f1

def compute_fdr(all_targets, all_predictions, fdr_cutoff=0.5):
    fdr_array = []
    for i in range(all_targets.shape[1]):
        try:
            precision, recall, thresholds = tf.metrics.precision_recall_curve(all_targets[:, i], all_predictions[:, i], pos_label=1)
            fdr = 1 - precision
            cutoff_index = tf.argmax(fdr <= fdr_cutoff)
            fdr_at_cutoff = recall[cutoff_index]
            if not tf.is_nan(fdr_at_cutoff):
                fdr_array.append(tf.math.unsorted_segment_mean(fdr_at_cutoff))
        except:
            pass
    
    fdr_array = tf.convert_to_tensor(fdr_array)
    mean_fdr = tf.reduce_mean(fdr_array)
    median_fdr = tf.reduce_median(fdr_array)
    var_fdr = tf.reduce_variance(fdr_array)
    return mean_fdr, median_fdr, var_fdr, fdr_array

def compute_aupr(all_targets, all_predictions):
    aupr_array = []
    for i in range(all_targets.shape[1]):
        precision, recall, thresholds = tf.metrics.precision_recall_curve(all_targets[:, i], all_predictions[:, i], pos_label=1)
        auPR = tf.metrics.auc(recall, precision)
        if not tf.is_nan(auPR):
            aupr_array.append(tf.math.unsorted_segment_mean(auPR))
    aupr_array = tf.convert_to_tensor(aupr_array)
    mean_aupr = tf.reduce_mean(aupr_array)
    median_aupr = tf.reduce_median(aupr_array)
    var_aupr = tf.reduce_variance(aupr_array)
    return mean_aupr, median_aupr, var_aupr, aupr_array

def compute_auc(all_targets, all_predictions):
    auc_array = []
    for i in range(all_targets.shape[1]):
        try:  
            auROC = tf.metrics.roc_auc_score(all_targets[:, i], all_predictions[:, i])
            auc_array.append(auROC)
        except ValueError:
            pass
    auc_array = tf.convert_to_tensor(auc_array)
    mean_auc = tf.reduce_mean(auc_array)
    median_auc = tf.reduce_median(auc_array)
    var_auc = tf.reduce_variance(auc_array)
    return mean_auc, median_auc, var_auc, auc_array

def compute_metrics(predictions, targets, threshold, all_metrics=True):
    all_targets = tf.convert_to_tensor(targets)
    all_predictions = tf.convert_to_tensor(predictions)

    if all_metrics:
        meanAUC, medianAUC, varAUC, allAUC = compute_auc(all_targets, all_predictions)
        meanAUPR, medianAUPR, varAUPR, allAUPR = compute_aupr(all_targets, all_predictions)
        meanFDR, medianFDR, varFDR, allFDR = compute_fdr(all_targets, all_predictions)
    else:
        meanAUC, medianAUC, varAUC, allAUC = 0, 0, 0, 0
        meanAUPR, medianAUPR, varAUPR, allAUPR = 0, 0, 0, 0
        meanFDR, medianFDR, varFDR, allFDR = 0, 0, 0, 0

    p_at_1 = ranking_precision_score(all_targets, all_predictions, k=1)
    p_at_3 = ranking_precision_score(all_targets, all_predictions, k=3)
    p_at_5 = ranking_precision_score(all_targets, all_predictions, k=5)

    optimal_threshold = threshold
    all_predictions = tf.where(all_predictions < optimal_threshold, 0, 1)

    acc_ = subset_accuracy(all_targets, all_predictions, axis=1, per_sample=True)
    hl_ = hamming_loss(all_targets, all_predictions, axis=1, per_sample=True)
    exf1_ = example_f1_score(all_targets, all_predictions, axis=1, per_sample=True)
    ACC = tf.reduce_mean(acc_)
    hl = tf.reduce_mean(hl_)
    HA = 1 - hl
    ebF1 = tf.reduce_mean(exf1_)
    tp, fp, fn = compute_tp_fp_fn(all_targets, all_predictions, axis=0)

    miF1 = f1_score_from_stats(tp, fp, fn, average='micro')
    maF1 = f1_score_from_stats(tp, fp, fn, average='macro')

    metrics_dict = {
        'ACC': ACC,
        'HA': HA,
        'ebF1': ebF1,
        'miF1': miF1,
        'maF1': maF1,
        'meanAUC': meanAUC,
        'medianAUC': medianAUC,
        'varAUC': varAUC,
        'allAUC': allAUC,
        'meanAUPR': meanAUPR,
        'medianAUPR': medianAUPR,
        'varAUPR': varAUPR,
        'allAUPR': allAUPR,
        'meanFDR': meanFDR,
        'medianFDR': medianFDR,
        'varFDR': varFDR,
        'allFDR': allFDR,
        'p_at_1': p_at_1,
        'p_at_3': p_at_3,
        'p_at_5': p_at_5
    }

    return metrics_dict


def compute_best_metrics(all_indiv_prob, all_label, THRESHOLDS):
    best_val_metrics = None
    METRICS = ['ACC', 'HA', 'ebF1', 'miF1', 'maF1', 'meanAUC', 'meanAUPR', 'meanFDR']

    for threshold in THRESHOLDS:
        val_metrics = compute_metrics(all_indiv_prob, all_label, threshold, all_metrics=True)

        if best_val_metrics is None:
            best_val_metrics = {metric: val_metrics[metric] for metric in METRICS}
        else:
            for metric in METRICS:
                if 'FDR' in metric:
                    best_val_metrics[metric] = tf.minimum(best_val_metrics[metric], val_metrics[metric])
                else:
                    best_val_metrics[metric] = tf.maximum(best_val_metrics[metric], val_metrics[metric])
    return best_val_metrics

