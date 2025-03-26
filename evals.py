import sys
from sklearn import metrics
import math
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
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
    Y_true = tf.reshape(Y_true, [-1])  # Ensure Y_true is 1D
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

def subset_accuracy(true_targets, predictions, axis=1, per_sample=False):
    true_targets = tf.cast(true_targets, tf.int32)  # Ensure both are int32
    predictions = tf.cast(predictions, tf.int32)

    result = tf.equal(true_targets, predictions)  # This returns a boolean tensor
    
    if per_sample:
        return tf.cast(result, tf.float32)  # Return per-sample accuracy (True=1, False=0)
    else:
        return tf.reduce_all(result, axis=axis)  # For overall accuracy across the specified axis


def hamming_loss(true_targets, predictions, axis=1, per_sample=False):
    true_targets = tf.cast(true_targets, tf.bool)  # Cast to boolean
    predictions = tf.cast(predictions, tf.bool)  # Cast to boolean

    result = tf.math.logical_xor(true_targets, predictions)  # XOR operation

    if per_sample:
        return tf.cast(result, tf.float32)  # Return per-sample loss (True=1, False=0)
    else:
        return tf.reduce_mean(tf.cast(result, tf.float32), axis=axis)  # Mean loss across axis

def compute_tp_fp_fn(true_targets, predictions, axis=1):
    # Convert to float32 for arithmetic operations
    true_targets = tf.cast(true_targets, tf.float32)
    predictions = tf.cast(predictions, tf.float32)

    tp = tf.reduce_sum(true_targets * predictions, axis=axis)  # True Positives
    fp = tf.reduce_sum((1 - true_targets) * predictions, axis=axis)  # False Positives
    fn = tf.reduce_sum(true_targets * (1 - predictions), axis=axis)  # False Negatives

    return tp, fp, fn


def compute_median(tensor):
    sorted_tensor = tf.sort(tensor)
    num_elements = tf.shape(sorted_tensor)[0]

    # If tensor is empty, return a default value
    if tf.equal(num_elements, 0):
        return tf.constant(0.0, dtype=tf.float32)

    middle_idx = num_elements // 2
    if num_elements % 2 == 0:
        return (sorted_tensor[middle_idx - 1] + sorted_tensor[middle_idx]) / 2
    else:
        return sorted_tensor[middle_idx]

def example_f1_score(true_targets, predictions, axis=1, per_sample=False):
    true_targets = tf.cast(true_targets, tf.float32)
    predictions = tf.cast(predictions, tf.float32)

    tp, fp, fn = compute_tp_fp_fn(true_targets, predictions, axis=axis)

    denominator = tf.reduce_sum(true_targets, axis=axis) + tf.reduce_sum(predictions, axis=axis)
    denominator = tf.where(denominator == 0, tf.ones_like(denominator), denominator)

    f1 = (2 * tp) / denominator

    if per_sample:
        return f1  # return per-sample F1 scores
    else:
        return tf.reduce_mean(f1)  # return the mean F1 score


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
    median_fdr = compute_median(fdr_array)
    var_fdr = tf.math.reduce_variance(fdr_array)
    return mean_fdr, median_fdr, var_fdr, fdr_array

def compute_aupr(all_targets, all_predictions):
    aupr_array = []
    for i in range(all_targets.shape[1]):
        if np.sum(all_targets[:, i]) == 0:  # No positive class
            auPR = 0.0  # Assign default value
        else:
            precision, recall, _ = precision_recall_curve(all_targets[:, i], all_predictions[:, i], pos_label=1)
            auPR = auc(recall, precision)
        
        if not math.isnan(auPR):
            aupr_array.append(np.nan_to_num(auPR))
    aupr_array = tf.convert_to_tensor(aupr_array)
    mean_aupr = tf.reduce_mean(aupr_array)
    median_aupr = compute_median(aupr_array)
    var_aupr = tf.math.reduce_variance(aupr_array)
    return mean_aupr, median_aupr, var_aupr, aupr_array



def compute_auc(all_targets, all_predictions):
    auc_array = []
    for i in range(all_targets.shape[1]):
        try:  
            auROC = roc_auc_score(all_targets[:, i], all_predictions[:, i])
            auc_array.append(auROC)
        except ValueError:
            pass
    auc_array = tf.convert_to_tensor(auc_array)
    mean_auc = tf.reduce_mean(auc_array)
    median_auc = compute_median(auc_array)
    var_auc = tf.math.reduce_variance(auc_array)
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

