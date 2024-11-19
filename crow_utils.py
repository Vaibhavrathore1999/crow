from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment as linear_assignment

def accuracy(output, target):

    num_correct = np.sum(output == target)
    res = num_correct / len(target)

    return res

def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    if y_pred.size == 0:
        return torch.zeros(1) - 1

    D = max(y_pred.max(), y_true.max()) + 1
    
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_pred.size

def cluster_acc_2(y_pred_truemask, y_true_truemask, seen_num):
    """
    This function is used to calculate the unseen accuracy.

    The main difference to the founction cluster_acc is 
    this function will directly consider the samples from 
    sunseen but being classified into seen to be worng cases.
    """

    y_pred_truemask=y_pred_truemask.astype(np.int64)
    y_true_truemask=y_true_truemask.astype(np.int64)

    assert y_pred_truemask.size == y_pred_truemask.size
    if y_pred_truemask.size == 0:
        return torch.zeros(1).item()
    
    D = max(y_pred_truemask.max(), y_true_truemask.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred_truemask.size):
        if y_pred_truemask[i] > seen_num:
            w[y_pred_truemask[i], y_true_truemask[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    return w[row_ind, col_ind].sum() / y_true_truemask.size    

def entropy(x):

    EPS = 1e-8
    x_ =  torch.clamp(x, min = EPS)
    b =  x_ * torch.log(x_)

    if len(b.size()) == 2: # Sample-wise entropy
        return - b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


def split_cluster_acc_v2(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc