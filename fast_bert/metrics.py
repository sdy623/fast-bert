from sklearn.metrics import (
    roc_curve,
    auc,
    hamming_loss,
    accuracy_score,
    confusion_matrix as sklearn_confusion_matrix,
)
import numpy as np
from torch import Tensor, argmax

import pdb
import logging

logger = logging.getLogger()

CLASSIFICATION_THRESHOLD: float = 0.5  # Best keep it in [0.0, 1.0] range

# def accuracy(out, labels):
#     outputs = np.argmax(out, axis=1)
#     return np.sum(outputs == labels)


def accuracy(y_pred: Tensor, y_true: Tensor, **kwargs):
    y_pred = y_pred.cpu()
    outputs = np.argmax(y_pred, axis=1)
    return np.mean(outputs.numpy() == y_true.detach().cpu().numpy())

@deprecated(reason="This function will be removed in future versions, use another function instead.")
def accuracy_multilabel(y_pred: Tensor, y_true: Tensor, sigmoid: bool = True, **kwargs):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    outputs = np.argmax(y_pred, axis=1)
    real_vals = np.argmax(y_true, axis=1)
    return np.mean(outputs.numpy() == real_vals.numpy())

def accuracy_multilabel_macro(
        y_pred: Tensor,
        y_true: Tensor,
        thresh: float = CLASSIFICATION_THRESHOLD,
        sigmoid: bool = True, **kwargs
):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    outputs = (y_pred > thresh).float()
    real_vals = y_true.float()
    correct_per_class = (outputs == real_vals).float().sum(dim=0) / real_vals.sum(dim=0) # TODO: Check if this is correct
    return correct_per_class.mean().item()

def accuracy_thresh(
    y_pred: Tensor,
    y_true: Tensor,
    thresh: float = CLASSIFICATION_THRESHOLD,
    sigmoid: bool = True,
    **kwargs
):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid:
        y_pred = y_pred.sigmoid()
    return ((y_pred > thresh) == y_true.bool()).float().mean().item()


#     return np.mean(((y_pred>thresh)==y_true.byte()).float().cpu().numpy(), axis=1).sum()


def fbeta(
    y_pred: Tensor,
    y_true: Tensor,
    thresh: float = 0.3,
    beta: float = 2,
    eps: float = 1e-9,
    sigmoid: bool = True,
    **kwargs
):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid:
        y_pred = y_pred.sigmoid()

    if len(kwargs['labels']) <= 2:
        y_pred = argmax(y_pred, dim=1)
        y_true = y_true.float()
        TP = (y_pred * y_true).sum()
        prec = TP / (y_pred.sum() + eps) # prevent division by zero error
        rec = TP / (y_true.sum() + eps)
    else:
        y_pred = (y_pred > thresh).float()
        y_true = y_true.float()
        TP = (y_pred * y_true).sum(dim=1)
        prec = TP / (y_pred.sum(dim=1) + eps)
        rec = TP / (y_true.sum(dim=1) + eps)
    res = (prec * rec) / (prec * beta2 + rec + eps) * (1 + beta2)
    return res.mean().item()


def roc_auc(y_pred: Tensor, y_true: Tensor, **kwargs):
    # ROC-AUC calcualation
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_true = y_true.detach().cpu().numpy()
    if len(kwargs['labels']) <= 2:
        y_pred = y_pred.sigmoid()[:,-1].detach().cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    '''
    print(y_true.shape)
    print(y_true.ravel())
    print(y_pred.shape)
    print(y_pred.ravel())
    '''
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    roc_auc['fpr'] = fpr["micro"]
    roc_auc["tpr"] = tpr["micro"]
    return roc_auc


def Hamming_loss(
    y_pred: Tensor,
    y_true: Tensor,
    sigmoid: bool = True,
    thresh: float = CLASSIFICATION_THRESHOLD,
    sample_weight=None,
    **kwargs
):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    return hamming_loss(y_true, y_pred, sample_weight=sample_weight)


def Exact_Match_Ratio(
    y_pred: Tensor,
    y_true: Tensor,
    sigmoid: bool = True,
    thresh: float = CLASSIFICATION_THRESHOLD,
    normalize: bool = True,
    sample_weight=None,
    **kwargs
):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    return accuracy_score(
        y_true, y_pred, normalize=normalize, sample_weight=sample_weight
    )


def F1(
    y_pred: Tensor,
    y_true: Tensor,
    threshold: float = CLASSIFICATION_THRESHOLD,
    **kwargs
):
    return fbeta(y_pred, y_true, thresh=threshold, beta=1, **kwargs)


def confusion_matrix(y_pred: Tensor, y_true: Tensor, **kwargs):
    try:
        y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
        return sklearn_confusion_matrix(
            y_true.detach().cpu().numpy(), y_pred
        )
    except Exception as e:
        logger.error(e)

