import numpy as np
from scipy import stats
from sklearn import metrics
import torch
from matplotlib import pyplot as plt
import sklearn as sk

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def calculate_stats(output, target, conf_print):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []
    if conf_print:
    #conf_matrix = create_confusion_matrix(np.argmax(target, 1), np.argmax(output, 1), classes)
        confusion = np.zeros((classes_num, classes_num), dtype=int)
        confusion_current = sk.metrics.confusion_matrix(np.argmax(target, 1), np.argmax(output, 1))
        confusion = confusion + confusion_current
        print('Confusion matrix:\n', confusion_current, '\n')
    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc,
                # note acc is not class-wise, this is just to keep consistent with other metrics
                'acc': acc
                }
        stats.append(dict)

    return stats

def create_confusion_matrix(y_true, y_pred, classes):
    """ creates and plots a confusion matrix given two list (targets and predictions)
    :param list y_true: list of all targets (in this case integers bc. they are indices)
    :param list y_pred: list of all predictions (in this case one-hot encoded)
    :param dict classes: a dictionary of the countries with they index representation
    """

    amount_classes = len(classes)

    confusion_matrix = np.zeros((amount_classes, amount_classes))
    for idx in range(len(y_true)):
        target = y_true[idx][0]

        output = y_pred[idx]
        output = list(output).index(max(output))

        confusion_matrix[target][output] += 1
    return confusion_matrix
    # fig, ax = plt.subplots(1)
    #
    # ax.matshow(confusion_matrix)
    # ax.set_xticks(np.arange(len(list(classes.keys()))))
    # ax.set_yticks(np.arange(len(list(classes.keys()))))
    #
    # ax.set_xticklabels(list(classes.keys()))
    # ax.set_yticklabels(list(classes.keys()))
    #
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    # plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # plt.savefig('hty.png')
    # plt.show()