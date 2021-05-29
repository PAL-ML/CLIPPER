import torch 
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics

def calculate_single_label_accuracy(pred, y, label_mapping):
    x = 0
    for i, p in enumerate(pred):
        pred_label = label_mapping[p]
        if pred_label == y[i]:
            x += 1
        else:
          print("Incorrect label pairs (ground truth, pred): {}, {}".format(y[i], pred_label))
    x = x/(i+1)

    return x

def accuracy(output, target, topk=(1,)):
    #pred1 = np.argpartition(output.cpu().numpy()[0], -K)[-K:]
    #print("pred1: {}".format(pred1))
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #print("pred: {}".format(pred))
    #print("target: {}".format(target.view(1, -1).expand_as(pred)))
    #print("correct: {}".format(correct))
    #print("########")
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]
    
def accuracy_score(pred, groundtruth):
    return metrics.accuracy_score(groundtruth, pred, average='weighted')

def precision(pred, groundtruth):
    return metrics.precision_score(groundtruth, pred, average='weighted')

def recall(pred, groundtruth):
    return metrics.recall_score(groundtruth, pred, average='weighted')

def acc_prec_recall_fscore_support(pred_y, eval_y, clip_label_mapping, average="weighted"):
    acc = calculate_single_label_accuracy(pred_y, eval_y, clip_label_mapping)
    pred = list(map(lambda x: clip_label_mapping[x], pred_y))
    recall = metrics.recall_score(eval_y, pred, average=average)
    precision = metrics.precision_score(eval_y, pred, average=average)
    f_score = metrics.f1_score(eval_y, pred, average=average)
    return acc, recall, precision, f_score

def compute_fp_fn_tp_tn(preds, groundtruths, label_order=None):
    cnf_matrix = metrics.confusion_matrix(groundtruths, preds, labels=label_order)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    return TP, FP, TN, FN

def sensitivity(TP, FP, TN, FN):
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    return TPR

def specificity(TP, FP, TN, FN):
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    return TNR

def negative_predictive_value(TP, FP, TN, FN):
    # Negative predictive value
    NPV = TN/(TN+FN)
    return NPV

def false_positive_rate(TP, FP, TN, FN):
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    return FPR

def false_negative_rate(TP, FP, TN, FN):
    # False negative rate
    FNR = FN/(TP+FN)
    return FNR

def false_discovery_rate(TP, FP, TN, FN):
    # False discovery rate
    FDR = FP/(TP+FP)
    return FDR

def compute_multi_label_metrics(
    eval_x, 
    eval_y, 
    clip_label_mapping, 
    clip_wi_cls,
    threshold=0.7,
    metrics=['jaccard', 'hamming']
  ):
  metrics_values = clip_wi_cls.evaluate_multi_label_metrics(
      eval_x, eval_y, clip_label_mapping, threshold, metrics
      )
  m = tuple([metrics_values[m] for m in metrics])
  return m

def compute_jaccard_index(clip_preds, wi_preds):
    jaccard_index = 0

    if len(clip_preds) == 0:
      return jaccard_index

    assert len(clip_preds) == len(wi_preds)

    for i, pred in enumerate(clip_preds):
        wi_list = set(wi_preds[i])
        clip_list = set(pred)

        len_intersection = len(wi_list.intersection(clip_list))
        len_union = len(wi_list.union(clip_list))

        jaccard_index += len_intersection/len_union

    jaccard_index = jaccard_index / len(clip_preds)

    return jaccard_index

def plot_metrics_by_threshold(
    metrics_thresholds, 
    thresholds, 
    metrics=['jaccard', 'hamming'],
    title_prefix=""
):
    legend = []

    for i, met in enumerate(metrics):
      mean_metric = [np.mean(m) for m in metrics_thresholds[i]]
      opt_threshold = thresholds[np.argmax(mean_metric)]
      plt.plot(thresholds, mean_metric)
      plt.axvline(opt_threshold)
      legend.append(met)
      legend.append(opt_threshold)

    plt.xlabel('Threshold')
    plt.ylabel('Value')
    plt.legend(legend)
    plt.title(title_prefix+" Multi label metrics by threshold")
    plt.show()

def plot_num_removed_vs_threshold(
    list_num_removed, 
    thresholds, 
    title_prefix=""
):
    legend = ["Num images corrected"]

    plt.plot(thresholds, list_num_removed)

    plt.xlabel('Threshold')
    plt.ylabel('Num images corrected')
    plt.legend(legend)
    plt.title(title_prefix+" Num images corrected by threshold")
    plt.show()
   
