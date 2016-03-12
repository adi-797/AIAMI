"""
En este script se calcula los F1 Score para cada instrumento individualmente
"""

import numpy as np

from Identification import svm


def intersection(first, second):
    output = np.array([iii for iii in first if iii in second])
    return output


def true_pos(predictions, yData):
    predictions = np.array(predictions)
    yData = np.array(yData)

    yClass = existingClasses(yData)
    true_positive = {}
    positives = np.where(predictions == yData)[0]

    for iii in yClass:
        classPositives = np.where(iii == predictions)[0]
        true_positive[iii] = len(intersection(positives, classPositives))

    return true_positive


def false_pos(predictions, yData):
    predictions = np.array(predictions)
    yData = np.array(yData)

    YClass = existingClasses(yData)
    false_positive = {}
    falses = np.where(predictions != yData)[0]

    for iii in YClass:
        clas_pos = np.where(iii == predictions)[0]
        true_pos_clas = np.where(iii == yData)[0]
        false_positive[iii] = len(set(clas_pos).difference(true_pos_clas))

    return false_positive


def false_neg(predictions, yData):
    false_negative = {}
    true_positive = true_pos(predictions, yData)

    for key in true_positive:
        trues = np.where(yData == key)[0]
        false_negative[key] = len(trues) - true_positive[key]

    return false_negative


def precision(XData, YData, clf=1):
    if clf == 1:
        clf = svm.trainSVM(XData, YData, True)

    predictions = []
    for iii in np.arange(YData.size):
        predictions = np.append(predictions, clf.predict(XData[iii, :])[0])

    tp = true_pos(predictions, YData)
    fp = false_pos(predictions, YData)

    precision_ = {}

    for key in tp.keys():
        try:
            precision_[key] = tp[key] / float(tp[key] + fp[key])
        except:
            precision_[key] = -1

    return precision_


def recall(XData, YData, clf=1):
    if clf == 1:
        clf = svm.trainSVM(XData, YData, True)

    predictions = []
    for iii in np.arange(YData.size):
        predictions = np.append(predictions, clf.predict(XData[iii, :])[0])

    tp = true_pos(predictions, YData)
    fn = false_neg(predictions, YData)

    recall = {}

    for key in tp.keys():
        recall[key] = tp[key] / float(tp[key] + fn[key])

    return recall


def existingClasses(yData):
    classes = []
    for iii in yData:
        if (iii not in classes):
            classes = np.append(classes, iii)

    return classes


def F1Score(XData, YData, clf=1):
    P = precision(XData, YData, clf)
    R = recall(XData, YData, clf)

    F1Score = {}
    for key in P.keys():
        if P[key] == -1:
            F1Score[key] = -1
        elif (P[key] == 0) and (R[key] == 0):
            F1Score[key] = 0
            continue
        else:
            F1Score[key] = 2 * ((P[key] * R[key]) / (P[key] + R[key]))

    return F1Score
