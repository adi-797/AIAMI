import numpy as np

from Identification import TrainCvTest, svm
from utils import FScore


def test_repe(times, normalized_features, yClass):
    every_f1 = {}
    array_means = np.array([])
    array_vars = np.array([])
    iii = 0

    while iii < times:
        x_train, y_train, x_cv, y_cv, x_test, y_test = TrainCvTest. \
            separateDatabases(normalized_features, yClass)
        clf = svm.trainSVM(x_train, y_train, call=True)
        F1Test = FScore.F1Score(x_cv, y_cv, clf)
        mTest = np.array([])
        for key in F1Test.keys():
            if key not in every_f1:
                every_f1[key] = F1Test[key]
            else:
                every_f1[key] = np.append(every_f1[key], F1Test[key])
            mTest = np.append(mTest, F1Test[key])

        array_means = np.append(array_means, mTest.mean())
        array_vars = np.append(array_vars, mTest.var())
        iii += 1

    return every_f1, array_means, array_vars


def test_repe_train_and_cv(times, features, classes):
    every_f1_train = {}
    every_f1_cv = {}
    array_means_train = np.array([])
    array_vars_train = np.array([])
    array_means_cv = np.array([])
    array_vars_cv = np.array([])
    iii = 0

    while iii < times:
        x_train, y_train, x_cv, y_cv, x_test, y_test = TrainCvTest. \
            separateDatabases(features, classes)
        clf = svm.trainSVM(x_train, y_train, call=True)

        F1Train = FScore.F1Score(x_train, y_train, clf)
        mTest = np.array([])
        for key in F1Train.keys():
            if key not in every_f1_cv:
                every_f1_train[key] = F1Train[key]
            else:
                every_f1_train[key] = np.append(every_f1_train[key], F1Train[key])
            mTest = np.append(mTest, F1Train[key])

        array_means_train = np.append(array_means_train, mTest.mean())
        array_vars_train = np.append(array_vars_train, mTest.var())

        F1CV = FScore.F1Score(x_cv, y_cv, clf)
        mCV = np.array([])
        for key in F1CV.keys():
            if key not in every_f1_cv:
                every_f1_cv[key] = F1CV[key]
            else:
                every_f1_cv[key] = np.append(every_f1_cv[key], F1CV[key])
            mCV = np.append(mCV, F1CV[key])

        array_means_cv = np.append(array_means_cv, mCV.mean())
        array_vars_cv = np.append(array_vars_cv, mCV.var())
        iii += 1

    return every_f1_train, every_f1_cv, array_means_train, \
           array_vars_train, array_means_cv, array_vars_cv


def test_repe_test(times, features, classes):
    every_f1_test = {}
    array_means_test = np.array([])
    array_vars_test = np.array([])
    iii = 0

    while iii < times:
        x_train, y_train, x_cv, y_cv, x_test, y_test = TrainCvTest. \
            separateDatabases(features, classes)
        clf = svm.trainSVM(x_train, y_train, call=True)

        F1Test = FScore.F1Score(x_test, y_test, clf)
        mTest = np.array([])
        for key in F1Test.keys():
            if key not in every_f1_test:
                every_f1_test[key] = F1Test[key]
            else:
                every_f1_test[key] = np.append(every_f1_test[key], F1Test[key])
            mTest = np.append(mTest, F1Test[key])

        array_means_test = np.append(array_means_test, mTest.mean())
        array_vars_test = np.append(array_vars_test, mTest.var())
        iii += 1

    return every_f1_test, array_means_test, array_vars_test


# Descriptors = load_descriptors('Inf', reverbs=True)
# normalized_features, yClass, features_names = preprocess(Descriptors)
# del Descriptors
# every_f1_test, array_means_test, array_vars_test = test_repe_test(100, normalized_features, yClass)
