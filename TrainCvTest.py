""" Este script separa la base de datos en 3 para asi poder tener la de entrenamiento, la de validacion cruzada y la de test.
"""

import numpy as np


def separate_databases(features_database, class_database):
    features_train = []
    class_train = []
    features_cross_validation = []
    class_cross_validation = []
    features_test = []
    class_test = []

    for aClass in set(class_database):
        idx = np.where(aClass == class_database)[0]
        np.random.shuffle(idx)
        idxMax = len(idx)
        idxMaxTrain = int(0.7 * idxMax)  # int transforma en entero con floor
        idxTrain = idx[:idxMaxTrain]
        idxMaxCV = int(0.9 * idxMax)
        idxCV = idx[idxMaxTrain:idxMaxCV]
        idxTest = idx[idxMaxCV:]

        features_train = np.append(features_train, features_database[idxTrain])
        class_train = np.append(class_train, class_database[idxTrain])
        features_cross_validation = np.append(features_cross_validation, features_database[idxCV])
        class_cross_validation = np.append(class_cross_validation, class_database[idxCV])
        features_test = np.append(features_test, features_database[idxTest])
        class_test = np.append(class_test, class_database[idxTest])

    features_train = np.reshape(features_train, (len(class_train), -1))
    features_cross_validation = np.reshape(features_cross_validation, (len(class_cross_validation), -1))
    features_test = np.reshape(features_test, (len(class_test), -1))

    return features_train, class_train, features_cross_validation, class_cross_validation, features_test, class_test
