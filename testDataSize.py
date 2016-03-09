# -*- coding: utf-8 -*-

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

from FScore import F1Score
from Identification.LoadDescriptors import load_all_descriptors
from Identification.PreprocessingDescriptors import preprocess
from Identification.TrainCvTest import separate_databases

Descriptors = load_all_descriptors(reverbs=True)
normalized_features, yClass, features_names = preprocess(Descriptors)
del Descriptors  # Ya no lo voy a utilizar
normalizedTrain, yTrain, normalizedCV, yCV, normalizedTest, yTest = separate_databases(normalized_features, yClass)


def test_data_size(training_features, training_classes, test_features, test_classes):
    index = np.arange(0, len(training_classes))
    np.random.shuffle(index)
    test_size = np.linspace(0.1, 1, 50) * len(index)
    test_size = [int(i) for i in test_size]
    f_train = []
    f_cv = []

    clf = svm.SVC(C=1.833, gamma=0.1366, cache_size=1000)

    for iii in test_size:
        clf.fit(training_features[index[0:iii]], training_classes[index[0:iii]])

        f_train = np.append(f_train, np.mean(F1Score(training_features[index[0:iii]],
                                                     training_classes[index[0:iii]], clf).values()))
        f_cv = np.append(f_cv, np.mean(F1Score(test_features, test_classes, clf).values()))

    return f_train, f_cv, test_size


F1Train, F1CV, testSize = test_data_size(normalizedTrain, yTrain, normalizedCV, yCV)

plt.xlabel("Cantidad de muestras", fontsize=20)
plt.ylabel("Error",fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.text(2000, 0.3, r'$C=1.833,\ \Gamma=0.1366$', fontsize=22)
plt.text(1000, 0.07, 'Medida-F en la base\nde entrenamiento', fontsize=20, color='blue')
text = unicode('Medida-F en la base\nde validación cruzada', 'utf-8')
plt.text(1000, 0.25, text, fontsize=20, color='green')
plt.plot(testSize, 1 - F1Train, c='blue', linewidth=4.0)
plt.plot(testSize, 1 - F1CV, color='green', linewidth=4.0)
plt.show()

