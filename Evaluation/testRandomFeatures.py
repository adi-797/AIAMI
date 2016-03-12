# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from Identification.LoadDescriptors import loadAllDescriptors
from Identification.PreprocessingDescriptors import preprocessDescriptors
from testRepe import test_repe_train_and_cv


def test_features(features_vector, classes_vector):
    features = np.arange(1, features_vector.shape[1], 1)
    print len(features)
    f1_train_per_feature = np.array([])
    f1_cv_per_feature = np.array([])

    np.random.shuffle(features_vector.T)  # shuffle columns

    for iii in features:
        f1_train, f1_cv, a, b, c, d = test_repe_train_and_cv(10, features_vector[:, -iii:], classes_vector)
        f1_train_per_feature = np.append(f1_train_per_feature, np.mean(f1_train.values()))
        f1_cv_per_feature = np.append(f1_cv_per_feature, np.mean(f1_cv.values()))

    return f1_cv_per_feature, f1_train_per_feature, features

Descriptors = loadAllDescriptors(reverbs=True)
normalized_features, yClass, features_names = preprocessDescriptors(Descriptors)
del Descriptors  # Ya no lo voy a utilizar
f1_cv, f1_train, features = test_features(normalized_features, yClass)


plt.xlabel("Cantidad de descriptores", fontsize=20)
plt.ylabel("Error", fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.text(10, 0.7, r'$C=2.77,\ \Gamma=0.0231$', fontsize=22)
plt.text(5, 0.1, 'Medida-F en la base\nde entrenamiento', fontsize=20, color='blue')
text = unicode('Medida-F en la base\nde validación cruzada', 'utf-8')
plt.text(15, 0.4, text, fontsize=20, color='green')
plt.plot(features, 1 - f1_train, c='blue', linewidth=4.0)
plt.plot(features, 1 - f1_cv, color='green', linewidth=4.0)
plt.show()

