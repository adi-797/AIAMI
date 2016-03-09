# -*- coding: utf-8 -*-

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

from utils.FScore import F1Score
from Identification.LoadDescriptors import load_all_descriptors
from Identification.PreprocessingDescriptors import preprocess
from Identification.TrainCvTest import separate_databases

Descriptors = load_all_descriptors(reverbs=False)
normalized_features, yClass, features_names = preprocess(Descriptors)
del Descriptors  # Ya no lo voy a utilizar
normalizedTrain, yTrain, normalizedCV, yCV, normalizedTest, yTest = separate_databases(normalized_features, yClass)

C = np.linspace(0.5, 2.5, 10)
Gamma = np.linspace(1e-2, 0.2, 10)


def test_c_gamma(training_features, training_classes, test_features, test_classes, c, gamma):
    f_cv = []
    f_train = []
    for iii in c:
        for jjj in gamma:
            clf = svm.SVC(C=iii, gamma=jjj, cache_size=1000)
            clf.fit(training_features, training_classes)
            f_train = np.append(f_train, np.mean(F1Score(training_features, training_classes, clf).values()))
            f_cv = np.append(f_cv, np.mean(F1Score(test_features, test_classes, clf).values()))
        print iii
    f_cv_matrix = np.reshape(f_cv, (len(C), len(gamma)))
    f_train_matrix = np.reshape(f_train, (len(C), len(gamma)))
    return f_train_matrix, f_cv_matrix


F1TrainScoreMatrix, F1CVScoreMatrix = test_c_gamma(training_features=normalizedTrain, training_classes=yTrain,
                                                   test_features=normalizedCV, test_classes=yCV, c=C, gamma=Gamma)

first_dimension = np.array([C] * F1TrainScoreMatrix.shape[1]).T
second_dimension = np.array([Gamma] * F1TrainScoreMatrix.shape[0])


fig = plt.figure()
ax = fig.gca(projection='3d')

jet = plt.get_cmap('jet')
# surf = ax.plot_surface(second_dimension, first_dimension, F1CVScoreMatrix,
#                        rstride=1, cstride=1, cmap=jet, linewidth=0, antialiased=False)

surf = ax.plot_surface(second_dimension, first_dimension, F1CVScoreMatrix,
                       rstride=1, cstride=1, cmap=jet, linewidth=0, antialiased=False)

ax.set_zlim3d(0, 1)
#title = unicode('Evaluación de los parámetros de\nregularización C y $\Gamma$ sobre la base de\nvalidación cruzada','utf-8')
#ax.set_title(title, fontsize=20)
ax.set_xlabel('$\Gamma$', fontsize=20)
ax.set_ylabel('C', fontsize=20)
plt.show()


# fig = plt.figure()
# ax = plt.plot(first_dimension[:, 1], F1CVScoreMatrix[:, 1], 'r', first_dimension[:, 1], F1TrainScoreMatrix[:, 1], 'b')
# plt.xlabel('C', fontsize=30)
# plt.text(1.9, 0.95, 'Medida-F en la base\nde entrenamiento',
#         color='blue', fontsize=25)
# text = unicode('Medida-F en la base\nde validacion cruzada', 'utf-8')
# plt.text(1.9, 0.9, text,
#         color='red', fontsize=25)
# plt.tick_params(axis='both', which='major', labelsize=15)
# plt.tick_params(axis='both', which='minor', labelsize=15)
# plt.setp(ax, linewidth=5.0)
# plt.show()
