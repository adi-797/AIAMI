import numpy as np
import matplotlib.pyplot as plt

from Identification import LoadDescriptors, PreprocessingDescriptors

Descriptors = LoadDescriptors.load_all_descriptors()
normalized, yClass, features = PreprocessingDescriptors.preprocess(Descriptors)
del (Descriptors)

x = normalized[:, 18]
y = normalized[:, 2]

colors = np.ones(len(x))

colors[np.where('Bass' == yClass)] = 0.5

"""Chequear por qué los valores no van entre -1 y 1 y si deberían ir entre -1 y 1"""

plt.scatter(x, y, c=colors)
plt.show()
