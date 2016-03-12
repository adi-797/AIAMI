# coding: utf-8
""" Este script toma la base de datos y normaliza los descriptores para que tengan varianza 1 y desviacion estandar 0
"""

import numpy as np


def preprocessDescriptors(descriptors):
    temp = []
    y = []  # vector con los nombres de los archivos en los mismos índices que su vector normalizado de descriptores
    y_class = []  # y un vector con las clases.
    features = []

    for key in descriptors.keys():
        for key2 in descriptors[key].keys():
            for key3 in descriptors[key][key2].keys():
                del (descriptors[key][key2][key3]['feature'][1]['Silence'])

                y = np.append(y, descriptors[key][key2][key3]['file'])
                y_class = np.append(y_class, key2.split('.')[0])
                features = np.append(features, descriptors[key][key2][key3]['feature'][1].keys())

                for key5 in descriptors[key][key2][key3]['feature'][1].keys():
                    temp = np.append(temp, descriptors[key][key2][key3]['feature'][1][key5])

    features = np.reshape(features, (len(y), -1))
    for iii in np.arange(features.shape[1]):
        if (features[:, iii] != features[0, iii]).any():
            raise Exception('Features are not well ordered')

    temp = np.reshape(temp, (len(y), -1))

    mu = np.mean(temp, axis=0)
    sigma = np.std(temp, axis=0)
    # sigma = np.amax(temp,axis=0)-np.amin(temp,axis=0)

    normalized = (temp - mu) / sigma

    return normalized, y_class, features
