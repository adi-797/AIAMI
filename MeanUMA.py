"""
This function calculates the mean for the descriptors. It expects the output from the lowLevelSpectralExtractor or silenceErase.
"""

import numpy as np


def calculate_mean(described):
    output = {}  # prepare output
    normal_length = len(described['mfcc'][0])

    for key in described.keys():
        condition = (np.array(described[key])).size / float(normal_length)  # checks the size of the descriptor.
        if condition > 1:  # The descriptions is divided into frequencies
            output[key] = [0] * int(condition)
            for index in range(int(condition)):  # Iteration through all the frequencies
                if not described[key][index].any():
                    output[key][index] = 0
                    continue  # there's an exception in numpy if the whole range is zero.
                output[key][index] = np.average(described[key][index]).tolist()
        elif condition <= 1 < np.size(described[key]):
            output[key] = np.average(described[key]).tolist()
        else:
            output[key] = np.array(described[key]).tolist()

    return output
