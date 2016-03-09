import numpy as np

from Identification.LoadDescriptors import load_descriptors
from Identification.PreprocessingDescriptors import preprocess
from Evaluation.testRepe import test_repe

__author__ = 'andres'


def count_instruments(class_vector):
    count = {}
    for i in set(class_vector):
        number = len(np.where(i == class_vector)[0])
        count[i] = number
    return count


def mean_per_instrument(dictionary_f1):
    mean = dict()
    for key in dictionary_f1.keys():
        mean[key] = np.mean(dictionary_f1[key])
    return mean


def std_per_instrument(dictionary_f1):
    standard_deviation = dict()
    for key in dictionary_f1.keys():
        standard_deviation[key] = np.std(dictionary_f1[key])
    return standard_deviation


def run_stats_analysis(maximum='Inf', reverbs=True):
    Descriptors = load_descriptors(maximum, reverbs)
    normalized_features, yClass, features_names = preprocess(Descriptors)
    every_f1_test, mMean, mVar = test_repe(100, normalized_features, yClass)
    instrument_count = count_instruments(yClass)
    mean = mean_per_instrument(every_f1_test)
    stand = std_per_instrument(every_f1_test)

    return mean, stand, instrument_count
