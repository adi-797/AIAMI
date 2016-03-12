import json
import os
import numpy as np

from utils.FScore import F1Score
from Identification.LoadDescriptors import loadDescriptors
from Identification.PreprocessingDescriptors import preprocessDescriptors
from Identification.svm import trainSVM

__author__ = 'andres'

format_extension = '.json'

# DescribeSounds.get(folder_of_test_sounds, extension_of_test_sounds)


def load_description(input_dir, maximum='Inf', reverbs=True):
    data_details = {}
    maximum = float(maximum)
    count = 0
    for path, directory_name, file_names in os.walk(input_dir):
        for file_name in file_names:
            if format_extension in file_name.lower():
                if not reverbs and (file_name[0:3] == 'R1_'):
                    continue
                cname, sname = path.split('/')[-2], path.split('/')[-1]
                second_key = sname.split('.')[0]
                if second_key not in data_details:
                    data_details[second_key] = {}
                fDict = json.load(open(os.path.join(cname, sname, file_name), 'r'))
                data_details[second_key][file_name] = {'file': file_name, 'feature': fDict}
                count += 1
            if count >= maximum:
                break

    return data_details


def classify_new_sounds(folder_of_test_sounds, target_class):
    new_descriptors = {target_class: load_description(folder_of_test_sounds)}
    new_normalized_features, new_y_class, new_features_names = preprocessDescriptors(new_descriptors)
    new_y_class = np.array([target_class]*len(new_y_class))
    descriptors = loadDescriptors(maximum='Inf', reverbs=True)
    normalized_features, yClass, features_names = preprocessDescriptors(descriptors)
    clf = trainSVM(normalized_features, yClass, call=True)
    F1 = F1Score(new_normalized_features, new_y_class, clf)

    return F1



F1 = classify_new_sounds('testBass', 'Bass')
