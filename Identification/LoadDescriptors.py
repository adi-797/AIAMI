""" Este script carga la informacion que guarde en la etapa anterior en los archivos json
"""

import os
import json
from numpy.random import shuffle

format_extension = '.json'


def loadUMA(inputDir, reverbs=True, maximum='Inf'):
    data_details = {'Piano': {}}
    maximum = float(maximum)
    count = 0
    for path, directory_name, file_names in os.walk(inputDir):
        for file_name in file_names:
            if not reverbs and (file_name[0:3] == 'R1_'):
                continue
            if format_extension in file_name.lower():
                fDict = json.load(open(os.path.join(path, file_name), 'r'))
                data_details['Piano'][file_name] = {'file': file_name, 'feature': fDict}
                count += 1
                print file_name
            if count >= maximum:
                break

    return data_details


def loadIRMAS(inputDir="IRMAS-TrainingData", reverbs=True, maximum='Inf'):
    dataDetails = {}
    maximum = float(maximum)
    count = 0
    for path, directory_name, file_names in os.walk(inputDir):
        for file_name in file_names:
            if format_extension in file_name.lower():
                if not reverbs and (file_name[0:3] == 'R1_'):
                    continue
                cname, sname = path.split('/')[-2], path.split('/')[-1]
                second_key = sname.split('.')[0]
                if second_key not in dataDetails:
                    dataDetails[second_key] = {}
                fDict = json.load(open(os.path.join(cname, sname, file_name), 'r'))
                dataDetails[second_key][file_name] = {'file': file_name, 'feature': fDict}
                count += 1
            if count >= maximum:
                break

    return dataDetails


def loadIOWA(input_directory, reverbs=True, maximum='Inf'):
    dataDetails = {}
    maximum = float(maximum)
    for path, directory_name, file_names in os.walk(input_directory):
        count = 0
        for file_name in file_names:
            if format_extension in file_name.lower():
                if not reverbs and (file_name[0:3] == 'R1_'):
                    continue
                remain, rname, cname, sname = path.split('/')[:-3], path.split('/')[-3], \
                                                    path.split('/')[-2], path.split('/')[-1]
                second_key = sname.split('.')[0]
                if cname not in dataDetails:
                    dataDetails[cname] = {}
                if second_key not in dataDetails[cname]:
                    dataDetails[cname][second_key] = {}
                fDict = json.load(open(os.path.join('/'.join(remain), rname, cname, sname, file_name), 'r'))
                dataDetails[cname][second_key][file_name] = {'file': file_name, 'feature': fDict}
                print file_name
                count += 1
            if count >= maximum:
                break

    for key in dataDetails.keys():
        for key2 in dataDetails[key].keys():
            n_examples = len(dataDetails[key][key2].keys())
            if n_examples > maximum:
                index = range(int(n_examples))
                shuffle(index)
                key3 = []
                for iii in range(int(n_examples-maximum)):
                    key3.append(dataDetails[key][key2].keys()[index[iii]])
                for iii in key3:
                    del dataDetails[key][key2][iii]
                print key2
    return dataDetails


def loadDescriptors(maximum, reverbs=True):
    descritors = {}
    descriptors = loadIOWA("IOWAMIS", reverbs, maximum)
    descriptors['UMA'] = loadUMA("UMAPiano-DB-Poly-1", reverbs, maximum)
    # descriptors['Irmas'] = loadIRMAS(reverbs, maximum)
    return descriptors


def loadAllDescriptors(reverbs=True):
    descriptors = {}
    descriptors = loadIOWA("IOWAMIS", reverbs)
    descriptors['UMA'] = loadUMA("UMAPiano-DB-Poly-1", reverbs)
    # descriptors['Irmas'] = loadIRMAS(reverbs)
    return descriptors
