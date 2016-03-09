""" Este script carga la informacion que guarde en la etapa anterior en los archivos json
"""

import os
import json

descExt = '.json'


def loadUMA(inputDir):
    data_details = {}

    for path, dname, fnames in os.walk(inputDir):
        for fname in fnames:
            if descExt in fname.lower():
                print fname
                fDict = json.load(open(os.path.join(path, fname), 'r'))
                data_details[fname] = {'file': fname, 'feature': fDict}

    return data_details
