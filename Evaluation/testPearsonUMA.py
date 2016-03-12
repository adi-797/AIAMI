import os

import essentia.standard

from Description import extracter
from utils import JsonWriter, Pearson
from Evaluation import Pearson

inputDir = "UMAPiano-DB-Poly-1"
descExt = '.wav'  # la extension que busco

exception = {}
meanCorr = {}
for path, dname, fnames in os.walk(
        inputDir):  # dname directories names in current directory, fnames file names in current directory.
    for fname in fnames:
        if descExt in fname.lower():  # si esta el archivo tiene la extension que busco...
            new_pid = os.fork()
            if new_pid == 0:  # Si new_pid == 0, entramos en el forked process.
                try:  # uso un try..except..finally para asegurarme de _siempre_ terminar el proceso
                    fDict = essentia.standard.MonoLoader(downmix="left", filename=path + "/" + fname)()  # lo cargo
                    extracted = extracter.extractAllDescriptors(
                        fDict)  # le aplico los descriptores, aca me vuelve un Pool de Essentia
                    correlation = Pearson.correlate(extracted)
                    meanCorr[fname] = correlation
                    JsonWriter.write_json(path + '/Correlation', fname,
                                          correlation)  # Guardo los resultados en archivos .json

                except Exception:
                    exception[fname] = ['oops']  # Me fijo si hubo alguna excepcion
                    # pass
                finally:
                    os._exit(0)  # Cierro el fork y vuelvo al proceso padre
            else:
                child = new_pid
            os.waitpid(child, 0)  # evito crear procesos paralelos, lo que limita la performance de mi programa
