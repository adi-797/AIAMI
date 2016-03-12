""" 
DescribeSounds realiza las descripciones de los sonidos.
Los extrae para cada archivo y los escala entre si.
Luego, los guarda en un archivo Json para la siguiente etapa del proyecto
"""

import os

import essentia.standard

import extracter
from utils import MeanCalculator, JsonWriter


def calculateDescriptorAndWriteToFile(database_directory, format_extension):
    exception = {}
    for path, directory_name, file_names in os.walk(database_directory):
        for file_name in file_names:
            if format_extension in file_name.lower():
                new_pid = os.fork()
                if new_pid == 0:  # Si new_pid == 0 > forked process.
                    try:  # uso un try..except..finally para asegurarme de _siempre_ terminar el proceso
                        print file_name
                        loaded_sound = essentia.standard.MonoLoader(downmix="left", filename=path + "/" + file_name)()
                        extracted = extracter.extract_all_descriptors(loaded_sound)
                        # descriptors = extracted.keys()  # Los nombres de los descriptores

                        averaged = MeanCalculator.calculate_mean(extracted)
                        JsonWriter.write_json(path, file_name, averaged)
                        print 'Done!'
                    except Exception:
                        exception[file_name] = ['oops']
                    finally:
                        os._exit(0)  # Cierro el fork y vuelvo al proceso padre
                else:
                    child = new_pid
                os.waitpid(child, 0)  # evito crear procesos paralelos

    return exception
