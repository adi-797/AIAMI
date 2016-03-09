""" 
Este script se encarga de realizar las descripciones de los sonidos.
Los extrae para cada archivo y los escala entre si.
Luego, los guarda en un archivo Json para la siguiente etapa del proyecto
"""

import os

import essentia.standard

import extracter
import MeanUMA
import JsonWriter
import config


def get(input_dir, file_format):


  exception = {}

  for path, dname, fnames in os.walk(input_dir): # dname directories names in current directory, fnames file names in current directory.
    for fname in fnames: 
      if (file_format in fname.lower()) :
        new_pid = os.fork()
        if new_pid == 0:  # Si new_pid == 0 > forked process.
          try:            # uso un try..except..finally para asegurarme de _siempre_ terminar el proceso
            print fname
            fDict         = essentia.standard.MonoLoader(downmix = "left", filename = path+"/"+ fname)() # lo cargo
            extracted     = extracter.extract_all_descriptors(fDict)            # Le aplico los descriptores y obtengo un Pool de Essentia.
            descriptors   = extracted.keys()                             # Los nombres de los descriptores
                      
            toWrite       = MeanUMA.calculate_mean(extracted)       #Calculo el promedio
            JsonWriter.write_json(path, fname, toWrite)  #Guardo los resultados en archivos .json
            print 'Done!'
          except Exception:
            exception[fname] = ['oops'] # De esta forma puedo fijarme si hubo alguna excepcion
            #pass  
          finally:
            os._exit(0)  # Cierro el fork y vuelvo al proceso padre
        else:
          child = new_pid
        os.waitpid(child, 0) # evito crear procesos paralelos, lo que limita la performance de mi programa
        
  return exception
  
  
def get_pitch(fname, window_size=config.windowSize, frame_size = config.fftSize, hop_size=config.hopSize, sample_rate=config.sampleRate):

    audio = essentia.standard.MonoLoader(downmix="left", filename=fname)()
    spectrum = essentia.standard.Spectrum(size=frame_size)  # FFT() would return the complex FFT, we need the magnitude
    w2 = essentia.standard.Windowing(size=window_size, type='hann')
    pitch = essentia.standard.PitchYin(frameSize=frame_size, sampleRate=sample_rate)
    pitchFFT = essentia.standard.PitchYinFFT(frameSize=frame_size, sampleRate=sample_rate)

  #no se por qu'e pitch tira error. es que le entro con un frameSize dsitinto a antes
    pitchs = []
    confidences = []
    pitchsFFT = []
    confidencesFFT = []

    for frame in essentia.standard.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
        p, confidence = pitch(w2(frame))
        pitchs.append(p)
        confidences.append(confidence)
        p, confidence = pitchFFT(spectrum(w2(frame)))
        pitchsFFT.append(p)
        confidencesFFT.append(confidence)

    return pitchs,confidences, pitchsFFT, confidencesFFT
