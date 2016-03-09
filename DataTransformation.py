import os
import numpy as np
from essentia.standard import AudioLoader, Resample, MonoWriter
import LoadImpulse

def get(inputDir, descExt):

  exception = {}
  output = {}

  for path, dname, fnames  in os.walk(inputDir): # dname directories names in current directory, fnames file names in current directory.
    for fname in fnames: 
      if descExt in fname.lower(): 
        new_pid = os.fork()
        if new_pid == 0:  # Si new_pid == 0 > forked process.
          try:            # uso un try..except..finally para asegurarme de _siempre_ terminar el proceso
            file_name = path+"/"+ fname
            [Sound, Fs, nChannels, md5, bit_rate, codec] = AudioLoader(filename = file_name)() # lo cargo
            Sound = Sound[:,0] # El algoritmo siempre tira dos canales
                   
            print file_name       
                        
            impulse, FsImpulse = LoadImpulse.get_impulse()
            impulse = impulse.astype('float32', casting = 'same_kind')
            
            if Fs != FsImpulse:
              Rs = Resample(inputSampleRate = FsImpulse, outputSampleRate = Fs)
              impulse = Rs(impulse)
            
            final = np.convolve(Sound,impulse)
            
            if descExt == '.aif' : descExt = '.aiff' 
            
            mw = MonoWriter(filename = path + '/R1_' + fname, sampleRate = Fs, format = descExt.split('.')[1])
            mw(final) 
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

