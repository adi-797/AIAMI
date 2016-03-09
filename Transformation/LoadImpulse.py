import random
import scipy.io as sio


def Site(choice):
    return {
        1 : 'booth',
        2 : 'office',
        3 : 'meeting',
        4 : 'lecture'}.get(choice, 'error')    # error is default if choice not found


def get_impulse():

  Room = Site(random.randint(1,4))
  channel = str(random.randint(0,1))
  rir_no = str(random.randint(1,3))


  file_name = ['AIR_1_4/air_binaural_' + Room + '_' + channel +
              '_1_' + rir_no + '.mat'][0] 


  mat =  sio.loadmat(file_name)
  Fs =  float(mat['air_info'][0]['fs'][0][0][0])
  sound = mat['h_air'][0]

  return sound, Fs
