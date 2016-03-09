windowSize = 1024          #Window Size
fftSize = 1024          #FFT size
hopSize = 512           #Hop size
sampleRate = 44100.0

'''
Sobre LogAttackTime
'''
attackTime = 1
releaseTime = 500
startAttackThreshold = 0.1

'''
descriptores!
'''

descriptors = ['mfcc', 'mfcc_bands', 'Inharmonicity', 'Energy', 'Silence', 'LogAttackTime', 'Standard-Dev', 'Variance', 'Skewness', 'kurtosis', 'mfcc-Std-f', 'mfcc-Var-f', 'mfcc-Skew-f', 'mfcc-Kurt-f', 'mfcc-1st', 'mfcc-2nd', 'Inharmonicity-1st', 'Inharmonicity-2nd']
  
