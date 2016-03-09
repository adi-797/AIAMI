# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats
from pylab import imshow, show, figure

import essentia.standard

from utils import config


def extract_all_descriptors(signal):
    """
    Extracts the descriptors expected for the analysis of a given audio file.
    """

    described = {}

    described['Silence'] = _silence = silence(signal)
    signal = signal[config.hopSize * _silence[0]:config.hopSize * _silence[1]] / np.max(
        signal)  # Tomo solo la parte con sonido y promedio para que todas las señales sean parejas.
    described['mfcc'] = mfccs(signal)
    described['Inharmonicity'] = inharmonicity_tesis(signal)
    described['Energy'] = energy(signal)
    described['LogAttackTime'] = log_attack_time(signal)
    described['Standard-Dev'] = standard_dev(signal)
    described['Variance'] = variance(signal)
    described['Skewness'] = skewness(signal)
    described['kurtosis'] = kurtosis(signal)

    # described['mfcc-1st'] = np.gradient(described['mfcc'])[1]
    # described['mfcc-2nd'] = np.gradient(described['mfcc-1st'])[1]
    described['Inharmonicity-1st'] = np.gradient(described['Inharmonicity'])
    described['Inharmonicity-2nd'] = np.gradient(described['Inharmonicity-1st'])

    described['mfcc-Std-f'], described['mfcc-Var-f'], described['mfcc-Skew-f'], described['mfcc-Kurt-f']\
        = mfcc_std_frequency(described)

    return described


def mfccs(audio, window_size=config.windowSize, fft_size=config.fftSize, hop_size=config.hopSize, plot=False):
    """
    Calculates the mfcc for a given audio file.
    """

    window_hann = essentia.standard.Windowing(size=window_size, type='hann')
    spectrum = essentia.standard.Spectrum(
        size=fft_size)  # FFT() would return the complex FFT, here we just want the magnitude spectrum
    mfcc = essentia.standard.MFCC(numberCoefficients=12, inputSize=fft_size / 2 + 1)

    pool = essentia.Pool()

    for frame in essentia.standard.FrameGenerator(audio, frameSize=fft_size, hopSize=hop_size):
        mfcc_bands, mfcc_coefficients = mfcc(spectrum(window_hann(frame)))
        pool.add('lowlevel.mfcc', mfcc_coefficients)
        pool.add('lowlevel.mfcc_bands', mfcc_bands)

    if plot:
        imshow(pool['lowlevel.mfcc'].T[1:, :], aspect='auto')
        show()  # unnecessary if you started "ipython --pylab"
        figure()
        imshow(pool['lowlevel.mfcc_bands'].T, aspect='auto', interpolation='nearest')
        # We ignored the first MFCC coefficient to disregard the power of the signal and only plot its spectral shape

    return pool['lowlevel.mfcc'].T


def inharmonicity_tesis(audio, window_size=config.windowSize, spectrum_size=config.fftSize,
                        hop_size=config.hopSize, sample_rate=config.sampleRate):
    """ Setting up everything """
    window_bh = essentia.standard.Windowing(size=window_size, type='blackmanharris92')
    spectrum = essentia.standard.Spectrum(size=spectrum_size)  # magnitude spectrum
    peaks = essentia.standard.SpectralPeaks(magnitudeThreshold=-120, sampleRate=sample_rate)

    window_hann = essentia.standard.Windowing(size=window_size, type='hann')
    pitch = essentia.standard.PitchYin(frameSize=window_size, sampleRate=sample_rate)
    pitch_fft = essentia.standard.PitchYinFFT(frameSize=window_size, sampleRate=sample_rate)

    harmonicpeaks = essentia.standard.HarmonicPeaks()
    inharmonicity = essentia.standard.Inharmonicity()

    vector_inharmonicity = np.array([])

    """ Actual signal processing """
    for frame in essentia.standard.FrameGenerator(audio, frameSize=spectrum_size, hopSize=hop_size):

        frequency, amplitude = peaks(20 * np.log10(spectrum(window_bh(frame))))
        if 0 in frequency:
            frequency = np.array([x for x in frequency if x != 0])  # elimino la informacion sobre la energia en 0 Hz
            amplitude = amplitude[1:len(
                amplitude)]  # Asumo que esta toda en la primer posicion, si no es asi va a saltar un error
        if len(frequency) == 0:
            continue

        value_pitch, confidence = pitch(window_hann(frame))
        value_pitch_fft, confidence_fft = pitch_fft(spectrum(window_hann(frame)))
        if (confidence and confidence_fft) < 0.2:
            continue
        else:
            if confidence > confidence_fft:
                value_pitch = value_pitch
            else:
                value_pitch = value_pitch_fft

        harmonic_frequencies, harmonic_magnitudes = harmonicpeaks(frequency, amplitude, value_pitch)
        vector_inharmonicity = np.append(vector_inharmonicity,
                                         inharmonicity(harmonic_frequencies, harmonic_magnitudes))

    return vector_inharmonicity


def energy(audio):
    return sum(audio * audio)


def log_attack_time(audio):
    enveloped = essentia.standard.Envelope(attackTime=config.attackTime, releaseTime=config.releaseTime)(audio)
    return essentia.standard.LogAttackTime(startAttackThreshold=config.startAttackThreshold)(enveloped)


def standard_dev(audio):
    return np.std(audio)


def variance(audio):
    return np.var(audio)


def skewness(audio):
    return scipy.stats.skew(audio)


def kurtosis(audio):
    return scipy.stats.kurtosis(audio)


def silence(audio, fft_size=config.fftSize, hop_size=config.hopSize):
    """
    Detects the begining and the end of the audio file.
    The output is a vector where the first 1 indicates the begining of the audio file and the last 1 the ending.
    The threshold is set at 90dB under the maximum of the file.
    The first 1 is set one frame before the real start of the sound.
    """

    threshold = 90.0
    real_threshold = 10.0 ** ((20.0 * np.log10(max(audio)) - threshold) / 20.0)

    l = []
    for frame in essentia.standard.FrameGenerator(audio, frameSize=fft_size, hopSize=hop_size):
        if sum(frame * frame) >= real_threshold:
            l.append(1)
        else:
            l.append(0)

    start = l.index(1)
    if start != 0:
        start -= 1

    end = len(l) - l[::-1].index(1)
    if end != len(l):
        end += 1

    return [start, end]


def mfcc_std_frequency(described):
    std = []
    var = []
    kurt = []
    skew = []
    inter = []

    for iii in range(len(described['mfcc'][0])):#temporal
        for jjj in range(len(described['mfcc'])):#frecuencial
            inter.append(described['mfcc'][jjj][iii])
        std.append(np.std(inter))  # Desviacion estandar de cada frame entre todas las frecuencias.
        var.append(np.var(inter))
        skew.append(scipy.stats.skew(inter))
        kurt.append(scipy.stats.kurtosis(inter))

    return std, var, skew, kurt


def extractor(signal, sample_rate=config.sampleRate):
    """
    Extracts pretty much every descriptors in Essentia.
    """

    algor = essentia.standard.Extractor(sampleRate=sample_rate, dynamicsFrameSize=4096, dynamicsHopSize=2048,
                                        lowLevelFrameSize=2048, lowLevelHopSize=1024, namespace="Tesis", rhythm=False)
    output = algor(signal)

    return output
