import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
import pywt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
 
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def calc_baseline(signal):
    ssds = np.zeros((3))
    cur_lp = np.copy(signal)
    iterations = 0
    while True:
        lp, hp = pywt.dwt(cur_lp, "db4")

        ssds = np.concatenate(([np.sum(hp ** 2)], ssds[:-1]))

        if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
            break

        cur_lp = lp[:]
        iterations += 1

    baseline = cur_lp[:]
    for _ in range(iterations):
        baseline = pywt.idwt(baseline, np.zeros((len(baseline))), "db4")

    return baseline[: len(signal)]

def normalizing(ecg):
    ecg = (ecg - ecg.min()) / (ecg.max() - ecg.min())
    return ecg

def preprocessing(ecg):
    ecg = signal.savgol_filter(ecg, 25, 3)
    ecg = normalizing(ecg)
    return ecg