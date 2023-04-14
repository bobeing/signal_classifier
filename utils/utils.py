import cv2
import matplotlib.pyplot as plt
from scipy import signal
import librosa
from tqdm import tqdm
import numpy as np
import torch

import os
import sys
import importlib
import h5py
import yaml

def read_h5_file(filename):
    with h5py.File(filename, "r") as f:
        attrs = dict(f.attrs)
        signal = np.array(f['signal'])
        return attrs, signal

def open_yaml(file_path):
    with open(file_path) as f:
        res = yaml.load(f, Loader=yaml.FullLoader)
    
    return res

def initialize_module(path: str, args: dict = None, initialize: bool = True):

    module_path = ".".join(path.split(".")[:-1])
    class_or_function_name = path.split(".")[-1]
    module = importlib.import_module(module_path)
    class_or_function = getattr(module, class_or_function_name)

    if initialize:
        if args:
            return class_or_function(**args)
        else:
            return class_or_function()
    else:
        return class_or_function


def get_IQ_vector(signal, mapsize=2048):

    real = np.round(signal.real * mapsize/2 + mapsize/2)
    imag = np.round(signal.imag * mapsize/2 + mapsize/2)

    result = np.zeros((mapsize, mapsize))

    for j,i in zip(imag, real):
        j = int(j)
        i = int(i)
        cv2.circle(result, center=(j,i), radius=4, color = 1, thickness=-1)
        plt.imshow(result)
        plt.draw()
        plt.pause(0.1)
        # result[j,i] = 1 

    return result

def get_af_vector(signal, mapsize=2048):

    real = np.round(signal.real * mapsize/2 + mapsize/2)
    imag = np.round(signal.imag * mapsize/2 + mapsize/2)

    result = np.zeros((mapsize, mapsize))

    for j,i in zip(signal.imag, signal.real):
        a = np.sqrt(j**2+i**2)
        f = np.arctan(j/i)
        # cv2.circle(result, center=(j,i), radius=4, color = 1, thickness=-1)
        plt.scatter(a, f)
        plt.draw()
        plt.pause(0.1)
        # result[j,i] = 1 

    return result

def draw_stft(f, t, Zxx):
    plt.figure(figsize=(12,5))
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', vmin=0, vmax=0.5*np.abs(Zxx).max())
    plt.title('STFT Magnitude'); plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]'); plt.ylim([0, 20]); 
    plt.show()

def draw_stft_I_Q(f, t, Zxx):
    plt.figure(1)
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', vmin=0, vmax=0.5*np.abs(Zxx).max())
    plt.title('STFT Magnitude'); plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]'); plt.ylim([0, 20]); 
    
    plt.figure(2)
    plt.plot(range(len(np.real(Zxx))), np.real(Zxx), 'b-', range(len(np.imag(Zxx))), np.imag(Zxx), 'r-')
    
    plt.show()


def calc_stft(symbols, Fs, nperseg):
    f, t, Zxx = signal.stft(symbols, Fs, nperseg=nperseg)
    # draw_stft(f, t, Zxx)
    # draw_stft_I_Q(f, t, Zxx)
    return np.abs(Zxx)



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



# def calc_stft(wave, Fs, n_fft=1024):
#     x = librosa.load(wave, Fs)[0]
#     y = librosa.stft(x, n_fft, hop_length=n_fft/2, win_length=n_fft)
    
#     print(y)
#     # plt.figure(figsize=(12,4))
#     # librosa.display.specshow()



def autocorr(x):
    result = np.correlate(x, x, mode='same')
    return result


def raised_root_cosine(upsample, num_positive_lobes, alpha):

    """
    Root raised cosine (RRC) filter (FIR) impulse response.

    upsample: number of samples per symbol

    num_positive_lobes: number of positive overlaping symbols
    length of filter is 2 * num_positive_lobes + 1 samples

    alpha: roll-off factor
    """

    N = upsample * (num_positive_lobes * 2 + 1)
    t = (np.arange(N) - N / 2) / upsample

    # result vector
    h_rrc = np.zeros(t.size, dtype=np.float)

    # index for special cases
    sample_i = np.zeros(t.size, dtype=np.bool)

    # deal with special cases
    subi = t == 0
    sample_i = np.bitwise_or(sample_i, subi)
    h_rrc[subi] = 1.0 - alpha + (4 * alpha / np.pi)

    subi = np.abs(t) == 1 / (4 * alpha)
    sample_i = np.bitwise_or(sample_i, subi)
    h_rrc[subi] = (alpha / np.sqrt(2)) \
                * (((1 + 2 / np.pi) * (np.sin(np.pi / (4 * alpha))))
                + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))

    # base case
    sample_i = np.bitwise_not(sample_i)
    ti = t[sample_i]
    h_rrc[sample_i] = np.sin(np.pi * ti * (1 - alpha)) \
                    + 4 * alpha * ti * np.cos(np.pi * ti * (1 + alpha))
    h_rrc[sample_i] /= (np.pi * ti * (1 - (4 * alpha * ti) ** 2))

    return h_rrc

def read_h5_file(filename):
    with h5py.File(filename, "r") as f:
        attrs = dict(f.attrs)
        signal = np.array(f['signal'])[:int(9600)]
        # signal = np.array(f['signal'])
        # print(len(signal))
        return attrs, signal