# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 15:18:42 2018

@author: 180218
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from package.loadsignal import *
import numpy.fft as fft
#plt.plot(data_fft)
#plt.figure(figsize=(3.5,3.5))
def emg_fft(data,samplerate,start,end,b_plt): #start=0,end=1000
    sampleperiod=1/samplerate
    yf = scipy.fftpack.fft(data[start:end])
    yf = 2.0/len(data[start:end]) * np.abs(yf[:len(data[start:end])//2])
    xf = np.linspace(0.0, 1.0/(2.0*sampleperiod), len(data[start:end])/2)
#    data_fft=fft.fft(data[start:end])  #如何切n這個參數，與輸出頻率有關
#    ps = np.abs(data_fft[0:len(data_fft)//2])
    
    if b_plt is True:
#        plt.figure(figsize=(3.5,3.5))
        plt.plot(xf, yf)
        
#        plt.plot(ps)
        plt.title("Frequency Spectrum")
        plt.xlabel("f") 
        plt.ylabel("Amplitude")
    rms = np.sqrt(np.mean(data[start:end]**2))
    return yf , rms
#    return ps , rms   
#
#mmg_ts,mmg_acc,mmg_delta_ts=loadsignal.load_mmg('./data/time_uncali/20180502_MMG/1.5/data0000.csv')
#
#data_fft=emg_fft(mmg_acc[:,0],800,0,1000)
