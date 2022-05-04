# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:17:32 2018

@author: 180218
"""
import scipy as sp
from scipy import signal
import numpy as np
def simu_filter(gyr,acc,mag,samplerate):
    bacc, aacc = sp.signal.butter(4, 5/samplerate, btype = 'lowpass')
    bmag, amag = sp.signal.butter(4, 5/samplerate, btype = 'lowpass')
    bgyr, agyr = sp.signal.butter(4, 5/samplerate, btype = 'lowpass')
    acc[:,0] = sp.signal.filtfilt(bacc, aacc, acc[:,0])
    acc[:,1] = sp.signal.filtfilt(bacc, aacc, acc[:,1])
    acc[:,2] = sp.signal.filtfilt(bacc, aacc, acc[:,2])
    mag[:,0] = sp.signal.filtfilt(bmag, amag, mag[:,0])
    mag[:,1] = sp.signal.filtfilt(bmag, amag, mag[:,1])
    mag[:,2] = sp.signal.filtfilt(bmag, amag, mag[:,2])
    gyr[:,0] = sp.signal.filtfilt(bgyr, agyr, gyr[:,0])
    gyr[:,1] = sp.signal.filtfilt(bgyr, agyr, gyr[:,1])
    gyr[:,2] = sp.signal.filtfilt(bgyr, agyr, gyr[:,2])
#    return gyr,acc,mag
def simu_filter2(gyr,acc,mag,samplerate):
    bacc, aacc = sp.signal.butter(4, 5/samplerate, btype = 'lowpass')
    bmag, amag = sp.signal.butter(4, 5/samplerate, btype = 'lowpass')
    bgyr, agyr = sp.signal.butter(4, 5/samplerate, btype = 'lowpass')
    
    acc[:,0] = sp.signal.filtfilt(bacc, aacc, acc[:,0])
    acc[:,1] = sp.signal.filtfilt(bacc, aacc, acc[:,1])
    acc[:,2] = sp.signal.filtfilt(bacc, aacc, acc[:,2])
    mag[:,0] = sp.signal.filtfilt(bmag, amag, mag[:,0])
    mag[:,1] = sp.signal.filtfilt(bmag, amag, mag[:,1])
    mag[:,2] = sp.signal.filtfilt(bmag, amag, mag[:,2])
    gyr[:,0] = sp.signal.filtfilt(bgyr, agyr, gyr[:,0])
    gyr[:,1] = sp.signal.filtfilt(bgyr, agyr, gyr[:,1])
    gyr[:,2] = sp.signal.filtfilt(bgyr, agyr, gyr[:,2])
#    return gyr,acc,mag
def vec_filter(vec,samplerate,l3db,h3db):
    low = l3db/samplerate
    if h3db is None:
        b, a = sp.signal.butter(4, low, btype = 'lowpass')
    else:
        high = h3db/samplerate
        b, a = sp.signal.butter(4, [low,high], btype = 'bandpass')
        
    for i in range(vec.shape[1]):
        vec[:,i] = sp.signal.filtfilt(b, a, vec[:,i])
    return vec
def all_filter(q,vec,samplerate,l3db,h3db=None):
    low = l3db/samplerate
    
    if h3db is None:
        b, a = sp.signal.butter(4, low, btype = 'lowpass')
    else:
        high = h3db/samplerate
        b, a = sp.signal.butter(4, [low,high], btype = 'bandpass')
    vec[:,0] = sp.signal.filtfilt(b, a, vec[:,0])
    vec[:,1] = sp.signal.filtfilt(b, a, vec[:,1])
    vec[:,2] = sp.signal.filtfilt(b, a, vec[:,2])
    q[:,0] = sp.signal.filtfilt(b, a, q[:,0])
    q[:,1] = sp.signal.filtfilt(b, a, q[:,1])
    q[:,2] = sp.signal.filtfilt(b, a, q[:,2])
    q[:,3] = sp.signal.filtfilt(b, a, q[:,3])
    return q,vec