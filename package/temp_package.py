# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 14:02:17 2018

@author: 180218
"""
#a=Quate
from os import listdir
from dtw import dtw
# from dtw import dtw4

#from .sql import SQLite,Table
from .sql import *
from os.path import isfile, join,isdir
import warnings
import time 
#from .peakdetect import *
from .peakdetect import *
from .file import *
#from pyquaternion import Quaternion
startTime = time.time()
from .emg_fft import emg_fft
from .ellipse_fitting import ellipse_fitting
from .ellipse_fitting import ellipse_fitting2
from .ellipse_fitting import ellipse_fitting_aiq_without_t
from matplotlib.animation import FuncAnimation
class Animation(object):
    def __init__(self,xs,ys):
        self.x=xs
        self.y=ys
        
        self.fig=plt.figure()
        self.axis=plt.axes(xlim=(np.min(self.x),np.max(self.x)),ylim=(np.min(self.y),np.max(self.y)))
        self.line, =self.axis.plot([],[],lw=2)
        self.xdata, self.ydata=[],[]
        
    def animateInit(self):
        self.line.set_data([],[])
        
        return self.line
    def animate(self,i):
        self.xdata.append(self.x[i])
        self.ydata.append(self.y[i])
        self.line.set_data(self.xdata,self.ydata)
        return self.line
    def animateGenerate(self):
        self.ani=FuncAnimation(self.fig ,self.animate, init_func=self.animateInit, frames=self.x.shape[0],interval=0.1,blit=True)
#        self.ani.save('anim.mp4',writer='ffmpeg',fps=30)
            
class CadenceBuffer(object):
    def __init__(self,t_index=3,angle_index=9,samplerate=None):
        self.cadence=[]
        self.delt_angle=[]
        self.delt_ts=[]
        self.last_delt_angle=999
        self.t_index=3
        self.angle_index=9
        self.samplerate=samplerate
        self.max=[]
        self.min=[]
        self.increase=0
        self.count=0
#        self.angle=temp[:,9]
#        ts=temp[:,3]
#        cum_ts=sum(ts)
#        max_angle=max(angle)
#        min_angle=min(angle)
#        max_index=np.argmax(angle)
#        min_index=np.argmin(angle)
#        delt_angle=abs(max_angle-min_angle)
        
    def update(self,temp):
        angle=temp[:,self.angle_index]
        ts=temp[:,self.t_index]
        cum_ts=sum(ts)
        max_angle=max(angle)
        min_angle=min(angle)
        max_index=np.argmax(angle)
        min_index=np.argmin(angle)
        delt_angle=abs(max_angle-min_angle)
        if delt_angle>200:
            delt_angle=360-min(angle[angle>0])+max(angle[angle<0])
            max_index=np.argmax(angle[angle>0])
            min_index=np.argmin(angle[angle<0])
#            if delt_angle>100:
#                print(max_angle,min_angle,delt_angle)
#            if delt_angle>100:
#                print(min(angle[angle>0]),max(angle[angle<0]),delt_angle)

        cum_ts=0
        for i in range(min_index,max_index+1):
            cum_ts+=ts[i]
        if delt_angle<20 or delt_angle>120 or cum_ts<0.15 or delt_angle>self.last_delt_angle*2 or delt_angle<self.last_delt_angle*0.7:
            if len(self.cadence)!=0:
                delt_angle=self.last_delt_angle
                cad=self.cadence[-1]
            else:
                cad=0
        else:
            cad=delt_angle/cum_ts/360*60
        self.cadence.append(cad)
        self.delt_ts.append(cum_ts)
        self.delt_angle.append(delt_angle)
        self.last_delt_angle=delt_angle
#        if delt_angle>100:
#            print(max_angle,min_angle,delt_angle)
    def update2(self,temp):
        angle=temp[:,self.angle_index]
#        ts=np.diff(temp[:,1])/1000
#        ts=np.append(np.zeros(1),ts)
#        ts=np.zeros(1).append(ts)
        ts=temp[:,self.t_index]
        cum_ts=sum(ts)
        max_angle=max(angle)
        min_angle=min(angle)
        max_index=np.argwhere(angle==max_angle)[0][0]
        min_index=np.argwhere(angle==min_angle)[0][0]
        delt_angle=abs(max_angle-min_angle)
        if delt_angle>200:
            delt_angle=360-min(angle[angle>0])+max(angle[angle<0])
            a=angle[angle>0]
            b=angle[angle<0]
            if len(a)==1:
                max_index=0
            else:
                max_index=np.argwhere(angle==min(angle[angle>0]))[0][0]
            if len(b)==1:
                min_index=0
            else:            
                min_index=np.argwhere(angle==max(angle[angle<0]))[0][0]
#            if delt_angle>100:
#                print(max_angle,min_angle,delt_angle)
#            if delt_angle>100:
#                print(min(angle[angle>0]),max(angle[angle<0]),delt_angle)
#            print(delt_angle)
        cum_ts=0
        if max_index>min_index:
            start=min_index
            end=max_index+1
        else:
            start=max_index
            end=min_index+1        
#        if end==1:
#            return
#        print(start,end)
#        print(max_index,min_index)
#        print(angle)
#        print(min(angle[angle>0]),max(angle[angle<0]))
        for i in range(start,end):
            cum_ts+=ts[i]
#        if(cum_ts==0.01):
#            print(start,end)
#        print(cum_ts)
        if delt_angle<20 or delt_angle>120 or cum_ts<0.15 or delt_angle>self.last_delt_angle*2 or delt_angle<self.last_delt_angle*0.7:
#            if len(self.cadence)!=0:
#                delt_angle=self.last_delt_angle
#                cad=self.cadence[-1]
#            else:
#                cad=0
#            cad=self.cadence[-1]
            cad=delt_angle/cum_ts/360*60
            pass
        else:
            cad=delt_angle/cum_ts/360*60

        self.cadence.append(cad)
        self.delt_ts.append(cum_ts)
        self.delt_angle.append(delt_angle)
        self.last_delt_angle=delt_angle
#        if cad>300:
#            print(max_angle,min_angle,delt_angle,cum_ts)
    def update4(self,temp):#apple notebook
        angle=temp[:,self.angle_index]
        ts=temp[:,self.t_index]
        cum_ts=sum(ts)
        max_angle=np.max(angle)
        min_angle=np.min(angle)
#        print(max_angle,min_angle)
        max_index=np.argwhere(angle==max_angle)[0]
        min_index=np.argwhere(angle==min_angle)[0]
        delt_angle=max_angle-min_angle
        diff_index=abs(max_index-min_index)
        if delt_angle>300:
            if angle[0]<0:
#                delt_angle=abs(angle[0])+angle[-1]
                delt_angle=360+angle[0]-angle[-1]
            else:
#                print('really??')
#                print(angle)
                delt_angle=360-angle[0]+angle[-1]
            diff_index=len(temp)
#                print(delt_angle)
#        if(diff_index<5):
#            return
#        cum_ts=0
#        for i in range(min_index,max_index):
#            cum_ts+=ts[i]
        cum_ts=diff_index/self.samplerate
        
        cad=delt_angle/360/(diff_index/self.samplerate/60)
#         
        self.cadence.append(cad)
        self.delt_ts.append(cum_ts)
        self.delt_angle.append(delt_angle)
        self.last_delt_angle=delt_angle
        self.max.append(angle[0])
        self.min.append(angle[-1])
        if delt_angle>300:
            print(angle[0],angle[-1],delt_angle,cum_ts)
            print(angle)
    def update5(self,temp):#apple notebook alter有一個low pulse
        angle=temp[:,self.angle_index]
#        cum_ts=sum(ts)
        max_angle=np.max(angle)
        min_angle=np.min(angle)
        max_index=np.argwhere(angle==max_angle)[0][0]
        min_index=np.argwhere(angle==min_angle)[0][0]
        delt_angle=max_angle-min_angle
        diff_index=abs(max_index-min_index)
        if delt_angle>50:
            if angle[0]<0:
                delt_angle=360+angle[0]-angle[-1]
                return
            else:
                delt_angle=360-angle[0]+angle[-1]
                return
        cum_ts=0
        if min_index<max_index:
            start=min_index
            end=max_index
        else:
            return
            start=max_index
            end=min_index            

            
        cum_ts=diff_index/self.samplerate
        cad=delt_angle/360/(cum_ts/60)
        self.cadence.append(cad)
        self.delt_ts.append(cum_ts)
        self.delt_angle.append(delt_angle)
        self.last_delt_angle=delt_angle
        self.max.append(angle[0])
        self.min.append(angle[-1])
        if delt_angle>300:
            print(angle[0],angle[-1],delt_angle,cum_ts)
            print(angle)
    def update6(self,temp):#跟目前app依樣判斷是否為遞升
        angle=temp[:,self.angle_index]
#        cum_ts=sum(ts)
        max_angle=np.max(angle)
        min_angle=np.min(angle)
        max_index=np.argwhere(angle==max_angle)[0][0]
        min_index=np.argwhere(angle==min_angle)[0][0]
        increase=np.diff(angle)
        if(len(np.where(increase>0)[0])==(len(temp)-1)):
            self.increase=True
            delt_angle=angle[-1]-angle[0]
            
        else:
#            pass
            self.increase=False
            if((max_angle-min_angle)>5):
                delt_angle=360+angle[-1]-angle[0]
                if(delt_angle>300):
                    delt_angle=angle[-1]-angle[0]
            else:
                delt_angle=angle[-1]-angle[0]
                
        
        cum_ts=(len(temp)-1)*(1/self.samplerate)

            
        cad=delt_angle/360/(cum_ts/60)
        self.cadence.append(cad)
        self.delt_ts.append(cum_ts)
        self.delt_angle.append(delt_angle)
        self.last_delt_angle=delt_angle
        self.max.append(angle[0])
        self.min.append(angle[-1])
#        plt.show()
#        plt.title(str(self.delt_angle[-1])+'/'+str(self.max[-1])+'/'+str(self.min[-1]))
#        plt.plot(angle)
#        print('saving'+'./window_pic/'+str(self.count)+'.jpg')
#        plt.savefig('./window_pic/'+str(self.count)+'.jpg')
#        plt.show()
        self.count+=1
#        if delt_angle>300:
#            print(angle[0],angle[-1],delt_angle,cum_ts)
#            print(angle)
    def update5_bk(self,temp):#apple notebook alter有一個low pulse
        angle=temp[:,self.angle_index]
        ts=temp[:,self.t_index]
#        cum_ts=sum(ts)
        max_angle=np.max(angle)
        min_angle=np.min(angle)
#        print(max_angle,min_angle)
        max_index=np.argwhere(angle==max_angle)[0][0]
        min_index=np.argwhere(angle==min_angle)[0][0]
        delt_angle=max_angle-min_angle
        diff_index=abs(max_index-min_index)
        if delt_angle>50:
            if angle[0]<0:
#                delt_angle=abs(angle[0])+angle[-1]
                delt_angle=360+angle[0]-angle[-1]
                return
            else:
#                print('really??')
#                print(angle)
                delt_angle=360-angle[0]+angle[-1]
                return
#            diff_index=len(temp)
#                print(delt_angle)
#        if(diff_index<5):
#            return
        cum_ts=0
#        print(min_index,max_index)
        if min_index<max_index:
            start=min_index
            end=max_index
        else:
            return
            start=max_index
            end=min_index            
        for i in range(start,end+1):
            cum_ts+=ts[i]
            
        cum_ts=diff_index/self.samplerate
        
        cad=delt_angle/360/(cum_ts/60)
#         
        self.cadence.append(cad)
        self.delt_ts.append(cum_ts)
        self.delt_angle.append(delt_angle)
        self.last_delt_angle=delt_angle
        self.max.append(angle[0])
        self.min.append(angle[-1])
        if delt_angle>300:
            print(angle[0],angle[-1],delt_angle,cum_ts)
            print(angle)
    def update3(self,temp):
        angle=temp[:,self.angle_index]
        ts=temp[:,self.t_index]
#        ts=np.diff(temp[:,1])/1000
#        ts=np.append(np.zeros(1),ts)
        cum_ts=sum(ts)
        max_angle=angle[-1]
        min_angle=angle[0]
        max_index=len(temp)
        min_index=0
        delt_angle=abs(max_angle-min_angle)
        if delt_angle>200:
            if angle[0]<0:
#                delt_angle=abs(angle[0])+angle[-1]
                delt_angle=360+angle[0]-angle[-1]
            else:
#                print('really??')
#                print(angle)
                delt_angle=360-angle[0]+angle[-1]
#                print(delt_angle)
#            a=angle[angle>0]
#            b=angle[angle<0]
        cum_ts=0
        start=0
        end=len(temp)

#        if end==1:
#            return
#        print(start,end)
#        print(max_index,min_index)
#        print(angle)
#        print(min(angle[angle>0]),max(angle[angle<0]))
        for i in range(start,end):
            cum_ts+=ts[i]
#        if(cum_ts==0.01):
#            print(start,end)
#        print(cum_ts)
        if delt_angle<20 or delt_angle>120 or cum_ts<0.15 or delt_angle>self.last_delt_angle*2 or delt_angle<self.last_delt_angle*0.7:
#            if len(self.cadence)!=0:
#                delt_angle=self.last_delt_angle
#                cad=self.cadence[-1]
#            else:
#                cad=0
#            cad=self.cadence[-1]
            cad=delt_angle/cum_ts/360*60
#            pass
        else:
            cad=delt_angle/cum_ts/360*60
#        print(cum_ts)
#        if(cum_ts==0.05):
#            self.cadence.append(self.cadence[-1])
#            self.delt_ts.append(self.delt_ts[-1])
#            self.delt_angle.append(self.delt_angle[-1])
#            self.last_delt_angle=self.last_delt_angle
#            print('cum_ts too small')
#            return
            
        self.cadence.append(cad)
        self.delt_ts.append(cum_ts)
        self.delt_angle.append(delt_angle)
        self.last_delt_angle=delt_angle
        if delt_angle>300:
            print(angle[0],angle[-1],delt_angle,cum_ts)
#            print(angle)
            
    def mov_avg_cad(self,windowsize,shift=1):
        self.ma_cad=[]
        
        for i in range(0,len(self.cadence)-windowsize,shift):
            window=self.cadence[i:i+windowsize]
            avg=np.average(window)
            self.ma_cad.append(avg)
        return np.array(self.ma_cad)
    def mov_avg_cad2(self,windowsize,shift=1):
        self.ma_cad2=[]
        
        for i in range(windowsize,len(self.delt_angle)-windowsize,shift):
            window=self.delt_angle[i-windowsize:i]
            avg=np.average(window)
            self.ma_cad2.append(avg/self.delt_ts[i]/360*60)
        return np.array(self.ma_cad2)    
    
def calc_angle_acc(acc):
    pitch=np.zeros(len(acc))
    pitch=np.arctan(acc[:,0]/np.sqrt(acc[:,1]**2+acc[:,2]**2))
    roll=np.zeros(len(acc))
    roll=np.arctan(acc[:,1]/np.sqrt(acc[:,0]**2+acc[:,2]**2))
    pitch=pitch/np.pi*180
    roll=roll/np.pi*180
    return np.array([roll,pitch]).transpose()
from scipy.signal import find_peaks
import numpy as np
import inspect
import scipy as sp
from scipy import signal
from package.quaternion import Quaternion
import copy
import pandas as pd
from package.calc_orientation import calc_orientation_mag
from package.calc_orientation import calc_orientation_c
from package.calc_orientation import calc_orientation
import matplotlib.pyplot as plt
from package.cali_mag import cali_mag
from package.simu_filter import all_filter
from package.simu_filter import vec_filter
import matplotlib.font_manager as mfm
from package.calc_angle_by_video import getangle as get_video_angle
from calibraxis import Calibraxis
from package.file import *
#def vec_filter(vec,samplerate,cut_off_f):
#    b, a = sp.signal.butter(4, cut_off_f/samplerate, btype = 'lowpass')
#    vec = sp.signal.filtfilt(b, a, vec,axis=0)
#    return vec
class Vector(object):
    def __init__(self,lenth,values=0):
        self.xyz=np.full([lenth,3],values)
        self.samplerate=50
    @property
    def norm(self):
        temp=np.linalg.norm(self.xyz,axis=1)
        temp=temp.reshape(len(temp),1)
        return temp
    def get_values(self):
        return self.xyz
    def print_value(self):
        print(self.xyz)
    def __repr__(self):
        return "print complete"
    def __str__(self):
#        return np.array(self.xyz,dtype=str)
        return "print complete"
    def lpf(self,cutOff):
        b, a = sp.signal.butter(4, cutOff/self.samplerate, btype = 'lowpass')

        self.xyz = sp.signal.filtfilt(b, a, self.xyz,axis=0)
    def hpf(self,cutOff):
        b, a = sp.signal.butter(4, cutOff/self.samplerate, btype = 'highpass')
        self.xyz = sp.signal.filtfilt(b, a, self.xyz,axis=0)

def average_adj_time(start,end,bias,ts):
    a=ts>start
    b=ts<end
    c=np.all([a,b],axis=0)
    c=c.flatten()
    start=ts[a][0]
    end=ts[b][-1]
    ts[c,0]=np.linspace(start,end,len(ts[c,:]))
    ts[c,0]+=bias
def retrieve_name(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]   



class Signal(object):
    def __init__(self,values=None,tag='',samplerate=None):

        self._values=values
        self._tag=tag
        self.shape=values.shape
        self.length=values.shape[0]
        self.samplerate=samplerate
        if samplerate is not None:
            self.ts=np.linspace(0,self.length/self.samplerate,self.length)
    def dtw(self,other,showplot=False):
#        import matplotlib.pyplot as plt

        euclidean_norm = lambda x, y: np.abs(x-y)

        d, cost_matrix, acc_cost_matrix, path = dtw(self.values, other.values, dist=euclidean_norm)

        if showplot:
            plt.show()
            plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
            plt.plot(path[0], path[1], 'w')
            plt.show()
        self.d=d
        self.cost_matrix=cost_matrix
        self.acc_cost_matrix=acc_cost_matrix
        self.path=path
        return d, cost_matrix, acc_cost_matrix, path 
    
    def dtw4(self,other,showplot=False):
#        import matplotlib.pyplot as plt
#        def innerdot_sqrt(q1,q2):
#            result=q1[0]*q2[0]+q1[1]*q2[1]+q1[2]*q2[2]+q1[3]*q2[3]
#            return result*result
#        euclidean_norm = lambda x, y: (np.arccos(2*innerdot_sqrt(x,y)-1))
        def innerdot_sqrt(q1,q2):
#            result=q1[0]*q2[0]+q1[1]*q2[1]+q1[2]*q2[2]+q1[3]*q2[3]
            qq1=Quaternion(q1)
            qq2=Quaternion(q2)
            q_delt=qq1*(qq2.conj())
#            print(q_delt.getTheta())
            return q_delt.getTheta()
        euclidean_norm = lambda x, y: innerdot_sqrt(x,y)
#        def innerdot_sqrt(q1,q2):
#
#            return q1[1]-q2[1]
#        
        d, cost_matrix, acc_cost_matrix, path = dtw4(self.values, other.values, dist=euclidean_norm)

        if showplot:
            plt.show()
            plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
            plt.plot(path[0], path[1], 'w')
            plt.show()
        self.d=d
        self.cost_matrix=cost_matrix
        self.acc_cost_matrix=acc_cost_matrix
        self.path=path
        return d, cost_matrix, acc_cost_matrix, path 
    def showdtw(self):
        plt.imshow(self.acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
        plt.plot(self.path[0], self.path[1], 'w')
        plt.show()
    def diff(self):
        self.values=np.diff(self._values,axis=0)
        return Signal(self.values,samplerate=self.samplerate)
    def copy(self):
        return copy.deepcopy(self)
    def __neg__(self):
        return self.__class__(-self.values)
    @property
    def tag(self):
        return self._tag
    @property
    def values(self):
        return self._values
    @values.setter
    def values(self,value):
        self._values=value
    def norm(self):
        temp=np.linalg.norm(self.values,axis=1)
        temp=temp.reshape(len(temp),1)
        return Signal(temp,samplerate=self.samplerate)
#        return Signal(np.linalg.norm(self.values,axis=1))
    def plot_n(self,title=None):
        if title is not None:
            plt.title(title)
        for i in range(self.values.shape[1]):
            plt.plot(self.values[:,i],label=str(i))
        plt.legend()
        plt.xlabel('n')
        plt.show()
    def plot_ts(self,title=None,label=['1','2','3'],tag=None,index=[1,2,3]):
        if self.samplerate is not None:
            if title is not None:
                plt.title(title)
            for i in index:
                plt.plot(self.ts,self.values[:,i],label=label[i],alpha=0.5)
            
            if tag is not None:
#                index=np.argwhere(self.ts==tag)
                shoot=np.searchsorted(self.ts,tag)
                
                print('1 to 1 shoot ',self.ts[shoot])
                #------scale fitting-----
#                ju=np.max(self.values[shoot,index])/10
                
                plt.scatter(self.ts[shoot],self.values[shoot,index], marker="x",linewidths=10,color='r',s=100,alpha=1)
                shoot_result=np.round(self.values[shoot,index],2)
                plt.title(shoot_result)
            plt.legend()
            plt.xlabel('ts')
            plt.show()      
        else:
            warnings.warn('this signal has no ts to plot')
            self.plot_n()
    def denoise(self,threshold):
        index=np.argwhere(self.norm().values>threshold)
        self.values=np.delete(self.values,index,axis=0)
    def trim(self,ts):
        if self.samplerate is not None:
            self.values=self.values[:int(self.samplerate*ts)]
            print('xxx',self.values.shape)
            try:
                self.values=self.values.reshape(self.values.shape[0],1)
            except:
                print('no need to parse')
            self.ts=self.ts[:int(self.samplerate*ts)]
        else:
            warnings.warn('There is no samplerate to trim')
        return self
            
    def denoise_by_std(self):
        mean=self.norm().values.mean(axis=0)
        std=self.norm().values.std(axis=0)
        threshold=mean+2*std
        index=np.argwhere(self.norm().values>threshold)
        self.values=np.delete(self.values,index,axis=0)
    def denoise_by_1std(self):
        mean=self.norm().values.mean(axis=0)
        std=self.norm().values.std(axis=0)
        threshold=mean+1*std
        index=np.argwhere(self.norm().values>threshold)
        self.values=np.delete(self.values,index,axis=0)
    def changeorder(self,order):
        sign=[i for i in range(len(order)) if order[i] < 0]
        order=[abs(x)-1 for x in order]
        self.values=self.values[:,order]
        self.values[:,sign]=-self.values[:,sign]
        return Signal(self.values,samplerate=self.samplerate)
    def show_by_index(self,index=[0]):
        sig_plt=Plot(title='signal_axis',dtype='axis',index=index)
        sig_plt.plot_angle(-self)
    def fft(self,samplerate,l3db,h3db=None):
        result=vec_filter(self.values,samplerate,l3db,h3db)
        return Signal(result,samplerate=self.samplerate)
    
    def feature_analysis(self,window_s,shift_s,samplerate,index,title='',starttime=0,save=False):
        samplerate=int(samplerate)
        window=window_s*samplerate #s
        shift=shift_s*samplerate
        std_emg=[]
        avg_emg=[]
        median_emg=[]
        l=len(self.values)
        data=self.values
        for j in range(0,l,shift): #15 or 0
            start=j
            end=j+window

            median_emg.append(np.median(data[start:end,0]))
            std_emg.append(np.std(data[start:end,0]))
            avg_emg.append(np.average(data[start:end,0]))

        ts=np.linspace(0,l/samplerate,len(std_emg))
#        print('lenlen',len(std_emg))
        plt.show()        
        d=self.values[:,index]
        x = np.arange(0,len(d))        
#        px = np.polyfit(x,d,30)
#        mmg_x_ploy=np.polyval(px,x)
#        plt.plot(x/samplerate+starttime,d,label=title+'_roll')
##        plt.plot(x*1/31+starttime,mmg_x_ploy,label='trend')
#        plt.xlabel('ts')
#        plt.title(title+"_roll (window="+str(window_s)+"(s)/shift="+str(shift_s)+"(s)")
#        plt.ylabel('degree')
#        plt.legend()
#        if save:
#            plt.savefig('./output/'+title+'_roll'+'.png')
#        plt.show()

        
        x = np.arange(0,len(std_emg))
#        px = np.polyfit(x,std_emg,30)
#        mmg_x_ploy=np.polyval(px,x)
#        plt.plot(ts/60+starttime/60,std_emg,label=title+'_std')
#        plt.scatter(ts/60+starttime/60,std_emg,label=title+'_std')
##        plt.plot(ts+starttime,mmg_x_ploy,label='trend')
#        plt.xlabel('ts')
#        plt.title(title+"_std (window="+str(window_s)+"(s)/shift="+str(shift_s)+"(s)")
#        plt.ylabel('degree')
##        plt.legend()
#        if save:
#            plt.savefig('./output/'+title+'_std'+'.png')
#        plt.show()
        plt.plot(ts/60+starttime/60,std_emg,label=title+'_std')
        plt.scatter(ts/60+starttime/60,std_emg)
        plt.plot(ts/60+starttime/60,avg_emg,label=title+'_avg')
        plt.scatter(ts/60+starttime/60,avg_emg)
        plt.plot(ts/60+starttime/60,median_emg,label=title+'_median')
        plt.scatter(ts/60+starttime/60,median_emg)
        x = np.arange(0,len(d))
        plt.plot(x/samplerate/60+starttime/60,d,label=title+'_roll', alpha=0.3)
#        px = np.polyfit(x,avg_emg,30)
#        mmg_x_ploy=np.polyval(px,x)
#        plt.plot(ts+starttime,mmg_x_ploy,label='trend')
        plt.xlabel('min')
#        plt.ylim(np.linspace(-90,90,5).tolist())
        
        plt.title(title+"_avg (window="+str(window_s)+"(s)/shift="+str(shift_s)+"(s)")
        plt.ylabel('degree')
#        plt.ylim([-90,90])
        plt.grid()
        plt.legend()
        
        if save:
            plt.savefig('./output/'+title+'_featire'+'.png')
        plt.show()
    def smoothListGaussian(self,strippedXs=False,degree=5,index=0):  
        data=self.values[:,index]
        window=degree*2-1  
        weight=np.array([1.0]*window)  
        weightGauss=[]  
        
        for i in range(window):  
            i=i-degree+1  
            frac=i/float(window)  
            gauss=1/(np.exp((4*(frac))**2))  
            weightGauss.append(gauss)
        weight=np.array(weightGauss)*weight  
        smoothed=[0.0]*(len(data)-window)  
        
        for i in range(len(smoothed)):        smoothed[i]=sum(np.array(data[i:i+window])*weight)/sum(weight)  
        temp=Signal(np.array(smoothed))
        temp.moving_avg=np.array(smoothed)
        
        return temp
        
#        x=l_press[:,1]
#        y=l_press[:,-1]
#        x = smoothListGaussian(x)
#        y = smoothListGaussian(y)
    def feature_analysis2(self,window_s,shift_s,samplerate,remind,index,title='',starttime=0,save=False):
        samplerate=int(samplerate)
        window=window_s*samplerate #s
        shift=shift_s*samplerate
        std_emg=[]
        avg_emg=[]
        median_emg=[]
        l=len(self.values)
        data=self.values
        for j in range(0,l,shift): #15 or 0
            start=j
            end=j+window
            temp=data[start:end,0]
#            print(len(temp))
            if remind=='remind':
                print('remind')
                if j==4*300*samplerate or j==11*300*samplerate:
                    temp=temp[:-(samplerate*60)]
                if j==5*300*samplerate or j==12*300*samplerate:
                    temp=temp[(samplerate*90):] 
            median_emg.append(np.median(temp))
            std_emg.append(np.std(temp))
            avg_emg.append(np.average(temp))

        ts=np.linspace(0,l/samplerate,len(std_emg))
        plt.show()        
        d=self.values[:,index]
        x = np.arange(0,len(d))        
        
        x = np.arange(0,len(std_emg))
        plt.plot(ts/60+starttime/60,std_emg,label=title+'_std')
        plt.scatter(ts/60+starttime/60,std_emg)
        plt.plot(ts/60+starttime/60,avg_emg,label=title+'_avg')
        plt.scatter(ts/60+starttime/60,avg_emg)
        plt.plot(ts/60+starttime/60,median_emg,label=title+'_median')
        plt.scatter(ts/60+starttime/60,median_emg)
        x = np.arange(0,len(d))
        plt.plot(x/samplerate/60+starttime/60,d,label=title+'_roll', alpha=0.3)
        plt.xlabel('min')
        
        plt.title(title+"_avg (window="+str(window_s)+"(s)/shift="+str(shift_s)+"(s)")
        plt.ylabel('degree')
        plt.grid()
        plt.legend()
        
        if save:
            plt.savefig('./output/'+title+'_featire'+'.png')
        plt.show()
    def show_peak(self,samplerate,distance=10,threshold=None):
        if threshold is None:
            threshold=np.mean(self.values)
        self.values=self.values.flatten()
        p_x, _ = find_peaks(self.values, height=threshold,distance=distance)
        plt.show()
        
        plt.plot(self.ts,self.values)
        plt.xlabel('ts')
        plt.ylabel('degree')
        self.peak_x=p_x/self.samplerate
        self.peak_y=self.values[p_x]
        plt.scatter(p_x/self.samplerate,self.values[p_x], marker="x",linewidths=10,color='r')
        plt.title(str(np.round(np.array(self.peak_y),1)))
        plt.show()

    def movingavg(self,window_s,shift_s,samplerate,index,title='',starttime=0,save=True,show_detail=False,showfig=False):
        print('calculating moving_avg')
        self.window_s=window_s
        self.shift_s=shift_s
        legend_tag=['_roll','_pitch','_yaw'][index[0]]
        print(index)
        print(legend_tag)
        samplerate=int(samplerate)
        window=int(window_s*samplerate) #s
        shift=int(shift_s*samplerate)
        avg_emg=[]
        l=len(self.values)
        data=self.values
        for j in range(0,l,shift): #15 or 0
            start=j
            end=j+window
            avg_emg.append(np.average(data[start:end,index[0]]))
        ts=np.linspace(0,l/samplerate,len(avg_emg))
#        plt.show()        
        d=self.values[:,index]
        x = np.arange(0,len(d))        
        self.after_moving=np.array(avg_emg)
#        plt.plot(x/samplerate+starttime,d,label=title+legend_tag)
#        plt.xlabel('ts')
#        plt.title(title+legend_tag+" (window="+str(window_s)+"(s)/shift="+str(shift_s)+"(s)")
#        plt.ylabel('degree')
#        plt.legend()
#        plt.show()   
        x = np.arange(0,len(avg_emg))
#        plt.plot(ts/60+starttime/60,avg_emg,linewidth=1,label=title+'_avg')
#        plt.scatter(ts/60+starttime/60,avg_emg)
        x = np.arange(0,len(d))
#        plt.plot(x/samplerate/60+starttime/60,d,label=title+legend_tag, alpha=0.3)
#        plt.xlabel('min')
#        plt.title(title+"_avg (window="+str(window_s)+"(s)/shift="+str(shift_s)+"(s)")
#        plt.ylabel('degree')
#        plt.grid()
#        plt.legend()
        if save:
            plt.savefig('./output/'+title+'_movingavg_'+str(index[0])+'.png')
#        plt.show()
        if show_detail:
            print('maxium',np.max(avg_emg))
            print('minium',np.min(avg_emg))
        h_index=np.argmax(avg_emg)
        l_index=np.argmin(avg_emg)
#        print(np.argmax(avg_emg))
#        print(np.argmin(avg_emg))
#        plt.scatter(ts[h_index],np.max(avg_emg))
#        plt.scatter(ts[l_index],np.min(avg_emg))
        self.mv_y=avg_emg
        self.mv_x=ts+starttime
        if showfig:
            plt.plot(x/samplerate+starttime,d,label=title+legend_tag)
            plt.xlabel('ts')
            plt.scatter(ts[h_index],np.max(avg_emg)+10, marker="x",linewidths=10,color='r')
            plt.scatter(ts[l_index],np.min(avg_emg)-10, marker="x",linewidths=10,color='r')
            plt.title(title+legend_tag+" (h_peak="+str(round(np.max(avg_emg),2))+" / l_peak="+str(round(np.min(avg_emg),2)))
            plt.ylabel('degree')
            plt.legend()
            plt.show()    
        self.peak_x=np.array([ts[l_index],ts[h_index]])
        self.peak_y=np.array([np.min(avg_emg),np.max(avg_emg)])
        self.moving=np.array(self.mv_y)
        self.moving_avg=Signal(self.moving.reshape(len(self.moving),1),samplerate=1/(self.window_s/(self.window_s/self.shift_s)))

#        from peakdetect import *
#        maxtab, mintab = peakdet(series,.5)
#        plot(series)
#        scatter(array(maxtab)[:,0], array(maxtab)[:,1], color='blue')
#        scatter(array(mintab)[:,0], array(mintab)[:,1], color='red')
#        show()
    def peak_detection(self,title='',save=True,show_detail=True,tag=''):
        maxtab, mintab = peakdet(self.values,.5)
        plt.show()
        plt.plot(self.values)
        plt.scatter(np.array(maxtab)[:,0], np.array(maxtab)[:,1], color='blue')
        plt.scatter(np.array(mintab)[:,0], np.array(mintab)[:,1], color='red')
        plt.savefig('./output/'+title+'_peak_detect_'+tag+'.png')
        plt.show()
        
        l=min(len(maxtab),len(mintab))
        
        self.dt=maxtab[:l,:]-mintab[:l,:]
        self.diff=maxtab[:l,:]-mintab[:l,:]
        self.diff[:l,0]=maxtab[:l,0]
        #num of max peak output
        num_peak=6
        index=self.diff[:,1].argsort()[-num_peak:]
        out=self.diff[np.sort(index),1]
        self.out=out
        print('answer is ',out)
        print(self.diff)
    def peak_cadence(self,title='',save=True,show_detail=True,tag='',threshold=.5):
        maxtab, mintab = peakdet(self.values,threshold)
        plt.show()
        plt.plot(self.values)
        plt.scatter(np.array(maxtab)[:,0], np.array(maxtab)[:,1], color='blue')
        plt.scatter(np.array(mintab)[:,0], np.array(mintab)[:,1], color='red')
        plt.savefig('./output/'+title+'_peak_detect_'+tag+'.png')
        plt.show()
        
        self.maxtab=maxtab
        self.mintab=mintab
#        l=min(len(maxtab),len(mintab))
        self.dt_min=np.diff(mintab[:,0])
#        self.dt_min_ts=np.cumsum(mintab[:,0])
        self.dt_min_ts=np.array(mintab[:,0],dtype=int)
        self.dt_max=np.diff(maxtab[:,0])
        self.dt_max_ts=np.array(maxtab[:,0],dtype=int)
#        self.dt=mintab[:l,:]-maxtab[:l,:]
#        self.diff=maxtab[:l,:]-mintab[:l,:]
#        self.diff[:l,0]=maxtab[:l,0]
#        #num of max peak output
#        num_peak=6
#        index=self.diff[:,1].argsort()[-num_peak:]
#        out=self.diff[np.sort(index),1]
#        self.out=out
#        print('answer is ',self.dt)
#        print(self.diff)
    def peak_green(self,title='',save=True,show_detail=True,tag='',threshold=30,sensitivity=0.1):
        maxtab, mintab = peakdet(self.values,sensitivity)
        plt.show()
        plt.plot(self.values)
        plt.scatter(np.array(maxtab)[:,0], np.array(maxtab)[:,1], color='blue')
        plt.scatter(np.array(mintab)[:,0], np.array(mintab)[:,1], color='red')
        plt.title(title+'_peak_detect_a_'+tag)
        plt.savefig('./output/'+title+'_peak_detect_a_'+tag+'.png')
        plt.show()
        
#        l=min(len(maxtab),len(mintab))
#        diff=mintab[1:l,:]-maxtab[:l-1,:]
#        diff[:l-1,0]=maxtab[:l-1,0]
        
        l=min(len(maxtab),len(mintab))
        diff=maxtab[:l,:]-mintab[:l,:] #計算峰對峰值
#        print('max',maxtab[:l,:].flatten())
#        print('min',mintab[:l,:].flatten())
        diff[:l,0]=maxtab[:l,0] #差值時間軸以max來算

        maxtab=maxtab[:l] #縮減長度
        mintab=mintab[:l] #縮減長度
        mk1=diff[:,1]>threshold #鎖定只要大角度差的數值
        mk2=diff[:,1]<130 #太大的也不要
        mk=np.logical_and(mk1,mk2) #合成mask
        trim_diff=diff[mk] 
        self.a1=maxtab[mk][:,1].astype(float)
        self.b1=mintab[mk][:,1].astype(float)
        print('max',maxtab[mk][:,1].astype(float))
        print('min',mintab[mk][:,1].astype(float))
#        print('trim_diff',trim_diff)
        #num of max peak output
#        print('after filter',diff)

        plt.plot(self.values)
        plt.scatter(np.array(maxtab)[mk,0], np.array(maxtab)[mk,1], color='g')
        plt.scatter(np.array(mintab)[mk,0], np.array(mintab)[mk,1], color='g')
#        print('maxshape',(np.array(mintab)[mk,1]).shape)

        plt.savefig('./output/'+title+'_peak_detect_b_'+tag+'.png')
        plt.title(title+'_peak_detect_'+tag)
        plt.show()
        num_peak=6
        index=trim_diff[:,1].argsort()[-num_peak:]
        out=trim_diff[np.sort(index),1]
        self.out_a=out
        self.temp=np.zeros(num_peak)
        self.temp[:len(self.out_a)]=self.out_a
        self.out_a=self.temp
#        print('answer is ',out)
        
        
        
        
#        index1=np.searchsorted(np.array(diff)[:,0],trim_diff[:2,0]).tolist()
#        temp_index2=trim_diff[2,0]
#        print('index2 is ',np.searchsorted(diff[:,0],temp_index2))
#        index2=np.array([temp_index2,temp_index2+1]).tolist()
##        threshold=min(diff[:,1])-1
##        print('threshold setting is ',threshold-1)
#        index=index1.append(index2)
#        
#        plt.show()
#        plt.plot(self.values)
#        plt.scatter(np.array(maxtab)[index,0], np.array(maxtab)[index,1], color='g')
#        plt.scatter(np.array(mintab)[index,0], np.array(mintab)[index,1], color='g')
##        plt.savefig('./output/'+title+'_peak_detect_'+tag+'.png')
#        plt.show()        
#        print(diff)
        
        maxtab=maxtab[1:,:]
        l=min(len(maxtab),len(mintab))
        maxtab=maxtab[:l,:]
        mintab=mintab[:l,:]

#        print('max',maxtab[:l,:].astype(float))
#        print('min',mintab[:l,:].astype(float))
        diff=mintab-maxtab
        
#        print('debug',diff)
        diff[:l,0]=maxtab[:l,0]
        mk1=diff[:,1]<-threshold
        mk2=diff[:,1]>-130
        mk=np.logical_and(mk1,mk2)
        self.a2=maxtab[mk][:,1].astype(float)
        self.b2=mintab[mk][:,1].astype(float)
        trim_diff=diff[mk]
#        print('maxtab??',(maxtab[:,1]).tolist())
#        print('mintab?',(mintab[:,1]).tolist())
        print('max',maxtab[mk][:,1].astype(float))
        print('min',mintab[mk][:,1].astype(float))
#        print('trim_diff',trim_diff)
        #num of max peak output
#        print('after filter',diff)

        plt.plot(self.values)
        plt.scatter(np.array(maxtab)[mk,0], np.array(maxtab)[mk,1], color='g')
        plt.scatter(np.array(mintab)[mk,0], np.array(mintab)[mk,1], color='g')
        plt.title(title+'_peak_detect_'+tag)
        plt.savefig('./output/'+title+'_peak_detect_'+tag+'.png')
        plt.show()
        num_peak=6
        index=trim_diff[:,1].argsort()[:num_peak]
        out=trim_diff[np.sort(index),1]
        self.out_b=out
        self.temp=np.zeros(num_peak)
        self.temp[:len(self.out_b)]=self.out_b
        self.out_b=self.temp
#        print('answer is ',out)
    @property
    def abs_sig(self):
        new=copy.copy(self)
        new.values=np.abs(new.values)
        return new
#    def moving_avg(self):
#        
#        self.moving_avg
#        return Signal(self.moving,samplerate=1/(self.window_s/(self.window_s/self.shift_s)))
#        
            
class Plot(object):
    
    def __init__(self,title,dtype,index=[-1]):
        self.title=title
        self.dtype=dtype
        self.fig = plt.figure()
        self.plt_angle=self.fig.add_subplot(111)
        self.plt_angle.set_title(self.title)
        self.index=index
        font_path = "C:/Windows/Fonts/kaiu.ttf"
        prop = mfm.FontProperties(fname=font_path, size=15)
        font_name = "Arial"
        txt_size=15

        if dtype=='axis':
            self.label_table=['x','y','z']
            self.plt_angle.set_ylabel('Amplitude')
#            self.plt_angle.ylable('Amplitude')
        elif dtype=='eu':
            self.plt_angle.set_ylabel('degree')
            self.label_table=['roll','pitch','yaw']
        elif dtype=='qq':
            self.plt_angle.set_ylabel('unit')
            self.label_table=['q0','q1','q2','q3']        
        elif dtype=='single':
            self.label_table=[str(self.title)]
    def plot_angle(self,*args,**kwargs):
        if self.index ==[-1]:
            count=range(args[-1].values.shape[1])
        else:
            count=self.index
        if len(args)==1:
            for i in count:
                self.plt_angle.plot(args[0].values[:,i],label=self.label_table[i])
            self.plt_angle.set_xlabel('n')
        else:

            for i in count:
                self.plt_angle.plot(args[0].values,args[1].values[:,i],label=self.label_table[i])
            self.plt_angle.set_xlabel('ts')
            
        self.plt_angle.legend()
        self.plt_angle.grid()
        self.fig.show()
        plt.show()
    def plot_compare_angle(self,dev1,dev2,index,err,showlegend=True):
        self.plt_angle.plot(dev1.ts.values,dev1.eu.values[:,index])
        self.plt_angle.plot(dev2.ts.values,dev2.eu.values[:,index])
#        self.plt_angle.plot(dev1.ts.values,dev1.eu.values[:,0])
#        self.plt_angle.plot(dev2.ts.values,dev2.eu.values[:,1])
        self.plt_angle.set_ylabel('degree')
        self.plt_angle.set_xlabel('ts')
#        ll=np.array(['imu_roll','imu_pitch','imu_yaw','vicon_roll','vicon_pitch','vicon_yaw'])
        ll=np.array(['imu_roll','imu_pitch','imu_yaw','ref_roll','ref_pitch','ref_yaw'])
        
        new_index = [x+3 for x in index]
        index=index+new_index
        coordinate=dev1.ts.values[0]
        if showlegend:
            self.plt_angle.text(coordinate, 0, 'static_err='+str(err[0])+'\ndynamic_err='+str(err[1]), style='italic', bbox={'facecolor':'yellow', 'alpha':0.5, 'pad':10})
        self.plt_angle.legend(ll[index].tolist())
        self.plt_angle.grid()        
        
    def plot_body_angle(self,body,index=None,cali_error=False,delay=0):
#        vicon_bias=body.vicon.bias
#        imgj_bias=body.ImgJ.bias
        imu_ts=body.imu1.ts.values
        txt_size=20
        imu_samplerate=body.imu1.samplerate
#        vicon_samplerate=120
        if body.imu1.position=='head' or body.imu1.position=='c3' or body.imu1.position=='c4':
            self.p1,self.p2=100,-45

            if index is None:
                index=[0]
#            self.plt_angle.set_yticks(range(-90,100,30))
#            self.plt_angle.set_ylim((-90,90))
            if not body.imu2 is None:#bodyframe:
                self.plt_angle.set_title(r'$\alpha_{1,'+body.imu1.position+'-'+body.imu2.position+'}$'+'\n', fontsize=txt_size)

            else:
                self.plt_angle.set_title(r'$\alpha_{1,'+body.imu1.position+'}$'+'\n', fontsize=txt_size)
#            self.imgj_ts=np.linspace(body.imu1.start,imu_ts[-1],len(ImgJ.shoulder_angle.values[imgj_bias:]))
#            self.imgj_angle=body.ImgJ.head_angle.values[imgj_bias:]
            self.alphaid='1'

        elif body.imu1.position=='ls' or body.imu1.position=='rs':
            if index is None:
                index=[2]
#       
#            self.plt_angle.set_yticks(range(-60,70,20))
#            self.plt_angle.set_ylim((-60,60))
            self.p1,self.p2=100,75
            if body.imu2 is not None:   #bodyframe:
                self.plt_angle.set_title(r'$\alpha_{3,'+body.imu1.position+'-'+body.imu2.position+'}$'+'\n', fontsize=txt_size)
            else:          
                self.plt_angle.set_title(r'$\alpha_{3,'+body.imu1.position+'}$'+'\n', fontsize=txt_size)
#            self.imgj_ts=np.linspace(body.imu1.start,imu_ts[-1],len(ImgJ.shoulder_angle.values[imgj_bias:]))
#            self.imgj_angle=body.ImgJ.shoulder_angle.values[imgj_bias:]
            self.alphaid='3'
        #畫圖  
#        self.imu_angle=body.imu1.eu.values[:,index[0]]
        self.imu_angle=body.imu1.eu.values[:,index]
        self.gyr=body.imu1.gyr.values
#        self.imu_ts=body.imu1.ts.values*1 #maybe not 1
        self.imu_ts=body.imu1_body.ts.values*1 #maybe not 1
        self.engin=int(self.alphaid)-1
#        self.vicon_angle=body.vicon.vicon_angle.values[vicon_bias*120:,self.engin]
#        self.vicon_ts=np.linspace(body.imu1.start,body.imu1.ts.values[-1],len(vicon.vicon_angle.values[vicon_bias*120:]))
#        if not body.imu1.adoptor.session==3:
#            self.vicon_angle-=self.vicon_angle[0]
#        else:
#            #特例
#            self.p1,self.p2=200,10
#            temp=np.mean(self.vicon_angle[100*120:110*120])-np.mean(self.imu_angle[100*75:110*75])
#            print(self.vicon_angle.shape)
#            print(self.imu_angle.shape)
#            self.vicon_angle-=temp
        imu_delay=int(delay*imu_samplerate)
#        vicon_delay=delay*vicon_samplerate
        imu_delay=0
        imgj_delay=delay*1
        imgj_delay=0

        print('imu_delay',imu_delay)
        for i in index:               
#            self.plt_angle.plot(self.imu_ts[imu_delay:],body.imu1.eu.values[imu_delay:,i],label=r'$\alpha_{'+self.alphaid+','+'imu'+'}$')
            self.plt_angle.plot(self.imu_ts[imu_delay:],body.imu1_body.eu.values[:,i],label=r'$\alpha_{'+self.alphaid+','+'imu'+'}$')

#        self.plt_angle.plot(self.imgj_ts[imgj_delay:],self.imgj_angle[imgj_delay:],label=r'$\alpha_{'+self.alphaid+','+'imgj'+'}$')
#        self.plt_angle.plot(self.vicon_ts[vicon_delay:],self.vicon_angle[vicon_delay:],label=r'$\alpha_{'+self.alphaid+','+'vicon'+'}$')
        #剪取訊號 後續算誤差用新的
        self.imu_ts=self.imu_ts[imu_delay:]
        body.imu1.eu.values=body.imu1.eu.values[imu_delay:,:]
        self.gyr=self.gyr[imu_delay:,:]
#        self.imgj_ts=self.imgj_ts[imgj_delay:]
#        self.imgj_angle=self.imgj_angle[imgj_delay:]
#        self.vicon_ts=self.vicon_ts[vicon_delay:]
#        self.vicon_angle=self.vicon_angle[vicon_delay:]
#            
            
        #    plt.yticks(range(-30,40,10),fontsize=txt_size)
#        self.gyr=np.linspace(body.imu1.start,body.imu1.gyr.values[-1],len(vicon.vicon_angle.values[vicon_bias*120:]))
            
        self.plt_angle.set_xlabel(r'$\mathit{t}$ ($\mathit{sec}$)', fontsize=txt_size-5)
        self.plt_angle.set_ylabel(r'$\mathit{degree}$', fontsize=txt_size-5)
#        
      
        if cali_error:
            mask=np.linalg.norm(self.gyr,axis=1)>=(5)#/180*np.pi)
            err_eu1_s=np.zeros([len(self.gyr),1])
            err_eu1_d=np.zeros([len(self.gyr),1])
            index2=np.empty([0],dtype=int)

            for i in range(len(self.gyr)):
                index2=np.append(index2,np.searchsorted(self.vicon_ts,self.imu_ts[i]))
                if index2[-1]==len(self.vicon_ts):
                    err_eu1_s[i]=0
                    err_eu1_d[i]=0
                    index2[-1]=len(self.vicon_ts)-1
                elif not(mask[i]):# 靜態
                    err_eu1_s[i]=self.imu_angle[i]-self.vicon_angle[index2[-1]]
                else:# 動態
                    err_eu1_d[i]=self.imu_angle[i]-self.vicon_angle[index2[-1]]
        #    print(np.vstack(ref_ts,esti_ts))
            rms_eu1_s = np.sqrt(np.mean(err_eu1_s**2))
            rms_eu1_d = np.sqrt(np.mean(err_eu1_d**2))
            
            rms_eu1=rms_eu1_s+rms_eu1_d
            rms_eu1_s=round(float(rms_eu1_s),2)
            rms_eu1_d=round(float(rms_eu1_d),2)
            rms_eu1=round(float(rms_eu1),2)
            self.plt_angle.text(self.p1, self.p2, 'static_err='+str(rms_eu1_s)+'\ndynamic_err='+str(rms_eu1_d), style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
#            self.plt_angle.text(self.p1, self.p2, 'rms_err='+str(rms_eu1), style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

        self.plt_angle.legend()
        self.plt_angle.grid()
      
class Adaptor(object):
    def __init__(self,date,subject,session,raw_or_cali=None):
        self.subject=subject
        self.session=session
        self.version=date
        self.raw_or_cali=raw_or_cali
        self.date=date
    def nedadj(self,gyr_order):
        acc_order=-np.copy(gyr_order)
        mag_order=np.copy(gyr_order)
        opt=gyr_order==1
        minus=gyr_order==-1
        mag_order[opt]=-1
        mag_order[minus]=1
#        print('mag_order',mag_order)
        return acc_order,gyr_order,mag_order
    def nedadj_mpu9250(self,gyr_order):
        acc_order=-np.copy(gyr_order)
        mag_order=np.copy(gyr_order)
        opt=gyr_order==1
        minus=gyr_order==-1
#        mag_order[opt]=-1
#        mag_order[minus]=1
        mag_order[0],mag_order[1]=mag_order[1],mag_order[0]
        mag_order[2]=-mag_order[2]
        print('mag_order',mag_order)
        return acc_order,gyr_order,mag_order
    def enuadj(self,gyr_order):
        acc_order=np.copy(gyr_order)
        mag_order=np.copy(gyr_order)
        opt=gyr_order==1
        minus=gyr_order==-1
        mag_order[opt]=-1
        mag_order[minus]=1
        return acc_order,gyr_order,mag_order
    def enuadj_mpu9250(self,gyr_order):
        acc_order=np.copy(gyr_order)
        mag_order=np.copy(gyr_order)
#        opt=gyr_order==1
#        minus=gyr_order==-1
#        mag_order[opt]=-1
#        mag_order[minus]=1
        mag_order[0],mag_order[1]=mag_order[1],mag_order[0]
        mag_order[2]=-mag_order[2]
        return acc_order,gyr_order,mag_order
    def to_qeb(self,start=1,end=3):#eath to body frame
        samplerate=int(self.samplerate)
        rest=range(start*samplerate,end*samplerate)
        qbe=Quaternion(np.mean(self.qq.values[rest,:],axis=0))
        self.theta=np.zeros([len(self.qq.values),1])
#        print('self.eu.values.shape',self.eu.values.shape)
#        self.eu=np.zeros([len(self.qq.values),3])
        for i in range(len(self.eu.values)):
            tempq=(qbe.conj())*Quaternion(self.qq.values[i,:])
            self.qq.values[i,:]=tempq.__array__()
#            self.qq.values[i,:]=tempq.elements
           
#            self.eu.values[i,0],self.eu.values[i,1],self.eu.values[i,2]=tempq.to_euler_angles_by_wiki()
#            print(tempq.to_euler_angles_by_wiki())
#            self.eu.values[i]=tempq.eu_angle
            self.eu.values[i,:]=tempq.to_euler_angles_by_wiki()
            
            self.theta[i]=np.arccos(self.qq.values[i,0])*2/np.pi*180
        self.eu.values=self.eu.values/np.pi*180
        self.theta=Signal(self.theta,samplerate=self.qq.samplerate)
    def to_nedenu(self):#eath to body frame
        samplerate=int(self.samplerate)
        self.qq.values[:,[0,1,2,3]]=self.qq.values[:,[0,2,1,3]]
        self.qq.values[:,3]=-self.qq.values[:,3]
        for i in range(len(self.qq.values)):
            tempq=Quaternion(self.qq.values[i,:])
            self.qq.values[i,:]=tempq._get_q()
#            self.eu.values[i,0],self.eu.values[i,1],self.eu.values[i,2]=tempq.eu_angle
            self.eu.values[i,0],self.eu.values[i,1],self.eu.values[i,2]=tempq.to_euler_angles_by_wiki
            
        self.eu.values=self.eu.values/np.pi*180
    def dehead(self,start):
        index=int(self.samplerate)*start
        self.start=start
        self.qq.values=self.qq.values[index:]
        self.eu.values=self.eu.values[index:]
        self.ts.values=self.ts.values[index:]
        self.acc.values=self.acc.values[index:]
        self.gyr.values=self.gyr.values[index:]        
        self.mag.values=self.mag.values[index:]    
    def show_eu(self,index=[0,1,2]):
        eu_plt=Plot(title=self.file_tag+str(self.position)+'_eu',dtype='eu',index=index)
        eu_plt.plot_angle(self.ts,self.eu)
    def show_qq(self,index=[0,1,2,3]):
        eu_plt=Plot(title=self.file_tag+str(self.position)+'_qq',dtype='qq',index=index)
        eu_plt.plot_angle(self.ts,self.qq)
    def show_eu_by_index(self,index=[0]):
        eu_plt=Plot(title=self.file_tag+str(self.position)+'axis',dtype='eu',index=index)
        eu_plt.plot_angle(self.ts,self.eu)
    def to_master_eu(self):
        for i in range(len(self.q.values)):
            self.eu.values[i,0],self.eu.values[i,1],self.eu.values[i,2]=math.asin(self.q.values[i,1]),math.asin(self.q.values[i,2]),math.asin(self.q.values[i,3])
class SafePlay(Adaptor):
    def __init__(self,adoptor):
        self.adoptor=adoptor
#        self.ts=adoptor.ts
#        self.delta_ts=adoptor.delta_ts
        self.date=adoptor.date
#        self.sampleperiod=((self.ts.values[-1]-self.ts.values[0])/len(self.ts.values))
#        self.samplerate=1/self.sampleperiod
#        self.file_tag=adoptor.file_tag
        self.dev_id=0
    def adj_time(self,arry):
        end=len(arry)
        df=pd.DataFrame(data=arry)
        array=df.iloc[0:end,1].str.replace('-',' ').str.replace(':',' ').str.split(' ',expand=True).as_matrix()
        array=np.array(array,dtype=float)
        temp_ts=array[:,3]*60*60+array[:,4]*60+array[:,5]+array[:,6]*0.001
        ts=temp_ts-temp_ts[0]
        return ts
    def adj_press(self,arry):
        factor = np.array([4.7882, 6.3838, 5.4735, 5.1219, 3.34813, 3.34813, 6.3338, 2.5824, 3.5912, 6.4659, 3.9352, 4.3927, 4.3981, 4.585, 3.9352, 4.44149, 4.4054, 4.7767])
        Rinsole=arry[:,2:20]
        Rinsole=np.array(Rinsole,dtype='float')
        RnewtonSeries = np.zeros_like(Rinsole)
#        LnewtonSeries = np.zeros_like(Linsole)
        #F(newton)
        print(factor.shape[0])
        for i in range(0,factor.shape[0]):   
            RnewtonSeries[:,i] = Rinsole[:,i]*factor[i]*0.015*1.02*9.8
#            LnewtonSeries[:,i] = Linsole[:,i]*factor[i]*0.015*1.02*9.8
        RnewtonSeries=np.cumsum(RnewtonSeries,axis=1)[:,-1]
        return RnewtonSeries

    def get_data(self):
        if self.date=='190423':
            db=SQLite(path='D:/iii/exp_data/safeplay/190423cycling2min.sqlite')
            print(db.table_list)
            #print('try to read ',tablename)
            tablenames=['leftinsole_imu','rightinsole_imu','leftinsole_pressure','rightinsole_pressure']
            self.file_tag=self.date='safeplay'
            output=[]
            self.condition='time BETWEEN \'2019-04-23 00:28\' AND \'2019-04-23 12:31\''
        if self.date=='190423-2':
            db=SQLite(path='D:/iii/exp_data/safeplay/190423cycling2min.sqlite')
            print(db.table_list)
            #print('try to read ',tablename)
            tablenames=['leftinsole_imu','rightinsole_imu','leftinsole_pressure','rightinsole_pressure']
            self.file_tag=self.date='safeplay'
            output=[]
            self.condition='time BETWEEN \'2019-04-23 12:28\' AND \'2019-04-23 17:31\''
        if self.date=='190423-3':
            db=SQLite(path='D:/iii/exp_data/safeplay/190423cycling2min.sqlite')
            print(db.table_list)
            #print('try to read ',tablename)
            tablenames=['leftinsole_imu','rightinsole_imu','leftinsole_pressure','rightinsole_pressure']
            self.file_tag=self.date='safeplay'
            output=[]
            self.condition='time BETWEEN \'2019-04-23 17:28\' AND \'2019-04-23 19:31\''
        for tablename in tablenames:
            imu007=Table(db,tablename)
            raw=imu007.export_data(condition=self.condition)
            
            raw[:,1]=self.adj_time(raw)
            raw=np.array(raw,dtype=float)
            self.table=tablename
            self.position=tablename
            
            self.ts=Signal(raw[:,1])
            self.delta_ts=Signal(np.diff(self.ts.values))
            self.beta=0.5
            self.file_tag=tablename
            if tablename=='leftinsole_pressure'or tablename=='rightinsole_pressure':
                raw[:,20]=self.adj_press(raw)
                output.append(raw)
            else:
#                self.acc=raw[:,2:5]
#                self.gyr=raw[:,6:9]
#                self.mag=raw[:,10:13]
                self.sampleperiod=((self.ts.values[-1]-self.ts.values[0])/len(self.ts.values))
                self.samplerate=1/self.sampleperiod
                self.acc=Signal(values=raw[:,2:5]/4096,tag=self.file_tag+'acc',samplerate=self.samplerate)
                self.gyr=Signal(values=raw[:,6:9]/32,tag=self.file_tag+'gyr',samplerate=self.samplerate)
#                self.acc=self.acc.changeorder([1,-3,2])
#                self.gyr=self.gyr.changeorder([1,-3,2])
                self.mag=Signal(values=raw[:,10:13],tag=self.file_tag+'mag',samplerate=self.samplerate)

                
                self.body_table=tablename
                fake_imu=IMU(self,mag_on=False,beta=0.5,mag_cali_on=False,acc_cali_on=False)
#                self.acc=Signal(values=raw[:,2:5],tag=self.file_tag+'acc',samplerate=self.samplerate)
#                self.acc=Signal(values=raw[:,2:5],tag=self.file_tag+'acc',samplerate=self.samplerate)
                output.append(fake_imu)
        l_imu,r_imu,l_press,r_press=output[0],output[1],output[2],output[3]
        
        
        return l_imu,r_imu,l_press,r_press
    

class IMU(Adaptor):
    def __init__(self,adoptor,mag_on,beta,mag_cali_on,acc_cali_on):
        self.adoptor=adoptor
        self.id=self.adoptor.dev_id
        self.table=self.adoptor.body_table
        print('iiiiiiiddd',self.id)
        self.position=self.table[self.id]
        self.acc=adoptor.acc
        self.gyr=adoptor.gyr
        self.mag=adoptor.mag
        self.ts=adoptor.ts
        self.delta_ts=adoptor.delta_ts
        self.beta=beta
        self.mag_on=mag_on
        self.sampleperiod=((self.ts.values[-1]-self.ts.values[0])/len(self.ts.values))
        self.samplerate=1/self.sampleperiod
        self.file_tag=adoptor.file_tag
        if(adoptor.accMag is not None):
            self.accMag=self.adoptor.accMag
        if(adoptor.quaternion is not None):
            self.qq=adoptor.quaternion
        #        self.file_tag=adoptor.file_tag+'_'+str(round(self.samplerate,0))
        print('the samplerate is',self.samplerate)
        if mag_cali_on:
            print('Loading cali_parameter')
            if self.adoptor.version=='0724':
                self.para=Adp_aiq(self.adoptor.version,self.adoptor.subject,self.adoptor.session,self.adoptor.dev_name,raw_or_cali='cali')
                self.para.mag.denoise(300)

                self.mag.values=cali_mag(self.para.mag.values,self.mag.values)
            elif self.adoptor.version=='0801':
                self.para=Adp_aiq(self.adoptor.version,self.adoptor.subject,self.adoptor.session,self.adoptor.dev_name,raw_or_cali='cali')
                self.para.mag.denoise(300)
#                self.para.mag.denoise_by_std()

            elif self.adoptor.version=='0825':
                self.para=Adp_aiq(self.adoptor.version,self.adoptor.subject,self.adoptor.session,self.adoptor.dev_name,raw_or_cali='cali')
                self.para.mag.denoise(300)
                
                self.gyr_para=np.array([[0.865023275,-0.660821468,0.099422234]
                                    ,[1.047329048,-2.241167943,-0.530916302]
                                    ,[1.164640317,-1.485872538,-0.165426696]
                                    ,[0.114630744,-1.263877188,0.027753009]
                                    ,[0,0,0]
                                    ,[0.639037199,-0.682739333,0.354738786]
                                    ,[0.769789387,-3.622423414,-1.214967177]
                                    ,[0.752423414,-0.870873906,0.479009847]])
                self.adoptor.gyr.values-=self.gyr_para[subject,:]
            elif self.adoptor.version=='1004':
                self.para=Adp_aiq(self.adoptor.version,self.adoptor.subject,self.adoptor.session,self.adoptor.dev_name,raw_or_cali='cali')
#                self.para.mag.denoise(300)
                print(self.para.mag.values)
            elif self.adoptor.version=='1017':
                self.para=Adp_aiq(self.adoptor.version,self.adoptor.subject,self.adoptor.session,self.adoptor.dev_name,raw_or_cali='cali')
#                self.para.mag.denoise(300)
                print(self.para.mag.values)
            elif self.adoptor.version=='1102':
                self.para=Adp_aiq(self.adoptor.version,self.adoptor.subject,self.adoptor.session,self.adoptor.dev_name,raw_or_cali='cali')
                print(self.para.mag.values)
            elif self.adoptor.version=='220329':
                self.para=Adp_ngimu(self.adoptor.version,self.adoptor.subject,self.adoptor.session,self.adoptor.dev_name,raw_or_cali='cali')
                print('220329 try to calibrate mag')
                print(self.para.mag.values)
            elif self.adoptor.version=='1205':
                self.para=Adp_aiq(self.adoptor.version,self.adoptor.subject,self.adoptor.session,self.adoptor.dev_name,raw_or_cali='cali')
                print(self.para.mag.values)
            elif self.adoptor.version=='1218':
                self.para=Adp_aiq(self.adoptor.version,self.adoptor.subject,self.adoptor.session,self.adoptor.dev_name,raw_or_cali='cali')
                self.mag.values=ellipse_fitting2(self.para.mag.values,self.mag.values)
            elif self.adoptor.version=='191031':
                self.para=Adp_aiq(self.adoptor.version,self.adoptor.subject,self.adoptor.session,self.adoptor.dev_name,raw_or_cali='cali',cali_path=self.adoptor.cali_path)
#                self.mag.values=ellipse_fitting2(self.para.mag.values,self.mag.values)
                print('calipath is ',self.para.pathway)
                self.mag.values=ellipse_fitting(self.mag.values,self.para.pathway,self.para.mag.values)

            elif self.adoptor.version=='remind' or self.adoptor.version=='190117' :
                self.para=Adp_aiq(self.adoptor.version,self.adoptor.subject,self.adoptor.session,self.adoptor.dev_name,raw_or_cali='cali')
#                self.para.mag.fft(30,10)
#                self.para.mag.denoise_by_1std()
                plt.show()
                plt.scatter(self.mag.values[:,0],self.mag.values[:,1])
                plt.scatter(self.mag.values[:,1],self.mag.values[:,2])
                plt.scatter(self.mag.values[:,0],self.mag.values[:,2])
                plt.title('before calibration')
                plt.show()
                self.mag.values=ellipse_fitting2(self.para.mag.values,self.mag.values)
                return
            elif self.adoptor.version=='without_remind':
                self.para=Adp_aiq(self.adoptor.version,self.adoptor.subject,self.adoptor.session,self.adoptor.dev_name,raw_or_cali='cali')
#                self.para.mag.denoise_by_1std()
                plt.show()
                plt.scatter(self.mag.values[:,0],self.mag.values[:,1])
                plt.scatter(self.mag.values[:,1],self.mag.values[:,2])
                plt.scatter(self.mag.values[:,0],self.mag.values[:,2])
                plt.title('before calibration')
                plt.show()
                self.mag.values=ellipse_fitting2(self.para.mag.values,self.mag.values)
                return
            elif self.adoptor.version=='ntuh_work':
                self.para=Adp_aiq(self.adoptor.version,self.adoptor.subject,self.adoptor.session,self.adoptor.dev_name,raw_or_cali='cali')
#                self.para.mag.denoise_by_1std()
                plt.show()
                plt.scatter(self.mag.values[:,0],self.mag.values[:,1])
                plt.scatter(self.mag.values[:,1],self.mag.values[:,2])
                plt.scatter(self.mag.values[:,0],self.mag.values[:,2])
                plt.title('before calibration')
                plt.show()
                self.mag.values=cali_mag(self.para.mag.values,self.mag.values)

#                self.mag.values=ellipse_fitting2(self.para.mag.values,self.mag.values)
                return            
            try:
                print(sadfae)
            except:
#                from termcolor import colored
#                print(colored('hello', 'red')
#                print('mag)parameter no match!!!!')
                
                from termcolor import colored, cprint
                # 红色背景字符不发光
                text = colored('mag)parameter no match!!!!', 'red', attrs=['reverse', 'blink'])
                print(text)
        if acc_cali_on:
            from calibraxis import Calibraxis
            c = Calibraxis()
            self.acc_para=Adp_aiq(self.adoptor.version,self.adoptor.subject,self.adoptor.session,self.adoptor.dev_name,raw_or_cali='acc_cali')
            
            plt.show()
            plt.plot(self.acc_para.acc.values)
            plt.show()
            
            plt.plot(self.acc_para.gyr.norm().values)
            plt.show()            
#            self.acc_para.acc.denoise
#            self.acc_para.acc.denoise_by_std()
#            mk=self.acc_para.gyr.norm().values<10
#            mk2=self.acc_para.acc.norm().values<1.1
            
#            mk2=np.abs(self.acc_para.gyr.acc.values)>0.8
#            mk=np.logical_and(mk,mk2)
#            print(mk.flatten())
#            self.acc_para.acc.values=self.acc_para.acc.values[mk.flatten(),:]
            plt.plot(self.acc_para.acc.values)
            plt.show()        
            c.add_points(self.acc_para.acc.values)
            print(self.acc_para.acc.values.shape)
            print('acc_calibration calculating')
            c.calibrate_accelerometer() 
            print('acc_calibration complete')
            self.acc.values=c.batch_apply(self.acc.values)
#            if not self.adoptor.version=='1218':
#                self.mag.values=cali_mag(self.para.mag.values,self.mag.values)
#                c = Calibraxis()
#                points=self.para.acc.values
#                c.add_points(points)
#                self.acc.values=c.batch_apply(points)
#                
#                self.mag.values=cali_mag(self.para.mag.values,self.mag.values)

    def calc_angle(self,fake_t_on,filter_on=False,acc_algo=False):
#        self.qq,self.eu=calc_orientation(self.gyr.values/180*np.pi,self.acc.values,100,Quaternion(1,0,0,0),self.beta,self.delta_ts.values,1)
#        self.qq=Signal(self.qq,tag=self.adoptor.file_tag+'_qq',samplerate=self.samplerate)
#        self.eu=Signal(self.eu,tag=self.adoptor.file_tag+'_eu',samplerate=self.samplerate)
#        return
        if not acc_algo:            
            print('calculating_angle_by_madgwick')
            if (fake_t_on):
                self.delta_ts.values=np.ones_like(self.delta_ts.values)*self.sampleperiod
            if self.mag_on:
#                self.qq,self.eu=calc_orientation_mag(self.gyr.values/180*np.pi,self.acc.values,self.mag.values,100,Quaternion(1,0,0,0),self.beta,self.delta_ts.values/1,1)
                self.qq,self.eu=calc_orientation_mag(self.gyr.values/180*np.pi,self.acc.values,self.mag.values,100,Quaternion(1,0,0,0),self.beta,self.delta_ts.values/1,1)
#                self.qq,self.eu,self.qa,self.eua=calc_orientation_c(self.gyr.values/180*np.pi,self.acc.values,self.mag.values,100,Quaternion(1,0,0,0),20,5,self.delta_ts.values/1,1)
            else:
                warnings.warn('calculate angle without mag may cause yaw drift')
                warnings.warn('calculate angle without mag/adj_time/realtime')
                
                self.qq,self.eu=calc_orientation(self.gyr.values/180*np.pi,self.acc.values,100,Quaternion(1,0,0,0),self.beta,self.delta_ts.values,1)

            if filter_on:
                l3db=4
                h3db=15
                self.qq,self.eu=all_filter(self.q,self.eu,self.samplerate,l3db,h3db)
        else:
            print('calculating_angle_by_acc')

            import math
            angle=calc_angle_acc(self.acc.values)
            self.eu=np.zeros_like(self.acc.values)

            self.qq=np.zeros([len(self.eu),4])
            
            self.eu[:,0:2]=angle
            for i in range(len(self.eu)):
                x,y,z=self.eu[i,0],self.eu[i,1],self.eu[i,2]
                z = z/2.0
                y = y/2.0
                x = x/2.0
                cz = math.cos(z)
                sz = math.sin(z)
                cy = math.cos(y)
                sy = math.sin(y)
                cx = math.cos(x)
                sx = math.sin(x)
                self.qq[i,:]=np.array([
                         cx*cy*cz - sx*sy*sz,
                         cx*sy*sz + cy*cz*sx,
                         cx*cz*sy - sx*cy*sz,
                         cx*cy*sz + sx*cz*sy])
            
        self.qq=Signal(self.qq,tag=self.adoptor.file_tag+'_qq',samplerate=self.samplerate)
        self.eu=Signal(self.eu,tag=self.adoptor.file_tag+'_eu',samplerate=self.samplerate)
#        showbias()
        print('calc_angle succeed')
    def show_raw(self):
        # eu_plt=Plot(title=str(self.dev_name)+'_eu',dtype='eu',index=index)
        acc_plt=Plot(title=self.file_tag+str(self.position)+'_acc',dtype='axis')
        acc_plt.plot_angle(self.acc)
        gyr_plt=Plot(title=self.file_tag+str(self.position)+'_gyr',dtype='axis')
        gyr_plt.plot_angle(self.gyr)
        mag_plt=Plot(title=self.file_tag+str(self.position)+'_mag',dtype='axis')
        mag_plt.plot_angle(self.mag)
        mag_norm_plt=Plot(title=self.file_tag+str(self.position)+'mag_norm',dtype='single')
        mag_norm_plt.plot_angle(self.ts,self.mag.norm())
        

#        self.eu.values[]
#        self.eu.values=self.eu.values/np.pi*180*2
    def denoise(self,threshold):
        index=np.argwhere(self.mag.norm().values>threshold)
        self.acc.values=np.delete(self.acc.values,index,axis=0)
        self.gyr.values=np.delete(self.gyr.values,index,axis=0)
        self.mag.values=np.delete(self.mag.values,index,axis=0)
        self.ts.values=np.delete(self.ts.values,index,axis=0)
    def autoadg_by_gyr(self,index=0,threshold=10,dynamic=False):
        static_gry=self.gyr.norm().values>threshold
        static_gry=np.where(static_gry)
        static_gry=static_gry[0]
        cum=np.empty([0,2])
        for i in range(len(static_gry)-1):
            if (static_gry[i+1]-static_gry[i])>(10*self.samplerate):
                
                temp=np.array([static_gry[i],static_gry[i+1]])

                cum=np.vstack((cum,temp))
        print(cum)
        
        print(self.eu.values[int(cum[0,1]),2])
        print(self.eu.values[int(cum[0,0]),2])
        print((int(cum[0,1])-int(cum[0,0])))
        print("index from 0 to",len(cum)-1)
        self.cum=cum
        start=int(cum[index,0])
        end=int(cum[index,1])
        diff_yaw=self.eu.values[end,2]-self.eu.values[start,2]
        print('diff_yaw',diff_yaw/(start-end))
        diff_yaw=diff_yaw/(start-end)
        
        plt.show()
#        plt.title('static click at',self.adadoptor.body_table[self.adoptor.dev_id])
#        print(self.adoptor.body_table)
#        print(self.adoptor.dev_id)
#        print(self.adoptor.body_table[self.adoptor.dev_id])
#        plt.title('static click at '+self.adoptor.body_table[self.adoptor.dev_id])
        ys=self.eu.values[start:end,2]
        
        ts=self.ts.values[start:end]
        x = np.arange(0,len(ys))
        px = np.polyfit(x,ys,1)
        mmg_x_ploy=np.polyval(px,x)
        plt.plot(ts,self.eu.values[start:end,:],label='clip')
#        plt.plot(ts,ys)
        plt.plot(ts,mmg_x_ploy,label='fit curve',linestyle='--',linewidth=3)
        plt.legend()
        plt.show()
        if dynamic:
            self.rotatebackrate_dynamic()
        else:
            self.rotatebackrate(-diff_yaw)
    def autorotate2(self,theta):
#        cum=0
        for i in range(len(self.eu.values)-1):
#            cum+=theta
            a=theta*(i+1)
#            self.eu.values[i,2]+=a
#            ba=abs(int(self.eu.values[i,2]/180))
##            print(ba)
            self.eu.values[i,2]+=a
            while(self.eu.values[i,2]>150):
                if (self.eu.values[i,2]>150):
                    self.eu.values[i,2]-=180
            while(self.eu.values[i,2]<-150):
                if (self.eu.values[i,2]<-150):
                    self.eu.values[i,2]+=180
#            if (self.eu.values[i,2]<-180):
#                self.eu.values[i,2]+=180*(ba-1)
            
            
#            if(self.eu.values[i+1,2]>self.eu.values[i,2]):
#                self.eu.values[i,2]+=a
#            else:
#                self.eu.values[i,2]-=a
            
#        
    def autorotate(self,theta):
        rad2=theta/180*np.pi
        for i in range(len(self.qq.values)):
            rad=rad2*i
            tempq=Quaternion(self.qq.values[i])
            tempq=tempq*Quaternion(np.cos(rad/2),0,0,-np.sin(rad/2))
#            self.qq.values[i]=tempq.elements
            self.qq.values[i]=tempq.__array()
#            self.eu.values[i,0],self.eu.values[i,1],self.eu.values[i,2]=tempq.eu_angle
            self.eu.values[i,0],self.eu.values[i,1],self.eu.values[i,2]=tempq.to_euler_angles_by_wiki()

        self.eu.values=self.eu.values/np.pi*180
#        self.eu.values=-self.eu.values
    def rotatebackrate(self,theta):
        rad2=theta/180*np.pi
        for i in range(len(self.qq.values)):
            rad=rad2*i
            tempq=Quaternion(self.qq.values[i])
#            q2=Quaternion(1,0,0,0).from_angle_axis(rad,0,0,1)
            q2=q2=Quaternion(1,0,0,0).rotateaxis(rad/np.pi*180,[0,0,1])
#            tempq=tempq*q2
            tempq=q2*tempq

#            self.qq.values[i]=tempq.elements
            self.qq.values[i]=tempq.__array__()
#            self.eu.values[i,0],self.eu.values[i,1],self.eu.values[i,2]=tempq.eu_angle
            self.eu.values[i,0],self.eu.values[i,1],self.eu.values[i,2]=tempq.to_euler_angles_by_wiki()
            
        self.eu.values=self.eu.values/np.pi*180
    def rotatebackrate_dynamic(self):
        error=[]
        slope_stack=[]
        for j in range(len(self.cum)):
            print('fit duration')
            start=int(self.cum[j,0])
            end=int(self.cum[j,1])
            ys=self.eu.values[start:end,2]
            slope=(self.eu.values[end,2]-self.eu.values[start,2])/(end-start)
#            rad2=theta/180*np.pi
            slope_stack.append(slope)
            x = np.arange(0,len(ys))
            px = np.polyfit(x,ys,1)
            mmg_x_ploy=np.polyval(px,x)
            err=np.mean(np.abs(mmg_x_ploy-ys))
            print(j,'err=',np.mean(np.abs(mmg_x_ploy-ys)))
#            plt.plot(mmg_x_ploy)
#            plt.plot(ys)
#            plt.show()
            error.append(err)
                
        drift=0
        for i in range(len(self.eu.values)):
            for k in range(len(self.cum)):
                if i<self.cum[0,0]:
                    if slope_stack[0]<5:
                        slope=slope_stack[0]
                    elif slope_stack[1]<5:
                        slope=slope_stack[1]   
                    elif slope_stack[2]<5:
                        slope=slope_stack[2]  
                if i>self.cum[j,0]:
                    if slope_stack[j]<5:
                        if slope!=slope_stack[j]:
                        
                            slope=slope_stack[j]
                            print('change rate to',slope_stack[j])

#            print(slope,drift)
            drift+=slope
            tempq=Quaternion(self.qq.values[i])
#            q2=Quaternion(1,0,0,0).from_angle_axis(rad,0,0,1)
            q2=q2=Quaternion(1,0,0,0).rotateaxis(drift,[0,0,1])
#            tempq=tempq*q2
            tempq=q2*tempq
#            self.q.values[i]=tempq.elements
            self.qq.values[i]=tempq.__array__()
            
#            self.eu.values[i,0],self.eu.values[i,1],self.eu.values[i,2]=tempq.eu_angle
            self.eu.values[i,0],self.eu.values[i,1],self.eu.values[i,2]=tempq.to_euler_angles_by_wiki()

        self.eu.values=self.eu.values/np.pi*180
    
     
        
#    def to_qeb(self,start=20,end=40):#eath to body frame
##        esti_eu=np.zeros([len(self.q),3])
#        samplerate=int(self.samplerate)
#        rest=range(start*samplerate,end*samplerate)
##        self.q.values[:,[1,2,3]]=self.q.values[:,[2,3,1]]
##        self.q.values[:,[3]]=-self.q.values[:,[3]]
##        qbe=Quaternion(np.mean(self.q.values[rest,:],axis=0))
#        print('q mean',np.mean(self.q.values[start*samplerate:end*samplerate,:],axis=0))
#        qbe=Quaternion(np.mean(self.q.values[start*samplerate:end*samplerate,:],axis=0))
#        for i in range(len(self.q.values)):
#            tempq=(qbe.conj())*Quaternion(self.q.values[i,:])
##            tempq=tempq.normalized()
#            tempq=tempq.conj()
#            self.q.values[i,:]=tempq._get_q()
#            self.eu.values[i,0],self.eu.values[i,1],self.eu.values[i,2]=tempq.to_euler_angles_by_wiki()
#        self.eu.values=self.eu.values/np.pi*180
#    def to_nedenu(self):#eath to body frame
#        samplerate=int(self.samplerate)
#        self.q.values[i,:]=self.q.values[i,[0,2,1,3]]
#        self.q.values[i,3]=-self.q.values[i,3]
#        for i in range(len(self.q.values)):
#            tempq=Quaternion(self.q.values[i,:])
#            self.q.values[i,:]=tempq._get_q()
#            self.eu.values[i,0],self.eu.values[i,1],self.eu.values[i,2]=tempq.to_euler_angles_by_wiki()
#        self.eu.values=self.eu.values/np.pi*180
#    def to_qeb2(self,start=20,end=40):#eath to body frame
#        samplerate=int(self.samplerate)
#        rest=range(start*samplerate,end*samplerate)
##        self.q.values[:,[1,2,3]]=self.q.values[:,[2,3,1]]
##        self.q.values[:,[3]]=-self.q.values[:,[3]]
##        qbe=Quaternion(np.mean(self.q.values[rest,:],axis=0))
#        print('q mean',np.mean(self.q.values[start*samplerate:end*samplerate,:],axis=0))
#        base=np.mean(self.eu.values[start*samplerate:end*samplerate,:],axis=0)
#        self.eu.values-=base
#        self.eu.values[self.eu.values<-360]+=360
#        self.eu.values[self.eu.values>360]-=360
class Adp_ImgJ(Adaptor):
    def __init__(self,date,subject,session):
        super().__init__(date,subject,session)
        self.bias=[[12,19,17,57],[20,17,17,17]]
        if date=='0801':
            self.bias=self.bias[subject][session]
            self.body_table=['T7','T3','C3','Head','C7','LS','RS']
            self.str_path=[['./imgj/0101.csv','./imgj/0102.csv','./imgj/0103.csv','./imgj/0104.csv'],['./imgj/0201.csv','./imgj/0202.csv','./imgj/0103.csv','./imgj/0104.csv']]
            self.pathway=self.str_path[subject][session]
            self.head_angle,self.shoulder_angle=get_video_angle(self.pathway)
            self.head_angle,self.shoulder_angle=Signal(self.head_angle),Signal(self.shoulder_angle)
class Adp_Vicon(Adaptor):
    def __init__(self,date,subject,session,dev_name):
        super().__init__(date,subject,session)
        self.dev_name=dev_name
#        self.bias=[[40,50,52,47],[56,57,52,37]]
        self.samplerate=120
        if date=='0801':
            self.bias=[[56,50,52,47],[56,57,52,37]]
            self.body_table=['T7','T3','C3','Head','C7','LS','RS']
            self.str_path=[['./vicon/0101.csv','./vicon/0102.csv','./vicon/0103.csv','./vicon/0104.csv'],['./vicon/0201.csv','./vicon/0202.csv','./vicon/0103.csv','./vicon/0104.csv']]
            
            self.pathway=self.str_path[subject][session]
            self.dev_id=self.body_table.index(dev_name)
            self.bias=self.bias[subject][session]
            df = pd.read_csv(self.pathway,header=10)
            self.start=[0,0,178,190,0,160,169]
            vicon_angle=df.as_matrix(columns=df.columns[self.start[self.dev_id]:self.start[self.dev_id]+3])
            if not dev_name=='Head':
                vicon_angle=-vicon_angle
            ts=df.as_matrix(columns=df.columns[0:1])
            self.ts=Signal(ts*(1/120),samplerate=self.samplerate)
            self.vicon_angle=Signal(vicon_angle,samplerate=self.samplerate)
        if date=='0825':
            self.body_table=['dev_0','dev_1','dev_2','dev_3','dev_4','dev_5','dev_6','dev_7']
            self.scale_label=np.array([[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[0,105],[100,225],[225,-10]]
                            ,[[6,105],[100,221],[221,-1]]
                            ,[[0,100],[100,200],[200,-1]]
                            ,[[0,100],[100,220],[218,-1]]
                            ,[[0,84],[82,172],[195,-1]]
                            ,[[0,90],[90,160],[158,-1]]])*120
            self.bias=np.array([[7,5,2],
                               [13,0,12],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0]])*120
#                            ,[[0,-1],[3755,5600],[5600,-1]]])
#            self.bias=np.zeros(self.scale_label.shape)
            self.str_path=[['./vicon/0825/20180824-imu-0-roll01.csv','./vicon/0825/20180824-imu-0-pitch01.csv','./vicon/0825/20180824-imu-0-yaw01.csv',]
                ,['./vicon/0825/20180824-imu-1-roll02.csv','./vicon/0825/20180824-imu-1-pitch01.csv','./vicon/0825/20180824-imu-1-yaw01.csv',]
                ,['./vicon/0825/20180824-imu-2-roll_pitch_yaw01.csv','./vicon/0825/20180824-imu-2-roll_pitch_yaw01.csv','./vicon/0825/20180824-imu-2-roll_pitch_yaw01.csv',]
                ,['./vicon/0825/20180824-imu-3-roll_pitch_yaw01.csv','./vicon/0825/20180824-imu-3-roll_pitch_yaw01.csv','./vicon/0825/20180824-imu-3-roll_pitch_yaw01.csv',]
                ,['./vicon/0825/20180824-imu-4-roll_pitch_yaw01.csv','./vicon/0825/20180824-imu-4-roll_pitch_yaw01.csv','./vicon/0825/20180824-imu-4-roll_pitch_yaw01.csv',]
                ,['./vicon/0825/20180824-imu-5-roll_pitch_yaw01.csv','./vicon/0825/20180824-imu-5-roll_pitch_yaw01.csv','./vicon/0825/20180824-imu-5-roll_pitch_yaw01.csv',]
                ,['./vicon/0825/20180824-imu-6-roll_pitch_yaw01.csv','./vicon/0825/20180824-imu-6-roll_pitch_yaw01.csv','./vicon/0825/20180824-imu-6-roll_pitch_yaw01.csv',]
                ,['./vicon/0825/20180824-imu-7-roll_pitch_yaw01.csv','./vicon/0825/20180824-imu-7-roll_pitch_yaw01.csv','./vicon/0825/20180824-imu-7-roll_pitch_yaw01.csv',]]
#            self.str_path=[['./vicon/0101.csv','./vicon/0102.csv','./vicon/0103.csv','./vicon/0104.csv'],['./vicon/0201.csv','./vicon/0202.csv','./vicon/0103.csv','./vicon/0104.csv']]
            self.pathway=self.str_path[subject][session]
            df = pd.read_csv(self.pathway, index_col=False)
            df=df[20:-10]
            df=df[self.scale_label[subject][session][0]:self.scale_label[subject][session][1]]
#            print(self.bias[subject][session])
            df=df[self.bias[subject][session]:-1]
            ref_q=np.zeros([len(df.values[:,0]),4])
#            temp_mts=np.zeros(len(df.values[:,11]))
#            temp_ts=np.zeros(len(df.values[:,10]))
            
            
            ref_qq=df.as_matrix(columns=df.columns[6:10])
            temp_mts=df.as_matrix(columns=df.columns[11:12])
            temp_ts=df.as_matrix(columns=df.columns[10:11])
            ref_ts=np.copy(temp_ts)
            ref_ts[:]=temp_ts[:]+temp_mts[:]*0.001
            ref_eu=np.zeros([len(ref_q),3])
            ref_delta_t=np.zeros(len(df.values[:,11]))
            for i in range(len(temp_ts)-1):
                tempp=temp_ts[i+1]-temp_ts[i]
                if tempp>=0:
                    ref_delta_t[i]=tempp
                else:
                    ref_delta_t[i]=(61-temp_ts[i])+temp_ts[i+1]
            for i in range(2,len(ref_delta_t)):
                if ref_delta_t[i]>0.5:
                    ref_delta_t[i]=(ref_delta_t[i-1]+ref_delta_t[i-2])/2
            ref_ts=np.cumsum(ref_delta_t)


#            ref_qq=ref_qq[:,[0,1,3,2]]
#            ref_qq[:,[2,3]]=-ref_qq[:,[2,3]]
#            ref_qq=ref_qq[:,[0,3,2,1]]
#            ref_qq[:,[1,3]]=-ref_qq[:,[1,3]]
            for i in range(len(ref_qq)):
                tempq=Quaternion(ref_qq[i])
                ref_eu[i,0],ref_eu[i,1],ref_eu[i,2]=tempq.to_euler_angles_no_Gimbal()
#            print(ref_eu[0,:])
            
#            print(ref_eu[0,:])
            self.eu=Signal(ref_eu/np.pi*180,samplerate=self.samplerate)
            self.qq=Signal(ref_qq,samplerate=self.samplerate)
            self.ts=Signal(ref_ts,samplerate=self.samplerate)
class QuickAdp(IMU):
    def __init__(self,data,tablename='quick',xsens_smalerate=None,order=None):
        self.file_tag=tablename
        raw=data
        self.pathway='./db'
        self.ts=Signal(values=raw[:,0]/1000)
        if xsens_smalerate:
            l=len(raw[:,0])
            self.ts=Signal(np.linspace(0,(raw[-1,0]-raw[0,0])/40,l))
        self.delta_ts=self.ts.copy()    
        self.delta_ts.values-=self.delta_ts.values[0]
        self.delta_ts=self.delta_ts.diff()
        raw=raw[:,1:10]
        
#        self.gyr_order=np.array([1,-2,-3])
        if order is None:
            self.gyr_order=np.array([1,-2,-3])
        else:
            self.gyr_order=np.array(order)
        self.sampleperiod=(((self.ts.values)[-1]-(self.ts.values)[0])/len(self.ts.values))
        self.samplerate=(1/self.sampleperiod)
        self.acc=Signal(values=raw[:,0:3],tag=self.file_tag+'acc',samplerate=self.samplerate)
        self.gyr=Signal(values=raw[:,3:6],tag=self.file_tag+'gyr',samplerate=self.samplerate)
        self.mag=Signal(values=raw[:,6:9],tag=self.file_tag+'mag',samplerate=self.samplerate)

#        self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[self.dev_id])
        self.acc_order,self.gyr_order,self.mag_order=self.nedadj_mpu9250(self.gyr_order)

        self.acc=self.acc.changeorder(self.acc_order.tolist())
        self.gyr=self.gyr.changeorder(self.gyr_order.tolist())
        self.mag=self.mag.changeorder(self.mag_order.tolist())
        print('loading succeed')    
        self.dev_id=0
        self.body_table=['qucik']
        self.position='unknown'
        
  
class Adp_Xsens(Adaptor):
    def __init__(self,date,subject,session,dev_name):
        super().__init__(date,subject,session)
        self.dev_name=dev_name
#        self.bias=[[40,50,52,47],[56,57,52,37]]
        self.samplerate=100
        if subject==2:# or subject==1:
            self.samplerate=40
        if date=='1218':
            self.body_table=['dev01','dev_1','dev_2','dev_3','dev_4','dev_5','dev_6','dev_7']
            self.scale_label=np.array([[[0,35],[35,62],[62,-1]]
                            ,[[0,15],[15,28],[26,-1]]
                            ,[[0,33],[30,65],[59,-1]]
                            ,[[6,105],[100,221],[221,-1]]
                            ,[[0,100],[100,200],[200,-1]]
                            ,[[0,100],[100,220],[218,-1]]
                            ,[[0,84],[82,172],[195,-1]]
                            ,[[0,90],[90,160],[158,-1]]])*self.samplerate
            self.bias=np.array([[0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0]])*self.samplerate
#                            ,[[0,-1],[3755,5600],[5600,-1]]])
#            self.bias=np.zeros(self.scale_label.shape)
                
            self.str_path=[['./1207elipse_fitting/black_validation_xsens.csv','./1207elipse_fitting/black_validation_xsens.csv','./1207elipse_fitting/black_validation_xsens.csv',]
                ,['./1207elipse_fitting/subject2/black_validation_xsens.csv','./1207elipse_fitting/subject2/black_validation_xsens.csv','./1207elipse_fitting/subject2/black_validation_xsens.csv',]
                ,['./1207elipse_fitting/subject3/black_validation_xsens.csv','./1207elipse_fitting/subject3/black_validation_xsens.csv','./1207elipse_fitting/subject3/black_validation_xsens.csv',]]
            self.pathway=self.str_path[subject][session]
#            print(self.pathway)
            df = pd.read_csv(self.pathway, index_col=False)
            df=df[0:-1]
            df=df[self.scale_label[subject][session][0]:self.scale_label[subject][session][1]]
            df=df[self.bias[subject][session]:-1]
            ref_q=np.zeros([len(df.values[:,0]),4])
            
            
            ref_qq=df.as_matrix(columns=df.columns[43:47])
            ref_eu=np.zeros([len(ref_qq),3])
#            temp_mts=df.as_matrix(columns=df.columns[11:12])
            ref_ts=df.as_matrix(columns=df.columns[0:1])/self.samplerate
            print(ref_ts)
            ref_ts-=ref_ts[0]
            ref_ts=ref_ts.flatten()
#            ref_qq=ref_qq[:,[0,1,3,2]]
#            ref_qq[:,[2,3]]=-ref_qq[:,[2,3]]
#            ref_qq=ref_qq[:,[0,3,2,1]]
#            ref_qq[:,[1,3]]=-ref_qq[:,[1,3]]
            for i in range(len(ref_qq)):
                tempq=Quaternion(ref_qq[i])
#                ref_eu[i,0],ref_eu[i,1],ref_eu[i,2]=tempq.eu_angle
                ref_eu[i,0],ref_eu[i,1],ref_eu[i,2]=tempq.to_euler_angles_by_wiki()
            ref_eu=ref_eu/np.pi*180
                
            self.eu=Signal(ref_eu/np.pi*180,samplerate=self.samplerate)
            self.qq=Signal(ref_qq,samplerate=self.samplerate)
            self.ts=Signal(ref_ts,samplerate=self.samplerate)
    def lpf(self,l3db,h3db=None):
        self.qq.values,self.eu.values=all_filter(self.qq.values,self.eu.values,self.samplerate,l3db,h3db)

    def show_eu(self,index=[0,1,2]):
        eu_plt=Plot(title=str(self.dev_name)+'_eu',dtype='eu',index=index)
        eu_plt.plot_angle(self.ts,self.eu)
    def show_qq(self,index=[0,1,2]):
        eu_plt=Plot(title=str(self.dev_name)+'_qq',dtype='qq',index=index)
        eu_plt.plot_angle(self.ts,-self.qq)
#    def to_qeb(self,start=1,end=3):#eath to body frame
#        samplerate=int(self.samplerate)
#        rest=range(start*samplerate,end*samplerate)
#        qbe=Quaternion(np.mean(self.qq.values[rest,:],axis=0))
#        for i in range(len(self.qq.values)):
#            tempq=(qbe.conj())*Quaternion(self.qq.values[i,:])
#            self.qq.values[i,:]=tempq._get_q()
#            self.eu.values[i,0],self.eu.values[i,1],self.eu.values[i,2]=tempq.to_euler_angles_by_wiki()
#        self.eu.values=self.eu.values/np.pi*180
#        self.eu.values[:,:]= self.eu.values[:,[1,0,2]]
#        self.eu.values[:,[2]]= -self.eu.values[:,[2]]
class Adp_iphone(Adaptor):#0725
    def __init__(self,date,subject,session,dev_name,raw_or_cali):
        super().__init__(date,subject,session,raw_or_cali)
        
        if date=='220310':
            self.subject=subject
            self.session=session
            self.body_table=['3B1FA','3CDD4','735EC','97681','E0798','LS','RS']
            self.str_path=['/190709github_package_data/220310iphoneDATA/accIphone5.csv', # 2 circle talbe movement 
                            '/190709github_package_data/220310iphoneDATA/accIphone6.csv',#1 pedestrain movement
                            '/190709github_package_data/220310iphoneDATA/accIphone7.txt', #1 circle talbe movement
                            '/190709github_package_data/220310iphoneDATA/accIphone8.txt',#2 circle talbe movement
                            '/190709github_package_data/210913devices1minStatic/KaptureE0798_SixAxis_210913.csv',
                            ]
            pathway=Reader.get_UpperPath('.')+self.str_path[dev_name]
            print(pathway)
            Reader.figure(path=pathway)
            raw=Reader.export()
            print('adp_initialing..')
            print('loading..data')
            self.file_tag='wistron_'
            self.dev_name=dev_name
            self.gyr=raw[:,1:4]
            self.acc=raw[:,1:4]
            
            self.mag=self.gyr
            
            self.ts=np.linspace(0,1/17,self.gyr.shape[0])
            # self.gyr_order=np.array([[3,-2,1]])
    
            # self.gyr_order=self.gyr_order[:,[2,0,1]]
            self.gyr_order=np.array([[2,1,-3]])
            self.dev_id=dev_name
            rate=self.gyr.shape[0]/60
            self.delta_ts=Signal(np.full((len(self.ts),1),1/rate).flatten())
            self.ts=Signal(np.cumsum(self.delta_ts.values))
            self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[0])
            self.sampleperiod=(self.ts.values[-1]-self.ts.values[0])/len(self.ts.values)
            self.samplerate=1/self.sampleperiod
            self.acc=Signal(values=raw[:,1:4],tag=self.file_tag+'acc',samplerate=self.samplerate)
            self.gyr=Signal(values=raw[:,1:4],tag=self.file_tag+'gyr',samplerate=self.samplerate)
            self.mag=self.gyr
            self.accMag=np.zeros([self.acc.length,1])
            for i in range(self.acc.length):
                self.accMag[i,0]=(self.acc.values[i,0]**2+self.acc.values[i,1]**2+(self.acc.values[i,2])**2)**0.5
#
            self.accMag=Signal(self.accMag)
class Adp_ngimu(Adaptor):#0725
    def __init__(self,date,subject,session,dev_name,raw_or_cali,cali_path=''):
        super().__init__(date,subject,session,raw_or_cali)
        self.cali_path=cali_path
        if date=='220329':
            self.subject=subject
            self.session=session
            self.body_table=['ngimu','3CDD4','735EC','97681','E0798','LS','RS']
            self.str_path=['/190709github_package_data/220329ngimudata/220329ngimudata/walkaround/NGIMU - 0029C6FC/sensors.csv',
                        '/190709github_package_data/220329ngimudata/220329ngimudata/walkaround2/NGIMU - 0029C6FC/sensors.csv',
                        '/190709github_package_data/220329ngimudata/220329ngimudata/walkaround3/NGIMU - 0029C6FC/sensors.csv',
                        '/190709github_package_data/220329ngimudata/220329ngimudata/walkaround4/NGIMU - 0029C6FC/sensors.csv',
                        '/190709github_package_data/220329ngimudata/220329ngimudata/magcalibrate/NGIMU - 0029C6FC/sensors.csv',
                        
                            ]
            
            self.str_path2=['/190709github_package_data/220329ngimudata/220329ngimudata/walkaround/NGIMU - 0029C6FC/quaternion.csv',
                            '/190709github_package_data/220329ngimudata/220329ngimudata/walkaround2/NGIMU - 0029C6FC/quaternion.csv',
                    '/190709github_package_data/220329ngimudata/220329ngimudata/walkaround3/NGIMU - 0029C6FC/quaternion.csv',
                    '/190709github_package_data/220329ngimudata/220329ngimudata/walkaround4/NGIMU - 0029C6FC/quaternion.csv',
                    '/190709github_package_data/220329ngimudata/220329ngimudata/magcalibrate/NGIMU - 0029C6FC/quaternion.csv',
                            ]
            pathway=Reader.get_UpperPath('.')+self.str_path[dev_name]
            print(pathway)
            Reader.figure(path=pathway,header=0)
            raw=Reader.export()
            print('adp_initialing..')
            print('loading..data')
            self.file_tag='wistron_'
            self.dev_name=dev_name
            self.gyr=raw[:,3:6]
            self.acc=raw[:,0:3]
            self.mag=self.gyr
            self.version=self.date
            self.ts=np.linspace(0,60,self.gyr.shape[0])
            self.dev_id='ngimu'
            pathway2=Reader.get_UpperPath('.')+self.str_path2[dev_name]
            print(pathway2)
            Reader.figure(path=pathway2,header=0)
            raw2=Reader.export()
            # self.gyr_order=np.array([[3,-2,1]])
    
            # self.gyr_order=self.gyr_order[:,[2,0,1]]
            self.gyr_order=np.array([[2,1,-3]])
            self.dev_id=dev_name
            rate=self.gyr.shape[0]/60
            self.delta_ts=Signal(np.full((len(self.ts),1),1/rate).flatten())
            self.ts=Signal(np.cumsum(self.delta_ts.values))
            self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[0])
            self.sampleperiod=(self.ts.values[-1]-self.ts.values[0])/len(self.ts.values)
            self.samplerate=1/self.sampleperiod
            self.acc=Signal(values=raw[:,4:7],tag=self.file_tag+'acc',samplerate=self.samplerate)
            self.gyr=Signal(values=raw[:,1:4],tag=self.file_tag+'gyr',samplerate=self.samplerate)
            self.mag=Signal(values=raw[:,7:10],tag=self.file_tag+'mag',samplerate=self.samplerate)
            self.accMag=np.zeros([self.acc.length,1])
            for i in range(self.acc.length):
                self.accMag[i,0]=(self.acc.values[i,0]**2+self.acc.values[i,1]**2+(self.acc.values[i,2])**2)**0.5
            self.accMag=Signal(self.accMag,tag=self.file_tag+'accMag',samplerate=self.samplerate)
            self.quaternion=Signal(values=raw2[:,1:5],tag=self.file_tag+'quaternion',samplerate=self.samplerate)
            
            Signal(values=raw[:,6:9],tag=self.file_tag+'mag',samplerate=self.samplerate)
        if date=='220330':
                self.subject=subject
                self.session=session
                self.body_table=['ngimu','3CDD4','735EC','97681','E0798','LS','RS']
                self.str_path=['/190709github_package_data/220330ngimudata/round5/NGIMU - 0029C6FC/sensors.csv',
                            '/190709github_package_data/220330ngimudata/round6/NGIMU - 0029C6FC/sensors.csv',
                            '/190709github_package_data/220330ngimudata/round7/NGIMU - 0029C6FC/sensors.csv',
                            '/190709github_package_data/220329ngimudata/220330ngimudata/round5/NGIMU - 0029C6FC/sensors.csv',
                            '/190709github_package_data/220329ngimudata/220329ngimudata/magcalibrate/NGIMU - 0029C6FC/sensors.csv',
                            
                                ]
                
                self.str_path2=['/190709github_package_data/220330ngimudata/round5/NGIMU - 0029C6FC/quaternion.csv',
                                '/190709github_package_data/220330ngimudata/round6/NGIMU - 0029C6FC/quaternion.csv',
                        '/190709github_package_data/220330ngimudata/round7/NGIMU - 0029C6FC/quaternion.csv',
                        '/190709github_package_data/220329ngimudata/220329ngimudata/walkaround4/NGIMU - 0029C6FC/quaternion.csv',
                        '/190709github_package_data/220329ngimudata/220329ngimudata/magcalibrate/NGIMU - 0029C6FC/quaternion.csv',
                                ]
                pathway=Reader.get_UpperPath('.')+self.str_path[dev_name]
                print(pathway)
                Reader.figure(path=pathway,header=0)
                raw=Reader.export()
                print('adp_initialing..')
                print('loading..data')
                self.file_tag='wistron_'
                self.dev_name=dev_name
                self.gyr=raw[:,3:6]
                self.acc=raw[:,0:3]
                self.mag=self.gyr
                self.version=self.date
                self.ts=np.linspace(0,60,self.gyr.shape[0])
                self.dev_id='ngimu'
                pathway2=Reader.get_UpperPath('.')+self.str_path2[dev_name]
                print(pathway2)
                Reader.figure(path=pathway2,header=0)
                raw2=Reader.export()
                # self.gyr_order=np.array([[3,-2,1]])
        
                # self.gyr_order=self.gyr_order[:,[2,0,1]]
                self.gyr_order=np.array([[2,1,-3]])
                self.dev_id=dev_name
                rate=self.gyr.shape[0]/60
                self.delta_ts=Signal(np.full((len(self.ts),1),1/rate).flatten())
                self.ts=Signal(np.cumsum(self.delta_ts.values))
                self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[0])
                self.sampleperiod=(self.ts.values[-1]-self.ts.values[0])/len(self.ts.values)
                self.samplerate=1/self.sampleperiod
                self.acc=Signal(values=raw[:,4:7],tag=self.file_tag+'acc',samplerate=self.samplerate)
                self.gyr=Signal(values=raw[:,1:4],tag=self.file_tag+'gyr',samplerate=self.samplerate)
                self.mag=Signal(values=raw[:,7:10],tag=self.file_tag+'mag',samplerate=self.samplerate)
                self.accMag=np.zeros([self.acc.length,1])
                for i in range(self.acc.length):
                    self.accMag[i,0]=(self.acc.values[i,0]**2+self.acc.values[i,1]**2+(self.acc.values[i,2])**2)**0.5
                self.accMag=Signal(self.accMag,tag=self.file_tag+'accMag',samplerate=self.samplerate)
                self.quaternion=Signal(values=raw2[:,1:5],tag=self.file_tag+'quaternion',samplerate=self.samplerate)
                
                Signal(values=raw[:,6:9],tag=self.file_tag+'mag',samplerate=self.samplerate)
        if date=='220420':
                self.subject=subject
                self.session=session
                self.body_table=['ngimu','3CDD4','735EC','97681','E0798','LS','RS']
                self.str_path=['/220427python/190910github_package/sensor_data.csv',
                            '/190709github_package_data/220330ngimudata/round6/NGIMU - 0029C6FC/sensors.csv',
                            '/190709github_package_data/220330ngimudata/round7/NGIMU - 0029C6FC/sensors.csv',
                            '/190709github_package_data/220329ngimudata/220330ngimudata/round5/NGIMU - 0029C6FC/sensors.csv',
                            '/190709github_package_data/220329ngimudata/220329ngimudata/magcalibrate/NGIMU - 0029C6FC/sensors.csv',
                            
                                ]
                
                self.str_path2=['/190709github_package_data/220330ngimudata/round5/NGIMU - 0029C6FC/quaternion.csv',
                                '/190709github_package_data/220330ngimudata/round6/NGIMU - 0029C6FC/quaternion.csv',
                        '/190709github_package_data/220330ngimudata/round7/NGIMU - 0029C6FC/quaternion.csv',
                        '/190709github_package_data/220329ngimudata/220329ngimudata/walkaround4/NGIMU - 0029C6FC/quaternion.csv',
                        '/190709github_package_data/220329ngimudata/220329ngimudata/magcalibrate/NGIMU - 0029C6FC/quaternion.csv',
                                ]
                pathway=Reader.get_UpperPath('.')+self.str_path[dev_name]
                print(pathway)
                Reader.figure(path=pathway,header=0)
                raw=Reader.export()
                print('adp_initialing..')
                print('loading..data')
                self.file_tag='wistron_'
                self.dev_name=dev_name
                self.gyr=raw[:,3:6]
                self.acc=raw[:,0:3]
                self.mag=self.gyr
                self.version=self.date
                self.ts=np.linspace(0,60,self.gyr.shape[0])
                self.dev_id='ngimu'
                pathway2=Reader.get_UpperPath('.')+self.str_path2[dev_name]
                print(pathway2)
                Reader.figure(path=pathway2,header=0)
                raw2=Reader.export()
                # self.gyr_order=np.array([[3,-2,1]])
        
                # self.gyr_order=self.gyr_order[:,[2,0,1]]
                self.gyr_order=np.array([[2,1,-3]])
                self.dev_id=dev_name
                rate=self.gyr.shape[0]/60
                self.delta_ts=Signal(np.full((len(self.ts),1),1/rate).flatten())
                self.ts=Signal(np.cumsum(self.delta_ts.values))
                self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[0])
                self.sampleperiod=(self.ts.values[-1]-self.ts.values[0])/len(self.ts.values)
                self.samplerate=1/self.sampleperiod
                self.acc=Signal(values=raw[:,4:7],tag=self.file_tag+'acc',samplerate=self.samplerate)
                self.gyr=Signal(values=raw[:,1:4],tag=self.file_tag+'gyr',samplerate=self.samplerate)
                self.mag=Signal(values=raw[:,7:10],tag=self.file_tag+'mag',samplerate=self.samplerate)
                self.accMag=np.zeros([self.acc.length,1])
                for i in range(self.acc.length):
                    self.accMag[i,0]=(self.acc.values[i,0]**2+self.acc.values[i,1]**2+(self.acc.values[i,2])**2)**0.5
                self.accMag=Signal(self.accMag,tag=self.file_tag+'accMag',samplerate=self.samplerate)
                self.quaternion=Signal(values=raw2[:,1:5],tag=self.file_tag+'quaternion',samplerate=self.samplerate)
                
                Signal(values=raw[:,6:9],tag=self.file_tag+'mag',samplerate=self.samplerate)
class Adp_wistron(Adaptor):#0725
    def __init__(self,date,subject,session,dev_name,raw_or_cali):
        super().__init__(date,subject,session,raw_or_cali)
        
        if date=='210913':
            self.subject=subject
            self.session=session
            
            self.body_table=['iphone','3CDD4','735EC','97681','E0798','LS','RS']
            self.str_path=['/190709github_package_data/210913devices1minStatic/Kapture3B1FA_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture3CDD4_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture735EC_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture97681_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/KaptureE0798_SixAxis_210913.csv',
                            ]
            pathway=Reader.get_UpperPath('.')+self.str_path[dev_name]
            print(pathway)
            Reader.figure(path=pathway,header=0)
            raw=Reader.export()

            print('adp_initialing..')
            print('loading..data')
            self.file_tag='wistron_'
            self.dev_name=dev_name
            self.gyr=raw[:,3:6]
            self.acc=raw[:,0:3]
            self.mag=self.gyr
            self.ts=np.linspace(0,60,self.gyr.shape[0])
            # self.gyr_order=np.array([[3,-2,1]])
    
            # self.gyr_order=self.gyr_order[:,[2,0,1]]
            self.gyr_order=np.array([[2,1,-3]])
            self.dev_id=dev_name
            rate=self.gyr.shape[0]/60
            self.delta_ts=Signal(np.full((len(self.ts),1),1/rate).flatten())
            self.ts=Signal(np.cumsum(self.delta_ts.values))
            self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[0])
            self.sampleperiod=(self.ts.values[-1]-self.ts.values[0])/len(self.ts.values)
            self.samplerate=1/self.sampleperiod
            self.acc=Signal(values=raw[:,0:3],tag=self.file_tag+'acc',samplerate=self.samplerate)
            self.gyr=Signal(values=raw[:,3:6],tag=self.file_tag+'gyr',samplerate=self.samplerate)
            self.mag=Signal(values=raw[:,6:9],tag=self.file_tag+'mag',samplerate=self.samplerate)
            self.quaternion=Signal(values=raw2[:,1:5],tag=self.file_tag+'quaternion',samplerate=self.samplerate)
            self.mag=self.gyr
            self.accMag=np.zeros([self.acc.length,1])
            for i in range(self.acc.length):
                self.accMag[i,0]=(self.acc.values[i,0]**2+self.acc.values[i,1]**2+(self.acc.values[i,2])**2)**0.5

            Signal(values=raw[:,6:9],tag=self.file_tag+'mag',samplerate=self.samplerate)
        elif date=='220310':
            self.subject=subject
            self.session=session
            self.body_table=['3B1FA','3CDD4','735EC','97681','E0798','LS','RS']
            self.str_path=['/190709github_package_data/220310kapture/220310kaptureMotion/Kapture42A7F_SixAxis_220310.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture3CDD4_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture735EC_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture97681_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/KaptureE0798_SixAxis_210913.csv',
                            ]
            pathway=Reader.get_UpperPath('.')+self.str_path[dev_name]
            print(pathway)
            Reader.figure(path=pathway)
            raw=Reader.export()
            print('adp_initialing..')
            print('loading..data')
            self.file_tag='wistron_'
            self.dev_name=dev_name
            self.gyr=raw[:,4:7]
            self.acc=raw[:,1:4]
            
            self.mag=self.gyr
            
            self.ts=np.linspace(0,1/17,self.gyr.shape[0])
            # self.gyr_order=np.array([[3,-2,1]])
    
            # self.gyr_order=self.gyr_order[:,[2,0,1]]
            self.gyr_order=np.array([[2,1,-3]])
            self.dev_id=dev_name
            rate=self.gyr.shape[0]/60
            self.delta_ts=Signal(np.full((len(self.ts),1),1/rate).flatten())
            self.ts=Signal(np.cumsum(self.delta_ts.values))
            self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[0])
            self.sampleperiod=(self.ts.values[-1]-self.ts.values[0])/len(self.ts.values)
            self.samplerate=1/self.sampleperiod
            self.acc=Signal(values=raw[:,1:4],tag=self.file_tag+'acc',samplerate=self.samplerate)
            self.gyr=Signal(values=raw[:,4:7],tag=self.file_tag+'gyr',samplerate=self.samplerate)
            self.mag=self.gyr
            self.accMag=np.zeros([self.acc.length,1])
            for i in range(self.acc.length):
                self.accMag[i,0]=(self.acc.values[i,0]**2+self.acc.values[i,1]**2+(self.acc.values[i,2])**2)**0.5

            Signal(values=raw[:,6:9],tag=self.file_tag+'mag',samplerate=self.samplerate)
        elif date=='220328':
            self.subject=subject
            self.session=session
            self.body_table=['C707D','3CDD4','735EC','97681','E0798','LS','RS']
            self.str_path=['/190709github_package_data/220328data/220328data/circle/KaptureC707D_SixAxis_220328.csv',
                           '/190709github_package_data/220328data/220328data/circle2/KaptureC707D_SixAxis_220328.csv',
                           '/190709github_package_data/220328data/220328data/circle3/KaptureC707D_SixAxis_220328.csv',
                           '/190709github_package_data/220328data/220328data/straight/KaptureC707D_SixAxis_220328.csv',
                            ]
            pathway=Reader.get_UpperPath('.')+self.str_path[dev_name]
            print(pathway)
            Reader.figure(path=pathway)
            raw=Reader.export()
            print('adp_initialing..')
            print('loading..data')
            self.file_tag='wistron_'
            self.dev_name=dev_name
            self.gyr=raw[:,4:7]
            self.acc=raw[:,1:4]
            
            self.mag=self.gyr
            
            self.ts=np.linspace(0,1/17,self.gyr.shape[0])
            # self.gyr_order=np.array([[3,-2,1]])
    
            # self.gyr_order=self.gyr_order[:,[2,0,1]]
            self.gyr_order=np.array([[2,1,-3]])
            self.dev_id=dev_name
            rate=self.gyr.shape[0]/60
            self.delta_ts=Signal(np.full((len(self.ts),1),1/rate).flatten())
            self.ts=Signal(np.cumsum(self.delta_ts.values))
            self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[0])
            self.sampleperiod=(self.ts.values[-1]-self.ts.values[0])/len(self.ts.values)
            self.samplerate=1/self.sampleperiod
            self.acc=Signal(values=raw[:,1:4],tag=self.file_tag+'acc',samplerate=self.samplerate)
            self.gyr=Signal(values=raw[:,4:7],tag=self.file_tag+'gyr',samplerate=self.samplerate)
            self.quaternion=Signal(values=raw[:,7:12],tag=self.file_tag+'quaternion',samplerate=self.samplerate)
            self.mag=self.gyr
            self.accMag=np.zeros([self.acc.length,1])
            for i in range(self.acc.length):
                self.accMag[i,0]=(self.acc.values[i,0]**2+self.acc.values[i,1]**2+(self.acc.values[i,2])**2)**0.5

            Signal(values=raw[:,6:9],tag=self.file_tag+'mag',samplerate=self.samplerate)
        elif date=='220308':
            self.subject=subject
            self.session=session
            self.body_table=['3B1FA','3CDD4','735EC','97681','E0798','LS','RS']
            self.str_path=['/190709github_package_data/220308positionTrack/Kapture42A7F_SixAxis_220308.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture3CDD4_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture735EC_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture97681_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/KaptureE0798_SixAxis_210913.csv',
                            ]
            pathway=Reader.get_UpperPath('.')+self.str_path[dev_name]
            print(pathway)
            Reader.figure(path=pathway)
            raw=Reader.export()
            print('adp_initialing..')
            print('loading..data')
            self.file_tag='wistron_'
            self.dev_name=dev_name
            self.gyr=raw[:,4:7]
            self.acc=raw[:,1:4]
            
            self.mag=self.gyr
            
            self.ts=np.linspace(0,1/17,self.gyr.shape[0])
            # self.gyr_order=np.array([[3,-2,1]])
    
            # self.gyr_order=self.gyr_order[:,[2,0,1]]
            self.gyr_order=np.array([[2,1,-3]])
            self.dev_id=dev_name
            rate=self.gyr.shape[0]/60
            self.delta_ts=Signal(np.full((len(self.ts),1),1/rate).flatten())
            self.ts=Signal(np.cumsum(self.delta_ts.values))
            self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[0])
            self.sampleperiod=(self.ts.values[-1]-self.ts.values[0])/len(self.ts.values)
            self.samplerate=1/self.sampleperiod
            self.acc=Signal(values=raw[:,1:4],tag=self.file_tag+'acc',samplerate=self.samplerate)
            self.gyr=Signal(values=raw[:,4:7],tag=self.file_tag+'gyr',samplerate=self.samplerate)
            self.mag=self.gyr
            self.accMag=np.zeros([self.acc.length,1])
            for i in range(self.acc.length):
                self.accMag[i,0]=(self.acc.values[i,0]**2+self.acc.values[i,1]**2+(self.acc.values[i,2])**2)**0.5

            Signal(values=raw[:,6:9],tag=self.file_tag+'mag',samplerate=self.samplerate)
        elif date=='190709':
            
            self.subject=subject
            self.session=session
            # self.body_table=['735EC','735EC','735EC','97681','E0798','LS','RS']
            self.body_table=['735EC','E0798','C3','Head','C7','LS','RS']
            self.str_path=['/190709github_package_data/210910wistron1min/Kapture735EC_SixAxis_210910.csv',
                            '/190709github_package_data/210910wistron1min/KaptureE0798_SixAxis_210910.csv',
                            ]
            pathway=Reader.get_UpperPath('.')+self.str_path[dev_name]
            print(pathway)
            Reader.figure(path=pathway)
            raw=Reader.export()

            print('adp_initialing..')
            print('loading..data')
            self.file_tag='wistron_'
            self.dev_name=dev_name
            self.gyr=raw[:,3:6]
            self.acc=raw[:,0:3]
            self.mag=self.gyr
            self.ts=np.linspace(0,60,self.gyr.shape[0])
            # self.gyr_order=np.array([[3,-2,1]])
    
            # self.gyr_order=self.gyr_order[:,[2,0,1]]
            self.gyr_order=np.array([[2,1,-3]])
            self.dev_id=dev_name
            rate=self.gyr.shape[0]/60
            self.delta_ts=Signal(np.full((len(self.ts),1),1/rate).flatten())
            self.ts=Signal(np.cumsum(self.delta_ts.values))
            self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[0])
            self.sampleperiod=(self.ts.values[-1]-self.ts.values[0])/len(self.ts.values)
            self.samplerate=1/self.sampleperiod
            self.acc=Signal(values=raw[:,0:3],tag=self.file_tag+'acc',samplerate=self.samplerate)
            self.gyr=Signal(values=raw[:,3:6],tag=self.file_tag+'gyr',samplerate=self.samplerate)
            self.mag=Signal(values=raw[:,6:9],tag=self.file_tag+'mag',samplerate=self.samplerate)
class Adp_wistron2(Adaptor):#0725
    def __init__(self,date,subject,session,dev_name,raw_or_cali):
        super().__init__(date,subject,session,raw_or_cali)
        
        if date=='210913':
            self.subject=subject
            self.session=session
            
            self.body_table=['iphone','3CDD4','735EC','97681','E0798','LS','RS']
            self.str_path=['/190709github_package_data/210913devices1minStatic/Kapture3B1FA_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture3CDD4_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture735EC_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture97681_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/KaptureE0798_SixAxis_210913.csv',
                            ]
            pathway=Reader.get_UpperPath('.')+self.str_path[dev_name]
            print(pathway)
            Reader.figure(path=pathway,header=0)
            raw=Reader.export()

            print('adp_initialing..')
            print('loading..data')
            self.file_tag='wistron_'
            self.dev_name=dev_name
            self.gyr=raw[:,3:6]
            self.acc=raw[:,0:3]
            self.mag=self.gyr
            self.ts=np.linspace(0,60,self.gyr.shape[0])
            # self.gyr_order=np.array([[3,-2,1]])
    
            # self.gyr_order=self.gyr_order[:,[2,0,1]]
            self.gyr_order=np.array([[2,1,-3]])
            self.dev_id=dev_name
            rate=self.gyr.shape[0]/60
            self.delta_ts=Signal(np.full((len(self.ts),1),1/rate).flatten())
            self.ts=Signal(np.cumsum(self.delta_ts.values))
            self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[0])
            self.sampleperiod=(self.ts.values[-1]-self.ts.values[0])/len(self.ts.values)
            self.samplerate=1/self.sampleperiod
            self.acc=Signal(values=raw[:,0:3],tag=self.file_tag+'acc',samplerate=self.samplerate)
            self.gyr=Signal(values=raw[:,3:6],tag=self.file_tag+'gyr',samplerate=self.samplerate)
            self.mag=Signal(values=raw[:,6:9],tag=self.file_tag+'mag',samplerate=self.samplerate)
            self.quaternion=Signal(values=raw2[:,1:5],tag=self.file_tag+'quaternion',samplerate=self.samplerate)
            self.mag=self.gyr
            self.accMag=np.zeros([self.acc.length,1])
            for i in range(self.acc.length):
                self.accMag[i,0]=(self.acc.values[i,0]**2+self.acc.values[i,1]**2+(self.acc.values[i,2])**2)**0.5

            Signal(values=raw[:,6:9],tag=self.file_tag+'mag',samplerate=self.samplerate)
        elif date=='220310':
            self.subject=subject
            self.session=session
            self.body_table=['3B1FA','3CDD4','735EC','97681','E0798','LS','RS']
            self.str_path=['/190709github_package_data/220310kapture/220310kaptureMotion/Kapture42A7F_SixAxis_220310.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture3CDD4_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture735EC_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture97681_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/KaptureE0798_SixAxis_210913.csv',
                            ]
            pathway=Reader.get_UpperPath('.')+self.str_path[dev_name]
            print(pathway)
            Reader.figure(path=pathway)
            raw=Reader.export()
            print('adp_initialing..')
            print('loading..data')
            self.file_tag='wistron_'
            self.dev_name=dev_name
            self.gyr=raw[:,4:7]
            self.acc=raw[:,1:4]
            
            self.mag=self.gyr
            
            self.ts=np.linspace(0,1/17,self.gyr.shape[0])
            # self.gyr_order=np.array([[3,-2,1]])
    
            # self.gyr_order=self.gyr_order[:,[2,0,1]]
            self.gyr_order=np.array([[2,1,-3]])
            self.dev_id=dev_name
            rate=self.gyr.shape[0]/60
            self.delta_ts=Signal(np.full((len(self.ts),1),1/rate).flatten())
            self.ts=Signal(np.cumsum(self.delta_ts.values))
            self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[0])
            self.sampleperiod=(self.ts.values[-1]-self.ts.values[0])/len(self.ts.values)
            self.samplerate=1/self.sampleperiod
            self.acc=Signal(values=raw[:,1:4],tag=self.file_tag+'acc',samplerate=self.samplerate)
            self.gyr=Signal(values=raw[:,4:7],tag=self.file_tag+'gyr',samplerate=self.samplerate)
            self.mag=self.gyr
            self.accMag=np.zeros([self.acc.length,1])
            for i in range(self.acc.length):
                self.accMag[i,0]=(self.acc.values[i,0]**2+self.acc.values[i,1]**2+(self.acc.values[i,2])**2)**0.5

            Signal(values=raw[:,6:9],tag=self.file_tag+'mag',samplerate=self.samplerate)
        elif date=='220328':
            self.subject=subject
            self.session=session
            self.body_table=['C707D','3CDD4','735EC','97681','E0798','LS','RS']
            self.str_path=['/190709github_package_data/220328data/220328data/circle/KaptureC707D_SixAxis_220328.csv',
                           '/190709github_package_data/220328data/220328data/circle2/KaptureC707D_SixAxis_220328.csv',
                           '/190709github_package_data/220328data/220328data/circle3/KaptureC707D_SixAxis_220328.csv',
                           '/190709github_package_data/220328data/220328data/straight/KaptureC707D_SixAxis_220328.csv',
                            ]
            pathway=Reader.get_UpperPath('.')+self.str_path[dev_name]
            print(pathway)
            Reader.figure(path=pathway)
            raw=Reader.export()
            print('adp_initialing..')
            print('loading..data')
            self.file_tag='wistron_'
            self.dev_name=dev_name
            self.gyr=raw[:,4:7]
            self.acc=raw[:,1:4]
            
            self.mag=self.gyr
            
            self.ts=np.linspace(0,1/17,self.gyr.shape[0])
            # self.gyr_order=np.array([[3,-2,1]])
    
            # self.gyr_order=self.gyr_order[:,[2,0,1]]
            self.gyr_order=np.array([[2,1,-3]])
            self.dev_id=dev_name
            rate=self.gyr.shape[0]/60
            self.delta_ts=Signal(np.full((len(self.ts),1),1/rate).flatten())
            self.ts=Signal(np.cumsum(self.delta_ts.values))
            self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[0])
            self.sampleperiod=(self.ts.values[-1]-self.ts.values[0])/len(self.ts.values)
            self.samplerate=1/self.sampleperiod
            self.acc=Signal(values=raw[:,1:4],tag=self.file_tag+'acc',samplerate=self.samplerate)
            self.gyr=Signal(values=raw[:,4:7],tag=self.file_tag+'gyr',samplerate=self.samplerate)
            self.quaternion=Signal(values=raw[:,7:12],tag=self.file_tag+'quaternion',samplerate=self.samplerate)
            self.mag=self.gyr
            self.accMag=np.zeros([self.acc.length,1])
            for i in range(self.acc.length):
                self.accMag[i,0]=(self.acc.values[i,0]**2+self.acc.values[i,1]**2+(self.acc.values[i,2])**2)**0.5

            Signal(values=raw[:,6:9],tag=self.file_tag+'mag',samplerate=self.samplerate)
        elif date=='220426':
            self.subject=subject
            self.session=session
            self.body_table=['787EC','3CDD4','735EC','97681','E0798','LS','RS']
            self.str_path=['/190709github_package_data/Kapture787EC_17-49-35.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture3CDD4_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture735EC_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/Kapture97681_SixAxis_210913.csv',
                            '/190709github_package_data/210913devices1minStatic/KaptureE0798_SixAxis_210913.csv',
                            ]
            pathway=Reader.get_UpperPath('.')+self.str_path[dev_name]
            print(pathway)
            Reader.figure(path=pathway)
            raw=Reader.export()
            print('adp_initialing..')
            print('loading..data')
            self.file_tag='wistron_'
            self.dev_name=dev_name
            self.gyr=raw[:,4:7]
            self.acc=raw[:,1:4]
            
            self.mag=self.gyr
            
            self.ts=np.linspace(0,1/17,self.gyr.shape[0])
            # self.gyr_order=np.array([[3,-2,1]])
    
            # self.gyr_order=self.gyr_order[:,[2,0,1]]
            self.gyr_order=np.array([[2,1,-3]])
            self.dev_id=dev_name
            rate=self.gyr.shape[0]/60
            self.delta_ts=Signal(np.full((len(self.ts),1),1/rate).flatten())
            self.ts=Signal(np.cumsum(self.delta_ts.values))
            self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[0])
            self.sampleperiod=(self.ts.values[-1]-self.ts.values[0])/len(self.ts.values)
            self.samplerate=1/self.sampleperiod
            self.acc=Signal(values=raw[:,1:4],tag=self.file_tag+'acc',samplerate=self.samplerate)
            self.gyr=Signal(values=raw[:,4:7],tag=self.file_tag+'gyr',samplerate=self.samplerate)
            self.mag=self.gyr
            self.accMag=np.zeros([self.acc.length,1])
            for i in range(self.acc.length):
                self.accMag[i,0]=(self.acc.values[i,0]**2+self.acc.values[i,1]**2+(self.acc.values[i,2])**2)**0.5

            Signal(values=raw[:,6:9],tag=self.file_tag+'mag',samplerate=self.samplerate)
        elif date=='190709':
            
            self.subject=subject
            self.session=session
            # self.body_table=['735EC','735EC','735EC','97681','E0798','LS','RS']
            self.body_table=['735EC','E0798','C3','Head','C7','LS','RS']
            self.str_path=['/190709github_package_data/210910wistron1min/Kapture735EC_SixAxis_210910.csv',
                            '/190709github_package_data/210910wistron1min/KaptureE0798_SixAxis_210910.csv',
                            ]
            pathway=Reader.get_UpperPath('.')+self.str_path[dev_name]
            print(pathway)
            Reader.figure(path=pathway)
            raw=Reader.export()

            print('adp_initialing..')
            print('loading..data')
            self.file_tag='wistron_'
            self.dev_name=dev_name
            self.gyr=raw[:,3:6]
            self.acc=raw[:,0:3]
            self.mag=self.gyr
            self.ts=np.linspace(0,60,self.gyr.shape[0])
            # self.gyr_order=np.array([[3,-2,1]])
    
            # self.gyr_order=self.gyr_order[:,[2,0,1]]
            self.gyr_order=np.array([[2,1,-3]])
            self.dev_id=dev_name
            rate=self.gyr.shape[0]/60
            self.delta_ts=Signal(np.full((len(self.ts),1),1/rate).flatten())
            self.ts=Signal(np.cumsum(self.delta_ts.values))
            self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[0])
            self.sampleperiod=(self.ts.values[-1]-self.ts.values[0])/len(self.ts.values)
            self.samplerate=1/self.sampleperiod
            self.acc=Signal(values=raw[:,0:3],tag=self.file_tag+'acc',samplerate=self.samplerate)
            self.gyr=Signal(values=raw[:,3:6],tag=self.file_tag+'gyr',samplerate=self.samplerate)
            self.mag=Signal(values=raw[:,6:9],tag=self.file_tag+'mag',samplerate=self.samplerate)

class Adp_aiq(Adaptor):#0725
    #俊逸3次，柏伸3次

    def __init__(self,date,subject,session,dev_name,raw_or_cali='raw',fulltag='true',testmode=False,headbias=None,tablename=None,raw_path=None,cali_path=None):
        self.bias=np.zeros([8,3],dtype=int)
        super().__init__(date,subject,session,raw_or_cali)
        print('adp_initialing..')
        print('loading..data')
        self.acctag=['acc','a_x','a_y','a_z']
        self.gyrtag=['gyr','g_x','g_y','g_z']
        self.magtag=['mag','m_x','m_y','m_z']
        self.eutag=['eu','roll','pitch','yaw']
        self.qtag=['quaternion','q0','q1','q2','q3']
        
        self.scale_label=np.array([[[0,-1],[0,-1],[0,-1]]
            ,[[0,-1],[0,-1],[0,-1]]
            ,[[0,-1],[0,-1],[0,-1]]
            ,[[0,-1],[0,-1],[0,-1]]
            ,[[0,-1],[0,-1],[0,-1]]
            ,[[0,-1],[0,-1],[0,-1]]
            ,[[0,-1],[0,-1],[0,-1]]
            ,[[0,-1],[0,-1],[0,-1]]])
        self.dev_name=dev_name
        if date=='0724':
            self.body_table=['T7','T3','C3','Head','C7','LS','RS']
            self.scale=[1,1,5000,1]
#mounttable[]
            self.dev_id=self.body_table.index(dev_name)
            if raw_or_cali=='raw':
                self.str_path=[['./aiq_data/s1/02.csv','./aiq_data/s1/04.csv','./aiq_data/s1/05.csv'],['./aiq_data/s2/01.csv','./aiq_data/s2/02.csv','./aiq_data/s2/03.csv']]
                self.pathway=self.str_path[subject][session]
            elif raw_or_cali=='cali':
                self.str_path=['./aiq_data/01.csv','./aiq_data/01.csv','./aiq_data/23.csv','./aiq_data/23.csv','./aiq_data/45.csv','./aiq_data/45.csv','./aiq_data/6.csv']
                self.pathway=self.str_path[self.dev_id]
            df = pd.read_csv(self.pathway,header=None)
            df=df.iloc[df.index[df[1]==self.dev_id],:]
            raw=df.as_matrix(columns=df.columns[2:11])
            self.ts=Signal(values=df.as_matrix(columns=df.columns[0:1])/1000**self.scale[3])
            self.delta_ts=self.ts.copy().diff()
            self.gyr_order=np.array([[3,-2,1],[3,-2,1],[3,2,-1],[1,2,3],[3,2,-1],[-3,-1,2],[-3,-1,2]])
            self.gyr_order=self.gyr_order[:,[2,0,1]]
        if date=='0801':
            self.body_table=['LS','C7','T3','T7','C3','Head','RS']
            self.dev_id=self.body_table.index(dev_name)
            self.scale=[1,1,1,1]
            if raw_or_cali=='raw':
                self.str_path=[['./aiq_data/0801/s1/01.csv','./aiq_data/0801/s1/02.csv','./aiq_data/0801/s1/03.csv','./aiq_data/0801/s1/04.csv'],['./aiq_data/0801/s2/01.csv','./aiq_data/0801/s2/02.csv','./aiq_data/0801/s2/03.csv','./aiq_data/0801/s2/04.csv']]
                self.pathway=self.str_path[subject][session]
            elif raw_or_cali=='cali':
                self.str_path=['./aiq_data/0801/01.csv','./aiq_data/0801/01.csv','./aiq_data/0801/23.csv','./aiq_data/0801/23.csv','./aiq_data/0801/45.csv','./aiq_data/0801/45.csv','./aiq_data/0801/6.csv']
                self.pathway=self.str_path[self.dev_id]
            df = pd.read_csv(self.pathway,header=None)
            df=df.iloc[df.index[df[1]==self.dev_id],:]
            raw=df.as_matrix(columns=df.columns[2:11])
            self.ts=Signal(values=df.as_matrix(columns=df.columns[0:1])/1000**self.scale[3])
            self.delta_ts=self.ts.copy().diff()
            self.gyr_order=np.array([[-3,-1,2],[3,2,-1],[3,-2,1],[3,-2,1],[3,2,-1],[1,2,3],[-3,-1,2]])
            self.gyr_order=self.gyr_order[:,[2,0,1]]
            
            self.delta_ts=Signal(np.full((len(self.ts),1),1/31).flatten())
            self.ts=Signal(np.cumsum(self.delta_ts.values))
            self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[self.dev_id])
#mounttable[]
        if date=='0825':
            self.body_table=['dev_0','dev_1','dev_2','dev_3','dev_4','dev_5','dev_6','dev_7']
            self.dev_id=self.body_table.index(dev_name)
            self.scale=[1,1,1,1]
            if raw_or_cali=='raw':
                self.scale_label=np.array([[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[1,130],[122,273],[273,-10]]
                            ,[[5,124],[119,265],[265,-1]]
                            ,[[0,100],[100,200],[200,-1]]
                            ,[[0,120],[120,260],[262,-1]]
                            ,[[0,100],[100,210],[236,-1]]
                            ,[[0,107],[107,193],[189,-1]]])*31
#                            ,[[0,-1],[3755,5600],[5600,-1]]])
                self.str_path=[['./aiq_data/0825/roll01.csv','./aiq_data/0825/pitch01.csv','./aiq_data/0825/yaw01.csv',]
                ,['./aiq_data/0825/roll11.csv','./aiq_data/0825/pitch11.csv','./aiq_data/0825/yaw11.csv',]
                ,['./aiq_data/0825/rollpitchyaw2123.csv','./aiq_data/0825/rollpitchyaw2123.csv','./aiq_data/0825/rollpitchyaw2123.csv',]
                ,['./aiq_data/0825/rollpitchyaw3123.csv','./aiq_data/0825/rollpitchyaw3123.csv','./aiq_data/0825/rollpitchyaw3123.csv',]
                ,['./aiq_data/0825/rollpitchyaw4123.csv','./aiq_data/0825/rollpitchyaw4123.csv','./aiq_data/0825/rollpitchyaw4123.csv',]
                ,['./aiq_data/0825/rollpitchyaw5123.csv','./aiq_data/0825/rollpitchyaw5123.csv','./aiq_data/0825/rollpitchyaw5123.csv',]
                ,['./aiq_data/0825/rollpitchyaw6123.csv','./aiq_data/0825/rollpitchyaw6123.csv','./aiq_data/0825/rollpitchyaw6123.csv',]
                ,['./aiq_data/0825/rollpitchyaw7123.csv','./aiq_data/0825/rollpitchyaw7123.csv','./aiq_data/0825/rollpitchyaw7123.csv',]]
                self.pathway=self.str_path[subject][session]
                self.bias=np.array([[0,0,0],
                               [0,10,0],
                               [0,0,0],
                               [23,23,23],
                               [0,0,0],
                               [0,0,0],
                               [7,7,7],
                               [0,0,0]])*30
            elif raw_or_cali=='cali':
                self.str_path=['./aiq_data/0825/01.csv','./aiq_data/0825/01.csv','./aiq_data/0825/23.csv','./aiq_data/0825/23.csv','./aiq_data/0825/45.csv','./aiq_data/0825/45.csv','./aiq_data/0825/67.csv','./aiq_data/0825/67.csv']
                self.pathway=self.str_path[self.dev_id]
        if date=='1001':
            self.body_table=['head','c4','t4','t7','dev_4','rs','c7','ls']
            self.dev_id=self.body_table.index(dev_name)
            self.scale=[1,1,1,1]
            if raw_or_cali=='raw':
                self.scale_label=np.array([[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[122,273],[273,-10]]
                            ,[[5,124],[119,265],[265,-1]]
                            ,[[0,100],[100,200],[200,-1]]
                            ,[[0,120],[120,260],[262,-1]]
                            ,[[0,100],[100,210],[236,-1]]
                            ,[[0,107],[107,193],[189,-1]]])*31
#                            ,[[0,-1],[3755,5600],[5600,-1]]])
                self.str_path=[['./aiq_data/1001/20180928083111.csv','./aiq_data/1001/20180928091221.csv','./aiq_data/1001/20180928093523.csv',]
                ,['./aiq_data/1004/20181003_HP2xAIQ_inNTU_LiPing.csv','./aiq_data/1004/20181003_HP2xAIQ_inNTU_LiPing_6motion.csv','./aiq_data/1004/20181004_HP2xAIQ_inNTU_LiPing_6motion.csv',]
                ,['./aiq_data/0825/rollpitchyaw2123.csv','./aiq_data/0825/rollpitchyaw2123.csv','./aiq_data/0825/rollpitchyaw2123.csv',]
                ,['./aiq_data/0825/rollpitchyaw3123.csv','./aiq_data/0825/rollpitchyaw3123.csv','./aiq_data/0825/rollpitchyaw3123.csv',]
                ,['./aiq_data/0825/rollpitchyaw4123.csv','./aiq_data/0825/rollpitchyaw4123.csv','./aiq_data/0825/rollpitchyaw4123.csv',]
                ,['./aiq_data/0825/rollpitchyaw5123.csv','./aiq_data/0825/rollpitchyaw5123.csv','./aiq_data/0825/rollpitchyaw5123.csv',]
                ,['./aiq_data/0825/rollpitchyaw6123.csv','./aiq_data/0825/rollpitchyaw6123.csv','./aiq_data/0825/rollpitchyaw6123.csv',]
                ,['./aiq_data/0825/rollpitchyaw7123.csv','./aiq_data/0825/rollpitchyaw7123.csv','./aiq_data/0825/rollpitchyaw7123.csv',]]
                self.pathway=self.str_path[subject][session]
                self.bias=np.array([[0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0]])*30
            elif raw_or_cali=='cali':
                self.str_path=['./aiq_data/0825/01.csv','./aiq_data/0825/01.csv','./aiq_data/0825/23.csv','./aiq_data/0825/23.csv','./aiq_data/0825/45.csv','./aiq_data/0825/45.csv','./aiq_data/0825/67.csv','./aiq_data/0825/67.csv']
                self.pathway=self.str_path[self.dev_id]         
            df = pd.read_csv(self.pathway,header=None)
            df=df.iloc[df.index[df[1]==self.dev_id],:]
            df = df[self.bias[subject][session]:-1]
            df = df[self.scale_label[subject][session][0]:self.scale_label[subject][session][1]]
#use bytearray time
#            tempts=df.iloc[:,0].str.replace('-',' ').str.split(' ',expand=True).applymap(lambda x: int(x, 16)).as_matrix()
#            self.ts=ms=(tempts[:,2])
#            delt_ms=np.diff(ms)
#            err_index=delt_ms<0
#            index=np.where(err_index)
#            for i in index:
#                delt_ms[i]=ms[i+1]+(255-ms[i])
#            delt_ms=np.append(np.array([0]),delt_ms)
#            self.ts=Signal(np.cumsum(delt_ms)/1000)
#            self.delta_ts=Signal(delt_ms/1000)
            
            #use apptime
            self.ts=Signal(values=df.as_matrix(columns=df.columns[0:1])/1000**self.scale[3])
            self.delta_ts=self.ts.copy().diff()
            self.ts.values-=self.ts.values[0]
            
            
            raw=df.as_matrix(columns=df.columns[2:11])
            self.gyr_order=np.full((8,3),[-3,-1,2]) # previous
#            self.gyr_order=np.full((8,3),[2,1,-3]) # after best
        if date=='1004':
            
            self.body_table=['head','c4','t4','t7','rs','0','c7','ls']

            self.dev_id=self.body_table.index(dev_name)
            self.scale=[1,1,1,1]

            if raw_or_cali=='raw':
                self.scale_label=np.array([[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[0,120],[120,260],[262,-1]]
                            ,[[0,100],[100,210],[236,-1]]
                            ,[[0,107],[107,193],[189,-1]]])*31
#                            ,[[0,-1],[3755,5600],[5600,-1]]])
                self.str_path=[['./aiq_data/1001/20180928083111.csv','./aiq_data/1001/20180928091221.csv','./aiq_data/1001/20180928093523.csv',]
                ,['./aiq_data/1004/20181003_HP2xAIQ_inNTU_LiPing.csv','./aiq_data/1004/20181003_HP2xAIQ_inNTU_LiPing_6motion.csv','./aiq_data/1004/20181004_HP2xAIQ_inNTU_LiPing_6motion.csv',]
                ,['./aiq_data/1017/20181016140720.csv','./aiq_data/1017/20181016141906.csv','./aiq_data/1017/20181016151910.csv',]
                ,['./aiq_data/1017/20181016160627.csv','./aiq_data/1017/20181016165233.csv','./aiq_data/1017/XX.csv',]
                ,['./aiq_data/1017/e20181019135238.csv','./aiq_data/1017/e20181019135445.csv','./aiq_data/1017/e20181019144115.csv',]
                ,['./aiq_data/0825/rollpitchyaw5123.csv','./aiq_data/0825/rollpitchyaw5123.csv','./aiq_data/0825/rollpitchyaw5123.csv',]
                ,['./aiq_data/0825/rollpitchyaw6123.csv','./aiq_data/0825/rollpitchyaw6123.csv','./aiq_data/0825/rollpitchyaw6123.csv',]
                ,['./aiq_data/0825/rollpitchyaw7123.csv','./aiq_data/0825/rollpitchyaw7123.csv','./aiq_data/0825/rollpitchyaw7123.csv',]]
                self.pathway=self.str_path[subject][session]
                self.bias=np.array([[0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0]])*30
#            elif raw_or_cali=='cali':
#                self.str_path=['./aiq_data/0825/01.csv','./aiq_data/0825/01.csv','./aiq_data/0825/23.csv','./aiq_data/0825/23.csv','./aiq_data/0825/45.csv','./aiq_data/0825/45.csv','./aiq_data/0825/67.csv','./aiq_data/0825/67.csv']
#                self.pathway=self.str_path[self.dev_id]  
            if raw_or_cali=='cali':
                self.str_path=['./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv']
                self.pathway=self.str_path[self.dev_id]
            df = pd.read_csv(self.pathway,header=None)
            df=df.iloc[df.index[df[1]==self.dev_id],:]
#            df = df[self.bias[subject][session]:-1]
#            df = df[self.scale_label[subject][session][0]:self.scale_label[subject][session][1]]
#use bytearray time
#            tempts=df.iloc[:,0].str.replace('-',' ').str.split(' ',expand=True).applymap(lambda x: int(x, 16)).as_matrix()
#            self.ts=ms=(tempts[:,2])
#            delt_ms=np.diff(ms)
#            err_index=delt_ms<0
#            index=np.where(err_index)
#            for i in index:
#                delt_ms[i]=ms[i+1]+(255-ms[i])
#            delt_ms=np.append(np.array([0]),delt_ms)
#            self.ts=Signal(np.cumsum(delt_ms)/1000)
#            self.delta_ts=Signal(delt_ms/1000)
            
            #use apptime
            self.ts=df.as_matrix(columns=df.columns[0:1])/1000**self.scale[3]
            self.delta_ts=Signal(np.full((len(self.ts),1),1/31).flatten())
            self.ts=Signal(np.cumsum(self.delta_ts.values))
#            self.ts.values[0]
#            self.delta_ts=self.ts.copy().diff()
#            self.ts.values-=self.ts.values[0]
            
            
            raw=df.as_matrix(columns=df.columns[2:11])
            #                             0        1         2           3        4       5         6         7
            self.gyr_order=np.array([[-3,-1,2],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,-1],[-2,-1,-3]])
        if date=='1017':
            
            self.body_table=['head','c4','t4','t7','rs','0','c7','ls']

            self.dev_id=self.body_table.index(dev_name)
            self.scale=[1,1,1,1]
            if raw_or_cali=='cali':
                self.str_path=['./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv']
                self.pathway=self.str_path[self.dev_id]
            if raw_or_cali=='raw':
                self.scale_label=np.array([[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[0,120],[120,260],[262,-1]]
                            ,[[0,100],[100,210],[236,-1]]
                            ,[[0,107],[107,193],[189,-1]]])*31
#                            ,[[0,-1],[3755,5600],[5600,-1]]])
                self.str_path=[['./aiq_data/1017/20181016140720.csv','./aiq_data/1017/20181016141906.csv','./aiq_data/1017/20181016151910.csv',]
                ,['./aiq_data/1017/20181016160627.csv','./aiq_data/1017/20181016165233.csv','./aiq_data/1004/20181004_HP2xAIQ_inNTU_LiPing_6motion.csv',]
                ,['./aiq_data/1016/20181017110201.csv','./aiq_data/1016/20181017112318.csv','./aiq_data/1016/20181017120937.csv',]
                ,['./aiq_data/1016/20181017142203.csv','./aiq_data/1016/20181017150834.csv','./aiq_data/1016/20181017160119.csv',]
                ,['./aiq_data/1017/20181019135238.csv','./aiq_data/1017/20181109141810.csv','./aiq_data/1017/20181109144113.csv',]
                ,['./aiq_data/1109/20181109150417.csv','./aiq_data/1109/20181109152720.csv','./aiq_data/1109/20181109135412.csv',]
                ,['./aiq_data/1109/20181109135412.csv','./aiq_data/1109/20181109135412.csv','./aiq_data/0825/rollpitchyaw6123.csv',]
                ,['./aiq_data/1101/20181101134243.csv','./aiq_data/1101/20181101142912.csv','./aiq_data/1101/20181101151519.csv',]]
                self.pathway=self.str_path[subject][session]
                self.bias=np.array([[0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0],
                               [0,0,0]])*30
#            elif raw_or_cali=='cali':
#                self.str_path=['./aiq_data/0825/01.csv','./aiq_data/0825/01.csv','./aiq_data/0825/23.csv','./aiq_data/0825/23.csv','./aiq_data/0825/45.csv','./aiq_data/0825/45.csv','./aiq_data/0825/67.csv','./aiq_data/0825/67.csv']
#                self.pathway=self.str_path[self.dev_id]  
            if raw_or_cali=='cali':
                self.str_path=['./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv','./aiq_data/1004/00.csv']
                self.pathway=self.str_path[self.dev_id]
            df = pd.read_csv(self.pathway,header=None)
            df=df.iloc[df.index[df[1]==self.dev_id],:]
#            df = df[self.bias[subject][session]:-1]
#            df = df[self.scale_label[subject][session][0]:self.scale_label[subject][session][1]]
#use bytearray time
#            tempts=df.iloc[:,0].str.replace('-',' ').str.split(' ',expand=True).applymap(lambda x: int(x, 16)).as_matrix()
#            self.ts=ms=(tempts[:,2])
#            delt_ms=np.diff(ms)
#            err_index=delt_ms<0
#            index=np.where(err_index)
#            for i in index:
#                delt_ms[i]=ms[i+1]+(255-ms[i])
#            delt_ms=np.append(np.array([0]),delt_ms)
#            self.ts=Signal(np.cumsum(delt_ms)/1000)
#            self.delta_ts=Signal(delt_ms/1000)
            
            #use apptime
            self.ts=Signal(values=df.as_matrix(columns=df.columns[0:1])/1000**self.scale[3])
            self.delta_ts=self.ts.copy()
            
            self.delta_ts.values-=self.delta_ts.values[0]
            self.delta_ts=self.delta_ts.diff()
#            self.ts.values[0]
#            self.delta_ts=self.ts.copy().diff()
#            self.ts.values-=self.ts.values[0]
            
            
            raw=df.as_matrix(columns=df.columns[2:11])
            #                             0        1         2           3        4       5         6         7
            self.gyr_order=np.array([[-3,-1,2],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])
        if date=='1102':
            self.body_table=['head','c4','t4','t7','rs','0','c7','ls']
            self.dev_id=self.body_table.index(dev_name)
            self.scale=[1,1,1,1]
            if raw_or_cali=='raw':
                self.str_path=[
                 ['./aiq_data/1022/20181022100716.csv','./aiq_data/1022/20181022105339.csv','./aiq_data/1022/20181022113946.csv','./aiq_data/1025/20181025091654.csv','./aiq_data/1025/20181025094026.csv',''] #1
                ,['./aiq_data/1018/20181018104759.csv','./aiq_data/1018/20181018111120.csv','./aiq_data/1018/20181018113424.csv','./aiq_data/1018/20181018115727.csv','./aiq_data/1018/20181018122031.csv',''] #2
                ,['./aiq_data/1114/20181114142923.csv','./aiq_data/1114/20181114151552.csv','./aiq_data/1114/20181114160159.csv','./aiq_data/1115/20181115142416.csv','./aiq_data/1115/20181115151035.csv','./aiq_data/1115/20181115155642.csv'] #3
                ,['./aiq_data/1019/20181019135238.csv','./aiq_data/1019/20181019135445.csv','./aiq_data/1019/20181019144115_2.csv','./aiq_data/1023/20181023144644.csv','./aiq_data/1023/20181023153317.csv','./aiq_data/1023/20181023161924.csv'] #4
                ,['./aiq_data/1024/20181024133438.csv','./aiq_data/1024/20181024142100.csv','./aiq_data/1024/20181024150707.csv','./aiq_data/1107/20181107132809.csv','./aiq_data/1107/20181107141430.csv','./aiq_data/1107/20181107150038.csv'] #5
                ,['./aiq_data/1024/20181024161225.csv','./aiq_data/1024/20181024165845.csv','./aiq_data/1024/20181024174451.csv','./aiq_data/1025/20181025155041.csv','./aiq_data/1025/20181025163707.csv','./aiq_data/1025/20181025172314.csv'] #6
                ,['./aiq_data/1026/20181026095301.csv','./aiq_data/1026/20181026103929.csv','./aiq_data/1026/20181026112536.csv','./aiq_data/1031/20181031091929.csv','./aiq_data/1031/20181031100622.csv','./aiq_data/1031/20181031105229.csv'] #7
                ,['./aiq_data/1029/20181029102341.csv','./aiq_data/1029/20181029111049.csv','./aiq_data/1029/20181029115657.csv','./aiq_data/1030/20181030101710.csv','./aiq_data/1030/20181030110358.csv','./aiq_data/1030/20181030115005.csv'] #8
                ,['./aiq_data/1030/20181030151705.csv','./aiq_data/1030/20181030160324.csv','./aiq_data/1030/20181030164931.csv','./aiq_data/1031/20181031151513.csv','./aiq_data/1031/20181031160136.csv','./aiq_data/1031/20181031164743.csv'] #9-12
                ,['./aiq_data/1101/20181101134243.csv','./aiq_data/1101/20181101142912.csv','./aiq_data/1101/20181101151519.csv','./aiq_data/1108/20181108131806.csv','./aiq_data/1108/20181108140425.csv','./aiq_data/1108/20181108145032.csv'] #10-12
                ,['./aiq_data/1102/20181102135136.csv','./aiq_data/1102/20181102143815.csv','./aiq_data/1102/20181102152422.csv','','',''] #11-1
                ,['./aiq_data/1109/20181109135412.csv','./aiq_data/1109/20181109141810.csv','./aiq_data/1109/20181109144113.csv','./aiq_data/1109/20181109150417.csv','./aiq_data/1109/20181109152720.csv','']#11-2
                ,['./aiq_data/1114/20181114101433.csv','./aiq_data/1114/20181114110057.csv','./aiq_data/1114/20181114114704.csv','','',''] #12-1
                ,['./aiq_data/1121/20181121100728.csv','./aiq_data/1121/20181121101639.csv','./aiq_data/1121/20181121110301.csv','./aiq_data/1121/20181121114908.csv','','']] #12-2 (sample size right)

#                print(len(self.str_path))
#                print(self.str_path[13][0])
                self.pathway=self.str_path[subject][session]
#                self.bias=np.zeros_like(self.str_path)
                self.bias=np.array([[1900/31/60,0,0,0,0,0],
                                   [8,0,0,0,0,0],
                                   [0,0,0,0,0,0],
                                   [0,7,0,0,0,0],
                                   [0,0,0,1.5,0,0], #toal 9
                                   [0,0,0,3.5,0,0],#5
                                   [1.2,0,0,0,0,0],
                                   [1,0,0,0,0,0],
                                   [1,0,0,0,0,0], #8
                                   [0,0,0,1,0,0],
                                   [1,0,0,0,0,0],
                                   [0,0,0,0,0,0],
                                   [1,0,0,0,0,0],
                                   [0,0,0,0,0,0]])*31*60
            if raw_or_cali=='cali':
#                self.str_path=['./aiq_data/1102/01-20181029163757.csv','./aiq_data/1102/01-20181029163757.csv','./aiq_data/1102/23-20181029164545.csv','./aiq_data/1102/23-20181029164545.csv','./aiq_data/1102/45-20181029165509.csv','./aiq_data/1102/45-20181029165509.csv','./aiq_data/1102/67-20181029165040.csv','./aiq_data/1102/67-20181029165040.csv']
                self.str_path=['./aiq_data/1102/01-20181029171123.csv','./aiq_data/1102/01-20181029171123.csv','./aiq_data/1102/23-20181029171957.csv','./aiq_data/1102/23-20181029171957.csv','./aiq_data/1102/45-20181029172911.csv','./aiq_data/1102/45-20181029172911.csv','./aiq_data/1102/67-20181029172438','./aiq_data/1102/67-20181029172438.csv']            
                self.pathway=self.str_path[self.dev_id]
            
            df = pd.read_csv(self.pathway,header=None)

            
            df=df.iloc[df.index[df[1]==self.dev_id],:]
#            print('len',len(df))
            print('bias',self.bias[subject][session])
            df = df[int(self.bias[subject][session]):-1]
            #use apptime
            self.ts=Signal(values=df.as_matrix(columns=df.columns[0:1])/1000**self.scale[3])
#            print('the ts[0] is',self.ts.values[0])
            self.delta_ts=self.ts.copy()    
            self.delta_ts.values-=self.delta_ts.values[0]
            
            
            self.delta_ts=self.delta_ts.diff()
            self.delta_ts.values=np.append(self.delta_ts.values,0)
            self.ts=Signal(np.cumsum(self.delta_ts.values))

            raw=df.as_matrix(columns=df.columns[2:11])
            #                             0        1         2           3        4       5         6         7
            self.gyr_order=np.array([[-3,-1,2],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])
        if date=='191031':

            self.body_table=['head','c3','t3','t7','rs','0','c7','ls']
            self.dev_id=self.body_table.index(dev_name)
            self.scale=[1,1,1,1]
            self.cali_path=cali_path
            if raw_or_cali=='raw':
                if raw_path is None:          
                    self.str_path=[
                     ['./raw/190117CROM/20190117083545.csv','./raw/190117CROM/20190117083926.csv','./raw/190117CROM/20190117084258.csv','./aiq_data/1025/20181025091654.csv','./aiq_data/1025/20181025094026.csv',''] #1
                    ,['./aiq_data/1121/20181121100728.csv','./aiq_data/1121/20181121101639.csv','./aiq_data/1121/20181121110301.csv','./aiq_data/1121/20181121114908.csv','','']] #12-2 (sample size right)
                    self.pathway=self.str_path[subject][session]
                else:
                    self.str_path=raw_path
                    self.pathway=raw_path
                self.bias=np.array([[0,0,0,0,0,0],
                                   [0,0,0,0,0,0]])*31*60
                print(self.pathway)
                start=1
            if raw_or_cali=='cali':
                if cali_path is None:
                    self.str_path=['./raw/190117CROM/D02/01.csv','./raw/190117CROM/D02/01.csv','./raw/190117CROM/D02/23.csv','./raw/190117CROM/D02/23.csv','./raw/190117CROM/D02/45.csv','./raw/190117CROM/D02/45.csv','./raw/190117CROM/D02/67.csv','./raw/190117CROM/D02/67.csv']
                    self.pathway=self.str_path[self.dev_id]
                else:
                    self.str_path=cali_path
                    self.pathway=cali_path
                start=1
            df = pd.read_csv(self.pathway,header=None)
            
            df=df.iloc[df.index[df[start+1]==self.dev_id],:]
#            print(df)
#            print('len',len(df))
#            print('bias',self.bias[subject][session])
            df = df[int(self.bias[subject][session]):-1]
            #use apptime5
            self.ts=Signal(values=df.as_matrix(columns=df.columns[1:2])/1000**self.scale[3])
#            print('the ts[0] is',self.ts.values[0])
            self.delta_ts=self.ts.copy() 
#            print('abc')
#            print(self.delta_ts.values)
#            print('abc2')
            self.delta_ts.values-=self.delta_ts.values[0]
            
            self.file_tag='crom_imu'
            self.delta_ts=self.delta_ts.diff()
            self.delta_ts.values=np.append(self.delta_ts.values,0)
            self.ts=Signal(np.cumsum(self.delta_ts.values))

            raw=df.as_matrix(columns=df.columns[start+2:12])
            #                             0        1         2           3        4       5         6         7
            self.gyr_order=np.array([[-3,-1,2],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])
#                if date=='190117':

            self.body_table=['head','c3','t3','t7','rs','0','c7','ls']
            self.dev_id=self.body_table.index(dev_name)
            self.scale=[1,1,1,1]
            df = pd.read_csv(self.pathway,header=None)
            
            df=df.iloc[df.index[df[start+1]==self.dev_id],:]
#            print('len',len(df))
#            print('bias',self.bias[subject][session])
            df = df[int(self.bias[subject][session]):-1]

            raw=df.as_matrix(columns=df.columns[start+2:12])
            #                             0        1         2           3        4       5         6         7
#            self.gyr_order=np.array([[-3,-1,2],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])
            self.gyr_order=np.array([[-3,1,-2],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])

        if date=='1206'or date=='without_remind':
            
#            subject=subject-1
            self.body_table=['head','c3','t4','t7','0','rs','c7','ls']
            self.bias=np.array([1,8,2,2,1,3.5,1,1,1,1,1,1,1,0.5,1,1])*31*60

            self.dev_id=self.body_table.index(dev_name)
            self.scale=[1,1,1,1]
            self.tablename=['imu01','imu02','imu03','imu04','imu05','imu06','imu07','imu08','imu09','imu10','imu11','imu12','imu013','imu014','imu015','imu016']
            if raw_or_cali=='raw':
                db=SQLite(path='./imu_without_remind.db')
                print(self.tablename[subject-1])
                print('dev_id=='+str(self.body_table.index(dev_name)))
                imu007=Table(db,self.tablename[subject-1])
#                raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name)))
#                raw = raw[int(self.bias[subject-1]):-1]
#                raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name))+' AND rowid<100000')
                raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name))+' AND rowid<100000')

#                raw = raw[int(self.bias[subject-1]):60000]
                
                self.pathway='./db'
            if raw_or_cali=='cali':
#                self.str_path=['./aiq_data/D01/01.csv','./aiq_data/D01/01.csv','./aiq_data/D01/23.csv','./aiq_data/D01/23.csv','./aiq_data/D01/45.csv','./aiq_data/D01/45.csv','./aiq_data/D01/67.csv','./aiq_data/D01/67.csv']
                self.str_path=['./aiq_data/D02/01.csv','./aiq_data/D02/01.csv','./aiq_data/D02/23.csv','./aiq_data/D02/23.csv','./aiq_data/D02/45.csv','./aiq_data/D02/45.csv','./aiq_data/D02/67.csv','./aiq_data/D02/67.csv']
#                self.str_path=['./aiq_data/D03/01.csv','./aiq_data/D03/01.csv','./aiq_data/D03/23.csv','./aiq_data/D03/23.csv','./aiq_data/D03/45.csv','./aiq_data/D03/45.csv','./aiq_data/D03/67.csv','./aiq_data/D03/67.csv']

#                self.str_path=['./aiq_data/D12/01.csv','./aiq_data/D12/01.csv','./aiq_data/D12/23.csv','./aiq_data/D12/23.csv','./aiq_data/D12/45.csv','./aiq_data/D12/45.csv','./aiq_data/D12/67.csv','./aiq_data/D12/67.csv']
#                self.str_path=['./aiq_data/D13/01.csv','./aiq_data/D13/01.csv','./aiq_data/D13/23.csv','./aiq_data/D13/23.csv','./aiq_data/D13/45.csv','./aiq_data/D13/45.csv','./aiq_data/D13/67.csv','./aiq_data/D13/67.csv']
                
                self.pathway=self.str_path[self.dev_id]
                df = pd.read_csv(self.pathway,header=None)
                df=df.iloc[df.index[df[1]==self.dev_id],:]
                raw=df.as_matrix(columns=df.columns[:])
                print('raw shape',self.pathway)
                print('raw shape',raw.shape)
#                print('raw shape',raw)
                print(raw[:,2])            
            self.file_tag=date+'_'+str(subject)+'_'+str(session)+'_'

            #use apptime
            self.ts=Signal(values=raw[:,0]/1000**self.scale[3])
            plt.plot(self.ts.values)
            print(self.ts.values)
            self.delta_ts=self.ts.copy()    
            self.delta_ts.values-=self.delta_ts.values[0]
            self.delta_ts=self.delta_ts.diff()
            raw=raw[:,2:11]
            
            #                             0        1         2           3        4       5         6         7
            self.gyr_order=np.array([[-3,-1,2],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])
        if date=='1207'or date=='remind':
#            subject=subject-1
#            self.body_table=['head','c3','t4','t7','rs','0','c7','ls']
            self.body_table=['head','c3','t4','t7','0','rs','c7','ls']
            
            self.bias=np.array([1,0.7,0.7,0.2,0.5,1.54,0.6,0.6,0.5,0.6,0.63,0.5,0.3,0,1,0.5])*31*60
            
            self.dev_id=self.body_table.index(dev_name)
            self.scale=[1,1,1,1]
            self.tablename=['imu01','imu02','imu03','imu04','imu05','imu06','imu07','imu08','imu09','imu10','imu11','imu12','imu13','imu14','imu15','imu016']
            if raw_or_cali=='raw':
                db=SQLite(path='./imu_remind.db')
                print(self.tablename[subject-1])
                print('dev_id=='+str(self.body_table.index(dev_name)))
                imu007=Table(db,self.tablename[subject-1])
#                raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name)))
                raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name))+' AND rowid<100000')
#                raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name))+' AND rowid<200000')

#                print(raw.shape)
                raw = raw[int(self.bias[subject-1]):-1]
#                raw = raw[int(self.bias[subject-1]):60000]
                self.pathway='./db'
            if raw_or_cali=='cali':
#                self.str_path=['./aiq_data/D01/01.csv','./aiq_data/D01/01.csv','./aiq_data/D01/23.csv','./aiq_data/D01/23.csv','./aiq_data/D01/45.csv','./aiq_data/D01/45.csv','./aiq_data/D01/67.csv','./aiq_data/D01/67.csv']
#                self.str_path=['./aiq_data/D02/01.csv','./aiq_data/D02/01.csv','./aiq_data/D02/23.csv','./aiq_data/D02/23.csv','./aiq_data/D02/45.csv','./aiq_data/D02/45.csv','./aiq_data/D02/67.csv','./aiq_data/D02/67.csv']
                self.str_path=['./aiq_data/D03/01.csv','./aiq_data/D03/01.csv','./aiq_data/D03/23.csv','./aiq_data/D03/23.csv','./aiq_data/D03/45.csv','./aiq_data/D03/45.csv','./aiq_data/D03/67.csv','./aiq_data/D03/67.csv']

#                self.str_path=['./aiq_data/D12/01.csv','./aiq_data/D12/01.csv','./aiq_data/D12/23.csv','./aiq_data/D12/23.csv','./aiq_data/D12/45.csv','./aiq_data/D12/45.csv','./aiq_data/D12/67.csv','./aiq_data/D12/67.csv']
#                self.str_path=['./aiq_data/D13/01.csv','./aiq_data/D13/01.csv','./aiq_data/D13/23.csv','./aiq_data/D13/23.csv','./aiq_data/D13/45.csv','./aiq_data/D13/45.csv','./aiq_data/D13/67.csv','./aiq_data/D13/67.csv']
     


                self.pathway=self.str_path[self.dev_id]
                df = pd.read_csv(self.pathway,header=None)
                df=df.iloc[df.index[df[1]==self.dev_id],:]
                raw=df.as_matrix(columns=df.columns[:])
                print('raw shape',self.pathway)
                print('raw shape',raw.shape)
#                print('raw shape',raw)
                print(raw[:,2])
            #use apptime
            self.ts=Signal(values=raw[:,0]/1000**self.scale[3])

            self.delta_ts=self.ts.copy()    
            self.delta_ts.values-=self.delta_ts.values[0]
            self.delta_ts=self.delta_ts.diff()
            raw=raw[:,2:11]
            self.file_tag=date+'_'+str(subject)+'_'+str(session)+'_'
            #                             0        1         2           3        4       5         6         7
            self.gyr_order=np.array([[-3,-1,2],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])
        if date=='190201':
#            subject=subject-1
            self.body_table=['head','c3','t4','t7','0','rs','c7','ls']
            
#            self.bias=np.array([1,0.7,0.7,0.2,0.5,1.54,0.6,0.6,0.5,0.6,0.63,0.5,0.3,0,1,0.5])*31*60
            self.dev_id=self.body_table.index(dev_name)
            self.scale=[1,1,1,1]
            self.tablename=['imu01','imu02','imu03','imu04','imu05','imu06','imu07','imu08','imu09','imu10','imu11','imu12','imu13','imu14','imu15','imu016']
            if raw_or_cali=='raw':
                db=SQLite(path='./crom_imu_compare.db')
                print(db.table_list)
                #print(self.tablename[subject-1])
                #print('dev_id=='+str(self.body_table.index(dev_name)))
                
                tablename='imu'+'{:0>3d}'.format(subject)+'_'+str(session)
                print('try to read ',tablename)
                imu007=Table(db,tablename)
                self.file_tag=tablename
#                raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name)))
                raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name))+' AND rowid<100000')
#                raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name))+' AND rowid<200000')

#                print(raw.shape)
                #raw = raw[int(self.bias[subject-1]):-1]
#                raw = raw[int(self.bias[subject-1]):60000]
                self.pathway='./db'
            if raw_or_cali=='cali':
#                self.str_path=['./aiq_data/D01/01.csv','./aiq_data/D01/01.csv','./aiq_data/D01/23.csv','./aiq_data/D01/23.csv','./aiq_data/D01/45.csv','./aiq_data/D01/45.csv','./aiq_data/D01/67.csv','./aiq_data/D01/67.csv']
#                self.str_path=['./aiq_data/D02/01.csv','./aiq_data/D02/01.csv','./aiq_data/D02/23.csv','./aiq_data/D02/23.csv','./aiq_data/D02/45.csv','./aiq_data/D02/45.csv','./aiq_data/D02/67.csv','./aiq_data/D02/67.csv']
                self.str_path=['./aiq_data/D03/01.csv','./aiq_data/D03/01.csv','./aiq_data/D03/23.csv','./aiq_data/D03/23.csv','./aiq_data/D03/45.csv','./aiq_data/D03/45.csv','./aiq_data/D03/67.csv','./aiq_data/D03/67.csv']

#                self.str_path=['./aiq_data/D12/01.csv','./aiq_data/D12/01.csv','./aiq_data/D12/23.csv','./aiq_data/D12/23.csv','./aiq_data/D12/45.csv','./aiq_data/D12/45.csv','./aiq_data/D12/67.csv','./aiq_data/D12/67.csv']
#                self.str_path=['./aiq_data/D13/01.csv','./aiq_data/D13/01.csv','./aiq_data/D13/23.csv','./aiq_data/D13/23.csv','./aiq_data/D13/45.csv','./aiq_data/D13/45.csv','./aiq_data/D13/67.csv','./aiq_data/D13/67.csv']
     


                self.pathway=self.str_path[self.dev_id]
                df = pd.read_csv(self.pathway,header=None)
                df=df.iloc[df.index[df[1]==self.dev_id],:]
                raw=df.as_matrix(columns=df.columns[:])
#                print('raw shape',self.pathway)
#                print('raw shape',raw.shape)
#                print('raw shape',raw)
#                print(raw[:,2])
            #use apptime
            self.ts=Signal(values=raw[:,0]/1000**self.scale[3])

            self.delta_ts=self.ts.copy()    
            self.delta_ts.values-=self.delta_ts.values[0]
            self.delta_ts=self.delta_ts.diff()
            raw=raw[:,2:11]
            self.file_tag=date+'_'+'{:0>2d}'.format(subject)+'_'+'{:0>2d}'.format(session)+'_'
            #                             0        1         2           3        4       5         6         7
            self.gyr_order=np.array([[-3,-1,2],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])
        if date=='190307': #測試新版app資料品質
#            subject=subject-1
            self.body_table=['head','c3','t3','t7','rs','0','c7','ls']
            
#            self.bias=np.array([1,0.7,0.7,0.2,0.5,1.54,0.6,0.6,0.5,0.6,0.63,0.5,0.3,0,1,0.5])*31*60
            self.dev_id=self.body_table.index(dev_name)
            self.scale=[1,1,1,1]
#            self.tablename=['imu01','imu02','imu03','imu04','imu05','imu06','imu07','imu08','imu09','imu10','imu11','imu12','imu13','imu14','imu15','imu016']
            if raw_or_cali=='raw':
                db=SQLite(path='./raw/190308test_app.db')
                print(db.table_list)
                #print(self.tablename[subject-1])
                #print('dev_id=='+str(self.body_table.index(dev_name)))
                
                tablename='imu'+'{:0>3d}'.format(subject)#+'_'+str(session)
                print('try to read ',tablename)
                imu007=Table(db,tablename)
                self.file_tag=tablename
#                raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name)))
                if testmode:
                    raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name))+' AND rowid<1000000')
                else:
                    raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name)))
                    
#                
#                if testmode:
#                    
#                    Reader.figure(path='./raw/20190227091323_3.csv',shape=[40000,11],specific=[1,self.dev_id])
#                else:
#                    
#                    Reader.figure(path='./raw/20190227091323_3.csv',specific=[1,self.dev_id])
#
#                print('specifcid is',self.dev_id)
#                raw=Reader.export()
                raw_delt=np.diff(raw[:,0])
            #    plt.plot(raw_delt)
                temp_ts=raw[:-1,0]#前
                temp_ts2=raw[1:,0]#後
                mk=raw_delt<0
                raw_delt[mk]=temp_ts2[mk]-temp_ts[mk]+65536
                raw_delt=raw_delt[1:]/1000
                ts=np.cumsum(raw_delt)
                ts-=ts[0]
                raw[:-2,0]=ts
                raw=raw[:-2]

#                print(raw.shape)
                #raw = raw[int(self.bias[subject-1]):-1]
#                raw = raw[int(self.bias[subject-1]):60000]
#                self.pathway='./db'
            if raw_or_cali=='cali':
#                self.str_path=['./aiq_data/D01/01.csv','./aiq_data/D01/01.csv','./aiq_data/D01/23.csv','./aiq_data/D01/23.csv','./aiq_data/D01/45.csv','./aiq_data/D01/45.csv','./aiq_data/D01/67.csv','./aiq_data/D01/67.csv']
#                self.str_path=['./aiq_data/D02/01.csv','./aiq_data/D02/01.csv','./aiq_data/D02/23.csv','./aiq_data/D02/23.csv','./aiq_data/D02/45.csv','./aiq_data/D02/45.csv','./aiq_data/D02/67.csv','./aiq_data/D02/67.csv']
                self.str_path=['./aiq_data/D03/01.csv','./aiq_data/D03/01.csv','./aiq_data/D03/23.csv','./aiq_data/D03/23.csv','./aiq_data/D03/45.csv','./aiq_data/D03/45.csv','./aiq_data/D03/67.csv','./aiq_data/D03/67.csv']

#                self.str_path=['./aiq_data/D12/01.csv','./aiq_data/D12/01.csv','./aiq_data/D12/23.csv','./aiq_data/D12/23.csv','./aiq_data/D12/45.csv','./aiq_data/D12/45.csv','./aiq_data/D12/67.csv','./aiq_data/D12/67.csv']
#                self.str_path=['./aiq_data/D13/01.csv','./aiq_data/D13/01.csv','./aiq_data/D13/23.csv','./aiq_data/D13/23.csv','./aiq_data/D13/45.csv','./aiq_data/D13/45.csv','./aiq_data/D13/67.csv','./aiq_data/D13/67.csv']
     


                self.pathway=self.str_path[self.dev_id]
                df = pd.read_csv(self.pathway,header=None)
                df=df.iloc[df.index[df[1]==self.dev_id],:]
                raw=df.as_matrix(columns=df.columns[:])
#                print('raw shape',self.pathway)
#                print('raw shape',raw.shape)
#                print('raw shape',raw)
#                print(raw[:,2])
            #use apptime
            self.ts=Signal(raw[:,0])
            self.delta_ts=self.ts.copy()    
            self.delta_ts.values-=self.delta_ts.values[0]
            self.delta_ts=self.delta_ts.diff()
#            
#            plt.show()
#            plt.plot(self.ts.values,label='self.ts')
#            plt.legend()
#            plt.show()    
#            print('do 2:11')
            raw=raw[:,2:11]
            #                raw[:,5:8]=raw[:,5:8]/np.pi*180
#            df=pd.DataFrame(raw[:,2:])
#            plt.show()
#            plt.plot(raw[:,0:3],label='acc')
#            plt.legend()
#            plt.show()
            raw[:,3:6]=raw[:,3:6]/np.pi*180
#            plt.plot(raw[:,3:6],label='gyr')   
#            plt.legend()
#            plt.show()
            self.file_tag='remind_appv2_test'+'_'
            #                             0        1         2           3        4       5         6         7
            self.gyr_order=np.array([[-3,-1,2],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])
            #0307
#            self.gyr_order=np.array([[-2,3,-1],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-1,3,2],[-2,-1,-3]])
#            self.gyr_order=np.array([[-3,-2,-1],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-1,3,2],[-2,-1,-3]])


#            self.temp=np.zeros_like(self.gyr_order)
#            for i in range(8):
#                self.gyr_order[i]=[1,2,3] 
        if date=='190121':
#            subject=subject-1
            self.body_table=['head','c3','t3','t7','0','rs','c7','ls']
            

#            self.bias=np.array([1,0.7,0.7,0.2,0.5,1.54,0.6,0.6,0.5,0.6,0.63,0.5,0.3,0,1,0.5])*31*60
            self.dev_id=self.body_table.index(dev_name)
            self.scale=[1,1,1,1]
            self.tablename=['imu01','imu02','imu03','imu04','imu05','imu06','imu07','imu08','imu09','imu10','imu11','imu12','imu13','imu14','imu15','imu016']
            if raw_or_cali=='raw':
                db=SQLite(path='./crom_imu_compare.db')
                if tablename is not None:
                    pass
                else:
                    print(self.tablename[subject-1])
                    print('dev_id=='+str(self.body_table.index(dev_name)))
                    
                    tablename='imu{:0>3d}'.format(subject)+'_'+str(session)
                    
                imu007=Table(db,tablename)
                self.file_tag=tablename
#                raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name)))
                raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name))+' AND rowid<100000')
#                raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name))+' AND rowid<200000')

#                print(raw.shape)
#                raw = raw[int(self.bias[subject-1]):-1]
#                raw = raw[int(self.bias[subject-1]):60000]
                self.pathway='./db'
            if raw_or_cali=='cali':
#                self.str_path=['./aiq_data/D01/01.csv','./aiq_data/D01/01.csv','./aiq_data/D01/23.csv','./aiq_data/D01/23.csv','./aiq_data/D01/45.csv','./aiq_data/D01/45.csv','./aiq_data/D01/67.csv','./aiq_data/D01/67.csv']
#                self.str_path=['./aiq_data/D02/01.csv','./aiq_data/D02/01.csv','./aiq_data/D02/23.csv','./aiq_data/D02/23.csv','./aiq_data/D02/45.csv','./aiq_data/D02/45.csv','./aiq_data/D02/67.csv','./aiq_data/D02/67.csv']
                self.str_path=['./aiq_data/D03/01.csv','./aiq_data/D03/01.csv','./aiq_data/D03/23.csv','./aiq_data/D03/23.csv','./aiq_data/D03/45.csv','./aiq_data/D03/45.csv','./aiq_data/D03/67.csv','./aiq_data/D03/67.csv']

#                self.str_path=['./aiq_data/D12/01.csv','./aiq_data/D12/01.csv','./aiq_data/D12/23.csv','./aiq_data/D12/23.csv','./aiq_data/D12/45.csv','./aiq_data/D12/45.csv','./aiq_data/D12/67.csv','./aiq_data/D12/67.csv']
#                self.str_path=['./aiq_data/D13/01.csv','./aiq_data/D13/01.csv','./aiq_data/D13/23.csv','./aiq_data/D13/23.csv','./aiq_data/D13/45.csv','./aiq_data/D13/45.csv','./aiq_data/D13/67.csv','./aiq_data/D13/67.csv']
     


                self.pathway=self.str_path[self.dev_id]
                df = pd.read_csv(self.pathway,header=None)
                df=df.iloc[df.index[df[1]==self.dev_id],:]
                raw=df.as_matrix(columns=df.columns[:])
#                print('raw shape',self.pathway)
#                print('raw shape',raw.shape)
#                print('raw shape',raw)
#                print(raw[:,2])
            #use apptime
            self.ts=Signal(values=raw[:,0]/1000**self.scale[3])

            self.delta_ts=self.ts.copy()    
            self.delta_ts.values-=self.delta_ts.values[0]
            self.delta_ts=self.delta_ts.diff()
            raw=raw[:,2:11]
            self.file_tag=date+'_'+str(subject)+'_'+str(session)+'_'
            #                             0        1         2           3        4       5         6         7
            self.gyr_order=np.array([[-3,-1,2],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])
        
        if date=='0115'or date=='ntuh_work':
#            subject=subject-1
#            self.body_table=['head','c3','t4','t7','rs','0','c7','ls']
            self.body_table=['head','c3','t4','t7','0','rs','c7','ls']
            #                   1, 2 , 3 ,  4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
            self.bias=np.array([8, 6,  5,  12, 4, 6, 6,6,6,6 , 6, 6,8,  4,5,  8, 7, 7, 7, 9, 6.5,5])*60

#            self.bias=np.array([8,47,33, 12, 6, 6, 6,6,6,6 ,27, 6,8,  4,5,  8, 7,33, 7, 9, 7,5])*60
            
            self.dev_id=self.body_table.index(dev_name)
            self.scale=[1,1,1,1]
#            self.tablename=['imu001','imu002','imu003','imu004','imu005','imu006','imu007','imu008','imu009','imu010','imu011','imu012','imu013','imu014','imu015','imu016','imu017','imu018','imu019','imu020','imu021','imu022']
            self.tablename="imu"+"{:0>3d}".format(subject)
            
            if session!=1:
                self.tablename+='_'+str(session)
            if raw_or_cali=='raw':
                db=SQLite(path='./ntuh_work.db')
                self.file_tag='ntuh_work'
                print(self.tablename)
                print('dev_id=='+str(self.body_table.index(dev_name)))
                imu007=Table(db,self.tablename)
                if testmode:
                    raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name))+' AND rowid<100000')
                else:
                    raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name)))
                    
#                raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name))+' AND rowid<200000')

#                print(raw.shape)
#                raw = raw[int(self.bias[subject-1]):-1]
#                raw = raw[int(self.bias[subject-1]):60000]
                self.pathway='./db'
            if raw_or_cali=='cali':
#                self.str_path=['./aiq_data/D01/01.csv','./aiq_data/D01/01.csv','./aiq_data/D01/23.csv','./aiq_data/D01/23.csv','./aiq_data/D01/45.csv','./aiq_data/D01/45.csv','./aiq_data/D01/67.csv','./aiq_data/D01/67.csv']
#                self.str_path=['./aiq_data/D02/01.csv','./aiq_data/D02/01.csv','./aiq_data/D02/23.csv','./aiq_data/D02/23.csv','./aiq_data/D02/45.csv','./aiq_data/D02/45.csv','./aiq_data/D02/67.csv','./aiq_data/D02/67.csv']
                self.str_path=['./aiq_data/D03/01.csv','./aiq_data/D03/01.csv','./aiq_data/D03/23.csv','./aiq_data/D03/23.csv','./aiq_data/D03/45.csv','./aiq_data/D03/45.csv','./aiq_data/D03/67.csv','./aiq_data/D03/67.csv']

#                self.str_path=['./aiq_data/D12/01.csv','./aiq_data/D12/01.csv','./aiq_data/D12/23.csv','./aiq_data/D12/23.csv','./aiq_data/D12/45.csv','./aiq_data/D12/45.csv','./aiq_data/D12/67.csv','./aiq_data/D12/67.csv']
#                self.str_path=['./aiq_data/D13/01.csv','./aiq_data/D13/01.csv','./aiq_data/D13/23.csv','./aiq_data/D13/23.csv','./aiq_data/D13/45.csv','./aiq_data/D13/45.csv','./aiq_data/D13/67.csv','./aiq_data/D13/67.csv']
     


                self.pathway=self.str_path[self.dev_id]
                df = pd.read_csv(self.pathway,header=None)
                df=df.iloc[df.index[df[1]==self.dev_id],:]
                raw=df.as_matrix(columns=df.columns[:])
                
                print('raw shape',self.pathway)
                print('raw shape',raw.shape)
#                print('raw shape',raw)
                print(raw[:,2])
            #use apptime
            #filter repeatly and unnecessary
            test=np.diff(raw[:,4])
            test=np.hstack((test,np.zeros(1)))
            mk=test!=0
            raw=raw[mk,:]
#            self.ts=Signal(values=raw[:,0]/1000**self.scale[3])
            self.ts=Signal(values=np.linspace(0,raw.shape[0]*(1/31),raw.shape[0]))
            self.delta_ts=self.ts.copy()    
            self.delta_ts.values-=self.delta_ts.values[0]
            self.delta_ts=self.delta_ts.diff()
            raw=raw[:,2:11]
            self.file_tag=date+'_'+str(subject)+'_'+str(session)+'_'
            #                             0        1         2           3        4       5         6         7
            self.gyr_order=np.array([[-3,-1,2],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])
        if date=='ntuh_work_baseline':
#            subject=subject-1
#            self.body_table=['head','c3','t4','t7','rs','0','c7','ls']
            self.body_table=['head','c3','t4','t7','0','rs','c7','ls']
            #                   1, 2 , 3 ,  4, 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
            self.bias=np.array([8, 6,  5,  12, 4, 6, 6,6,6,6 , 6, 6,8,  4,5,  8, 7, 7, 7, 9, 6.5,5])*60

#            self.bias=np.array([8,47,33, 12, 6, 6, 6,6,6,6 ,27, 6,8,  4,5,  8, 7,33, 7, 9, 7,5])*60
            
            self.dev_id=self.body_table.index(dev_name)
            self.scale=[1,1,1,1]
#            self.tablename=['imu001','imu002','imu003','imu004','imu005','imu006','imu007','imu008','imu009','imu010','imu011','imu012','imu013','imu014','imu015','imu016','imu017','imu018','imu019','imu020','imu021','imu022']
            self.tablename="imu"+"{:0>3d}".format(subject)
            
            if session!=1:
                self.tablename+='_'+str(session)
            if raw_or_cali=='raw':
                db=SQLite(path='./ntuh_work_baseline.db')
                self.file_tag='ntuh_work'
                print(self.tablename)
                print('dev_id=='+str(self.body_table.index(dev_name)))
                imu007=Table(db,self.tablename)
                if testmode:
                    raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name))+' AND rowid<100000')
                else:
                    raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name)))
                    
#                raw=imu007.export_data(condition='dev_id=='+str(self.body_table.index(dev_name))+' AND rowid<200000')

#                print(raw.shape)
#                raw = raw[int(self.bias[subject-1]):-1]
#                raw = raw[int(self.bias[subject-1]):60000]
                self.pathway='./db'
            if raw_or_cali=='cali':
#                self.str_path=['./aiq_data/D01/01.csv','./aiq_data/D01/01.csv','./aiq_data/D01/23.csv','./aiq_data/D01/23.csv','./aiq_data/D01/45.csv','./aiq_data/D01/45.csv','./aiq_data/D01/67.csv','./aiq_data/D01/67.csv']
#                self.str_path=['./aiq_data/D02/01.csv','./aiq_data/D02/01.csv','./aiq_data/D02/23.csv','./aiq_data/D02/23.csv','./aiq_data/D02/45.csv','./aiq_data/D02/45.csv','./aiq_data/D02/67.csv','./aiq_data/D02/67.csv']
                self.str_path=['./aiq_data/D03/01.csv','./aiq_data/D03/01.csv','./aiq_data/D03/23.csv','./aiq_data/D03/23.csv','./aiq_data/D03/45.csv','./aiq_data/D03/45.csv','./aiq_data/D03/67.csv','./aiq_data/D03/67.csv']

#                self.str_path=['./aiq_data/D12/01.csv','./aiq_data/D12/01.csv','./aiq_data/D12/23.csv','./aiq_data/D12/23.csv','./aiq_data/D12/45.csv','./aiq_data/D12/45.csv','./aiq_data/D12/67.csv','./aiq_data/D12/67.csv']
#                self.str_path=['./aiq_data/D13/01.csv','./aiq_data/D13/01.csv','./aiq_data/D13/23.csv','./aiq_data/D13/23.csv','./aiq_data/D13/45.csv','./aiq_data/D13/45.csv','./aiq_data/D13/67.csv','./aiq_data/D13/67.csv']
     


                self.pathway=self.str_path[self.dev_id]
                df = pd.read_csv(self.pathway,header=None)
                df=df.iloc[df.index[df[1]==self.dev_id],:]
                raw=df.as_matrix(columns=df.columns[:])
                
                print('raw shape',self.pathway)
                print('raw shape',raw.shape)
#                print('raw shape',raw)
                print(raw[:,2])
            #use apptime
            #filter repeatly and unnecessary
            test=np.diff(raw[:,4])
            test=np.hstack((test,np.zeros(1)))
            mk=test!=0
            raw=raw[mk,:]
#            self.ts=Signal(values=raw[:,0]/1000**self.scale[3])
            self.ts=Signal(values=np.linspace(0,raw.shape[0]*(1/31),raw.shape[0]))
            self.delta_ts=self.ts.copy()    
            self.delta_ts.values-=self.delta_ts.values[0]
            self.delta_ts=self.delta_ts.diff()
            raw=raw[:,2:11]
            self.file_tag=date+'_'+str(subject)+'_'+str(session)+'_'
            #                             0        1         2           3        4       5         6         7
            self.gyr_order=np.array([[-3,-1,2],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])
            self.file_tag='ntuh_work_baseline'
        if date=='1218' or date=='1218black_xsens_validation':
            self.body_table=['black','c4','t4','t7','rs','0','c7','ls']
            self.dev_id=self.body_table.index(dev_name)
            self.scale=[1,1,1,1]
            if raw_or_cali=='raw':
                self.str_path=[
                 ['./1207elipse_fitting/black_validation_black.csv','./1207elipse_fitting/black_validation_black.csv','./1207elipse_fitting/black_validation_black.csv','./aiq_data/1025/20181025091654.csv','./aiq_data/1025/20181025094026.csv',''] #1
                ,['./1207elipse_fitting/subject2/black_validation_black2.csv','./1207elipse_fitting/subject2/black_validation_black2.csv','./1207elipse_fitting/subject2/black_validation_black2.csv','./aiq_data/1121/20181121114908.csv','','']#12-2 (sample size right)
                ,['./1207elipse_fitting/subject3/black_validation_black3.csv','./1207elipse_fitting/subject3/black_validation_black3.csv','./1207elipse_fitting/subject3/black_validation_black3.csv','./aiq_data/1121/20181121114908.csv','','']] 
                self.pathway=self.str_path[subject][session]
                self.bias=np.array([[0,0,0,0,0,0],
                                    [0,0,0,0,0,0],
                                   [0,0,0,0,0,0]])*31*60
                self.scale_label=np.array([[[0,35],[35,62],[62,-1]]
                            ,[[0,15],[15,28],[26,-1]]
                            ,[[0,33],[30,65],[59,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[0,120],[120,260],[262,-1]]
                            ,[[0,100],[100,210],[236,-1]]
                            ,[[0,107],[107,193],[189,-1]]])*100
                
#                print(self.str_path)
            if raw_or_cali=='cali':
#                self.str_path=['./1207elipse_fitting/20181218110723.csv','./aiq_data/1102/01-20181029171123.csv','./aiq_data/1102/23-20181029171957.csv','./aiq_data/1102/23-20181029171957.csv','./aiq_data/1102/45-20181029172911.csv','./aiq_data/1102/45-20181029172911.csv','./aiq_data/1102/67-20181029172438','./aiq_data/1102/67-20181029172438.csv']            
#                self.str_path=['./1207elipse_fitting/20181219142045.csv','./aiq_data/1102/01-20181029171123.csv','./aiq_data/1102/23-20181029171957.csv','./aiq_data/1102/23-20181029171957.csv','./aiq_data/1102/45-20181029172911.csv','./aiq_data/1102/45-20181029172911.csv','./aiq_data/1102/67-20181029172438','./aiq_data/1102/67-20181029172438.csv']            
#                self.str_path=['./1207elipse_fitting/subject3/1220calibration.csv','./aiq_data/1102/01-20181029171123.csv','./aiq_data/1102/23-20181029171957.csv','./aiq_data/1102/23-20181029171957.csv','./aiq_data/1102/45-20181029172911.csv','./aiq_data/1102/45-20181029172911.csv','./aiq_data/1102/67-20181029172438','./aiq_data/1102/67-20181029172438.csv']            
                self.str_path=['./1207elipse_fitting/subject3/1220calibration2.csv','./aiq_data/1102/01-20181029171123.csv','./aiq_data/1102/23-20181029171957.csv','./aiq_data/1102/23-20181029171957.csv','./aiq_data/1102/45-20181029172911.csv','./aiq_data/1102/45-20181029172911.csv','./aiq_data/1102/67-20181029172438','./aiq_data/1102/67-20181029172438.csv']            
                
                self.pathway=self.str_path[self.dev_id]
            if raw_or_cali=='acc_cali':
                self.str_path=['./1207elipse_fitting/acc_calibration2.csv','./1207elipse_fitting/acc_calibration.csv','./aiq_data/1102/23-20181029171957.csv','./aiq_data/1102/23-20181029171957.csv','./aiq_data/1102/45-20181029172911.csv','./aiq_data/1102/45-20181029172911.csv','./aiq_data/1102/67-20181029172438','./aiq_data/1102/67-20181029172438.csv']            
#                self.str_path=['./1207elipse_fitting/20181219142045.csv','./aiq_data/1102/01-20181029171123.csv','./aiq_data/1102/23-20181029171957.csv','./aiq_data/1102/23-20181029171957.csv','./aiq_data/1102/45-20181029172911.csv','./aiq_data/1102/45-20181029172911.csv','./aiq_data/1102/67-20181029172438','./aiq_data/1102/67-20181029172438.csv']            
                self.pathway=self.str_path[self.dev_id]            
            df = pd.read_csv(self.pathway,header=0)
            df = df[int(self.bias[subject][session]):-1]
            #            df = df[self.bias[subject][session]:-1]
            df = df[self.scale_label[subject][session][0]:self.scale_label[subject][session][1]]
#            print(df)
            #use apptime
#            values=df.as_matrix(columns=df.columns[0:1])
#            print(values)
            self.ts=Signal(values=df.as_matrix(columns=df.columns[1:2])/1000**self.scale[3])
#            print('the ts[0] is',self.ts.values[0])
            self.delta_ts=self.ts.copy()    
            self.delta_ts.values-=self.delta_ts.values[0]
            
            
            self.delta_ts=self.delta_ts.diff()
            self.delta_ts.values=np.append(self.delta_ts.values,0)
            self.ts=Signal(np.cumsum(self.delta_ts.values))

            raw=df.as_matrix(columns=df.columns[1:10])
            #                             0        1         2           3        4       5         6         7
            self.gyr_order=np.array([[2,1,-3],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])
#            self.gyr_order=np.array([[1,-2,-3],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])
#            self.gyr_order=np.array([[-1,2,-3],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])
#            self.gyr_order=np.array([[-2,-1,-3],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])

#            self.gyr_order=np.array([[-2,1,3],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]])  #北以acc x
#            self.gyr_order=np.array([[1,2,3],[1,-3,2],[-1,-3,-2],[-1,-3,-2],[3,2,-1],[2,1,-3],[-2,-3,1],[-2,-1,-3]]) #北以mag x

#        if fulltag:
#            file_tag=self.pathway.replace('./aiq_data/','')
#            file_tag=file_tag.replace('2018','').replace(file_tag[:5],'').replace('.csv','').replace('/','_')
#            self.file_tag='{0}_{1}_{2}_{3}_{4}_'.format(subject,session,file_tag[:4],file_tag[4:6],file_tag[6:8])
#        else:
#        if fulltag is None:
        try:
            print('filetag=',self.file_tag)
        except NameError:
            self.file_tag=''
            print('filetag=',self.file_tag)
        print('pathawy is ',self.file_tag)
        print('warning vector filter on??')
        self.sampleperiod=(((self.ts.values)[-1]-(self.ts.values)[0])/len(self.ts.values))
        self.samplerate=(1/self.sampleperiod)
        self.acc=Signal(values=raw[:,0:3]*self.scale[0],tag=self.file_tag+'acc',samplerate=self.samplerate)
        self.gyr=Signal(values=raw[:,3:6]*self.scale[1],tag=self.file_tag+'gyr',samplerate=self.samplerate)
        self.mag=Signal(values=raw[:,6:9]*self.scale[2],tag=self.file_tag+'mag',samplerate=self.samplerate)
#        self.acc.fft(100,30)
#        self.gyr.fft(100,40)
#        self.mag.fft(100,10)
        print(self.mag.values)
#        print('order is ',self.gyr_order[self.dev_id])
        print('warnging ned enu aiq mpu9250 change')
        if not date=='1218':
            self.acc_order,self.gyr_order,self.mag_order=self.nedadj(self.gyr_order[self.dev_id])
        else:
            self.acc_order,self.gyr_order,self.mag_order=self.nedadj_mpu9250(self.gyr_order[self.dev_id])
#            self.acc_order,self.gyr_order,self.mag_order=self.enuadj_mpu9250(self.gyr_order[self.dev_id])
#            print(self.acc_order)
#            print(self.gyr_order)
#            print(self.mag_order)
        self.acc=self.acc.changeorder(self.acc_order.tolist())
        self.gyr=self.gyr.changeorder(self.gyr_order.tolist())
        self.mag=self.mag.changeorder(self.mag_order.tolist())
        print('loading succeed')

        
    def lpf_filter(self,l3db=15,cali_mag=True):
        print("samplerate",self.samplerate)
        self.acc.values=vec_filter(self.acc.values,self.samplerate,l3db,None)
        self.gyr.values=vec_filter(self.gyr.values,self.samplerate,l3db,None)
        if cali_mag:
            self.mag.values=vec_filter(self.mag.values,self.samplerate,l3db,None)
#    def bandpf_filter(self,cali_gyr=True,cali_mag=True):
#        self.acc.values=vec_filter(self.acc.values,75,5)
#        self.gyr.values=vec_filter(self.gyr.values,75,3)
#        self.mag.values=vec_filter(self.mag.values,75,5)
class Debug():
    def __init__(self):
        pass
class Connector(IMU):
    def __init__(self,imu1,imu2,ImgJ=None,vicon=None):
        self.imu1=imu1
        self.imu2=imu2
        self.imu1_body=copy.deepcopy(imu1)
        self.samplerate=imu1.samplerate
        self.file_tag=imu1.file_tag
        self.position=imu1.position
        self.gyr=imu1.gyr
        if not ImgJ==None:
            self.ImgJ=ImgJ
        if not vicon==None:
            self.vicon=vicon
        print('create connnector complete')
        try:
            self.angle_between_imus()
        except:
            q1=self.imu1.qq.values
            q2=self.imu2.qq.values
            
            uu1=self.imu1.eu.values
            uu2=self.imu2.eu.values
            
#            xx1=self.imu1.theta.values
#            xx2=self.imu2.theta.values              
            final_length=min(len(q1),len(q2))
            print('ff',final_length)
            q1=q1[:final_length]
            q2=q2[:final_length]  
            uu1=uu1[:final_length]
            uu2=uu2[:final_length] 
#            xx1=xx1[:final_length]
#            xx2=xx2[:final_length]             
            print('length different happened')
            print(len(self.imu1.qq.values))
            print(len(q1))
            self.angle_between_imus(final_length)
#        try:
#            self.imu1_body.eu.values=q1-q2
#        except ValueError:
#            final_length=min(len(q1),len(q2))
#            q1=q1[:final_length]
#            q2=q2[:final_length]
#            print('length diffent happened')
#        k=np.array([0,1,2,3,7])
#        target=self.imu1.adoptor.session
#        if np.searchsorted(k,target)>=0:
#            self.imu1.eu.values=self.imu1.eu.values[:,[1,0,2]]
#            self.imu1.eu.values=-self.imu1.eu.values
#    def __init__(self,imu1,vicon):
##        vicon_out=-vicon.eu.values
##        imu_out=dev01.eu.values
##        test1.imu1.eu.values=imu_out
#        self.imu1=copy.deepcopy(imu1)
#        self.imu1_body=copy.deepcopy(imu1)
#        self.vicon=copy.deepcopy(vicon)       
#        self.vicon.eu.values[:,[1,2]]=-self.vicon.eu.values[:,[1,2]]
#        self.debug=[]
    def angle_between_imus(self,final_length):
        self.qq=Signal(np.zeros([final_length,4]),samplerate=self.samplerate)
        
        self.theta=Signal(np.zeros([final_length,1]),samplerate=self.samplerate)
        self.eu=Signal(np.zeros([final_length,3]),samplerate=self.samplerate)
        print('quaternion calculation to qeb...ing',len(self.imu1.qq.values))
        for i in range(final_length):
#            print(i)
#            tempq=Quaternion(self.imu1.qq.values[i])*(Quaternion(self.imu2.qq.values[i]).conjugate)
            
            tempq=Quaternion(self.imu1.qq.values[i])*(Quaternion(self.imu2.qq.values[i]).conj())
#            tempq=Quaternion(self.imu1.qq.values[i]).conj()*(Quaternion(self.imu2.qq.values[i]))

#            self.qq.values[i]=tempq.elements
            self.qq.values[i]=tempq.__array__()
            self.theta.values[i]=np.arccos(self.qq.values[i,0])*2/np.pi*180
#            self.eu.values[i]=tempq.eu_angle
            self.eu.values[i]=tempq.to_euler_angles_by_wiki()
        self.eu.values=self.eu.values/np.pi*180
        plt.show()
#        plt.plot(self.theta.values,label='theta')
        self.ts=np.linspace(0,self.imu1.sampleperiod*len(self.theta.values),len(self.theta.values))
        self.ts=Signal(self.ts)
        plt.plot(self.ts.values,self.theta.values,label='theta')
        plt.legend()
        plt.title('theta')
        plt.xlabel('ts')
        plt.ylabel('degree')
        
        plt.show()
#        self.theta=Signal(self.theta)
    def adj_duration(self,no):
        session=self.vicon.session
        subject=self.vicon.subject
       
        self.scale_label=np.array([[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[1,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]
                            ,[[0,-1],[0,-1],[0,-1]]])
        self.vicon_scale=self.scale_label*120
        self.imu_scale=self.imu_scale*self.imu1.samplerate
        self.imu1.ts.values=self.imu1.ts.values[self.imu_scale[subject][session][0]:self.imu_scale[subject][session][1]]
        self.imu1.eu.values=self.imu1.eu.values[self.imu_scale[subject][session][0]:self.imu_scale[subject][session][1]]
        self.vicon.ts.values=self.vicon.ts.values[self.vicon_scale[subject][session][0]:self.vicon_scale[subject][session][1]]
        self.vicon.eu.values=self.vicon.eu.values[self.vicon_scale[subject][session][0]:self.vicon_scale[subject][session][1]]
    def fit_degree(self,inin):
        self.last_err=[0,0,0]
        i_l=np.arange(-10, 10,1)
        j_l=np.arange(-1, 1,0.1)
        k_l=np.arange(-0.1, 0.1,0.01)
        self.answer=np.array([])
        for i in i_l:#i is t(s)
            print('i=',i)
            u=copy.deepcopy(self)
            u.adj_duration_start_end(0,-1,i,1,0,0)
            u.align_sig2(plt_err=False,index3=inin)
            self.answer=np.append(self.answer,u.rms_eu_s[inin])
            if len(self.answer)>3:
                if self.answer[-1]>np.mean(self.answer[-3:-1]):
                    break
        s1=i_l[np.argmin(self.answer)]
        self.s1=s1
        plt.show()
#        plt.plot(self.answer,label='1')
        self.answer=np.array([])
        for j in j_l:#i is t(s)
            print('j=',j)
            u=copy.deepcopy(self)
            u.adj_duration_start_end(0,-1,j+s1,1,0,0)
            u.align_sig2(plt_err=False,index3=inin)
            self.answer=np.append(self.answer,u.rms_eu_s[inin])
            if len(self.answer)>3:
                if self.answer[-1]>np.mean(self.answer[-3:-1]):
                    break
        s2=j_l[np.argmin(self.answer)]
        print(s1+s2)
        self.s2=s2
#        plt.plot(self.answer,label='0.1')
        self.answer=np.array([])
        for k in k_l:#i is t(s)
            print('k=',k)
            u=copy.deepcopy(self)
            u.adj_duration_start_end(0,-1,k+s1+s2,1,0,0)
            u.align_sig2(plt_err=False,index3=inin)
            self.answer=np.append(self.answer,u.rms_eu_s[inin])
            if len(self.answer)>3:
                if self.answer[-1]>np.mean(self.answer[-3:-1]):
                    break
        s3=k_l[np.argmin(self.answer)]
              
        plt.plot(self.answer,label='0.01')
        plt.legend()
        plt.show()
        self.s3=s3
        print('final answer=',s1+s2+s3)  
        self.adj_duration_start_end(0,-1,s1+s2+s3,1,0,0)
        self.align_sig2(plt_err=True,index3=inin)
    def fit_degree_xsens(self,inin):
        self.last_err=[0,0,0]
        i_l=np.arange(-10, 10,1)
        j_l=np.arange(-1, 1,0.1)
        k_l=np.arange(-0.1, 0.1,0.01)
        self.answer=np.array([])
        for i in i_l:#i is t(s)
            print('i=',i)
            u=copy.deepcopy(self)
            u.adj_duration_start_end(0,-1,i,1,0,0)
            u.align_sig_xsens(plt_err=False,index3=inin)
            self.answer=np.append(self.answer,u.rms_eu_s[inin])
            if len(self.answer)>3:
                if self.answer[-1]>np.mean(self.answer[-3:-1]):
                    break
        s1=i_l[np.argmin(self.answer)]
        self.s1=s1
        plt.show()
#        plt.plot(self.answer,label='1')
        self.answer=np.array([])
        for j in j_l:#i is t(s)
            print('j=',j)
            u=copy.deepcopy(self)
            u.adj_duration_start_end(0,-1,j+s1,1,0,0)
            u.align_sig_xsens(plt_err=False,index3=inin)
            self.answer=np.append(self.answer,u.rms_eu_s[inin])
            if len(self.answer)>3:
                if self.answer[-1]>np.mean(self.answer[-3:-1]):
                    break
        s2=j_l[np.argmin(self.answer)]
        print(s1+s2)
        self.s2=s2
#        plt.plot(self.answer,label='0.1')
        self.answer=np.array([])
        for k in k_l:#i is t(s)
            print('k=',k)
            u=copy.deepcopy(self)
            u.adj_duration_start_end(0,-1,k+s1+s2,1,0,0)
            u.align_sig_xsens(plt_err=False,index3=inin)
            self.answer=np.append(self.answer,u.rms_eu_s[inin])
            if len(self.answer)>3:
                if self.answer[-1]>np.mean(self.answer[-3:-1]):
                    break
        s3=k_l[np.argmin(self.answer)]
              
        plt.plot(self.answer,label='0.01')
        plt.legend()
        plt.show()
        self.s3=s3
        print('final answer=',s1+s2+s3)  
        self.adj_duration_start_end(0,-1,s1+s2+s3,1,0,0)
        self.align_sig_xsens(plt_err=True,index3=inin)
    def fit_time(self,inin):
        self.last_err=[0,0,0]
        i_l=np.arange(-5, 5,1)
        j_l=np.arange(-1, 1,0.1)
        k_l=np.arange(-0.1, 0.1,0.01)
        self.answer=np.array([])
        for i in i_l:#i is t(s)
            print('i=',i)
            u=copy.deepcopy(self)
            u.adj_duration_start_end(0,-1,0,1,0,i)
            u.align_sig2(plt_err=False,index3=inin)
            self.answer=np.append(self.answer,u.rms_eu_d[inin])
            if len(self.answer)>3:
                if self.answer[-1]>np.mean(self.answer[-3:-1]):
                    break
        s1=i_l[np.argmin(self.answer)]
        self.s1=s1
        plt.show()
#        plt.plot(self.answer,label='1')
        self.answer=np.array([])
        for j in j_l:#i is t(s)
            print('j=',j)
            u=copy.deepcopy(self)
            u.adj_duration_start_end(0,-1,0,1,0,j+s1)
            u.align_sig2(plt_err=False,index3=inin)
            self.answer=np.append(self.answer,u.rms_eu_d[inin])
            if len(self.answer)>3:
                if self.answer[-1]>np.mean(self.answer[-3:-1]):
                    break
        s2=j_l[np.argmin(self.answer)]
        print(s1+s2)
        self.s2=s2
#        plt.plot(self.answer,label='0.1')
        self.answer=np.array([])
        for k in k_l:#i is t(s)
            print('k=',k)
            u=copy.deepcopy(self)
            u.adj_duration_start_end(0,-1,0,1,0,k+s1+s2)
            u.align_sig2(plt_err=False,index3=inin)
            self.answer=np.append(self.answer,u.rms_eu_d[inin])
            if len(self.answer)>3:
                if self.answer[-1]>np.mean(self.answer[-3:-1]):
                    break
        s3=k_l[np.argmin(self.answer)]
              
        plt.plot(self.answer,label='0.01')
        plt.legend()
        plt.show()
        self.s3=s3
        print('final answer=',s1+s2+s3)  
        self.align_sig2(plt_err=True,index3=inin)
        self.adj_duration_start_end(0,-1,0,1,0,s1+s2+s3)
    def fit_time_xsens(self,inin):
        self.last_err=[0,0,0]
        i_l=np.arange(-5, 5,1)
        j_l=np.arange(-1, 1,0.1)
        k_l=np.arange(-0.1, 0.1,0.01)
        self.answer=np.array([])
        for i in i_l:#i is t(s)
#            print('i=',i)
            u=copy.deepcopy(self)
            u.adj_duration_start_end(0,-1,0,1,0,i)
            u.align_sig_xsens(plt_err=False,index3=inin)
            self.answer=np.append(self.answer,u.rms_eu_d[inin])
            if len(self.answer)>3:
                if self.answer[-1]>np.mean(self.answer[-3:-1]):
                    break
        s1=i_l[np.argmin(self.answer)]
        self.s1=s1
        plt.show()
#        plt.plot(self.answer,label='1')
        self.answer=np.array([])
        for j in j_l:#i is t(s)
#            print('j=',j)
            u=copy.deepcopy(self)
            u.adj_duration_start_end(0,-1,0,1,0,j+s1)
            u.align_sig_xsens(plt_err=False,index3=inin)
            self.answer=np.append(self.answer,u.rms_eu_d[inin])
            if len(self.answer)>3:
                if self.answer[-1]>np.mean(self.answer[-3:-1]):
                    break
        s2=j_l[np.argmin(self.answer)]
#        print(s1+s2)
        self.s2=s2
#        plt.plot(self.answer,label='0.1')
        self.answer=np.array([])
        for k in k_l:#i is t(s)
#            print('k=',k)
            u=copy.deepcopy(self)
            u.adj_duration_start_end(0,-1,0,1,0,k+s1+s2)
            u.align_sig_xsens(plt_err=False,index3=inin)
            self.answer=np.append(self.answer,u.rms_eu_d[inin])
            if len(self.answer)>3:
                if self.answer[-1]>np.mean(self.answer[-3:-1]):
                    break
        s3=k_l[np.argmin(self.answer)]
              
        plt.plot(self.answer,label='0.01')

        plt.legend()
        plt.show()
        self.s3=s3
#        print('final answer=',s1+s2+s3)  
        
        self.adj_duration_start_end(0,-1,0,1,0,s1+s2+s3)
        self.align_sig_xsens(plt_err=True,index3=inin)
    def adj_duration_start_end(self,start,end,imu_values,scale,vicon_b,imu_b):
#        vicon_start=int(110*(start+vicon_b))
#        imu_start=int(31*(start+imu_b))
        vicon_start=int(self.imu2.samplerate*(start))
        imu_start=int(self.imu1.samplerate*(start))
        vicon_end=int(self.imu2.samplerate*end)
        imu_end=int(self.imu1.samplerate*end)
#        print(vicon_start)
#        print(imu_start)
#        print(vicon_end)
#        print(imu_end)
        self.imu1.eu.values+=imu_values
        self.imu1.eu.values*=scale
        self.imu1.ts.values=self.imu1.ts.values[imu_start:imu_end]
        self.imu1.gyr.values=self.imu1.gyr.values[imu_start:imu_end]
        self.imu1.eu.values=self.imu1.eu.values[imu_start:imu_end]
#        self.vicon.ts.values=self.vicon.ts.values[vicon_start:vicon_end]
#        self.vicon.eu.values=self.vicon.eu.values[vicon_start:vicon_end]
#        self.vicon.ts.values-=vicon_b         
        self.imu2.ts.values=self.imu2.ts.values[vicon_start:vicon_end]
        self.imu2.eu.values=self.imu2.eu.values[vicon_start:vicon_end]
        self.imu2.ts.values-=vicon_b          
        
        self.imu1.mag.values=self.imu1.mag.values[imu_start:imu_end]
        self.imu1.ts.values-=imu_b
    def get_angle(self):
        print('calclating connector angle')

        for i in range(len(self.imu1.q.values)):
            q1=self.imu1.q.values
            q2=self.imu2.q.values
            qb=self.imu1_body.q.values
#            tempq=(Quaternion(q1[i,:]).conj())*Quaternion(q2[i,:])
            tempq=Quaternion(q1[i,:])*(Quaternion(q2[i,:]).conjugate)
            
            tempq=tempq.conjugate
            qb[i,:]=tempq._get_q()
            self.imu1_body.q.values[i,:]=tempq._get_q()
#            self.imu1_body.eu.values[i,0],self.imu1_body.eu.values[i,1],self.imu1_body.eu.values[i,2]=tempq.eu_angle
            self.imu1_body.eu.values[i,0],self.imu1_body.eu.values[i,1],self.imu1_body.eu.values[i,2]=tempq.to_euler_angles_by_wiki()
            
        self.imu1_body.eu.values=self.imu1_body.eu.values/np.pi*180
    def get_angle_minus(self):
        q1=self.imu1.eu.values
        q2=self.imu2.eu.values
        try:
            self.imu1_body.eu.values=q1-q2
        except ValueError:
            final_length=min(len(q1),len(q2))
            q1=q1[:final_length]
            q2=q2[:final_length]
            
            self.imu1_body.eu.values=q1-q2
            self.imu1_body.ts.values=self.imu1_body.ts.values[:final_length]
            print('warning operands could not be broadcast together with shapes....')
        print('calclating connector angle')
    def dehead(self,duration=None):
        #duration (s)
        if not duration==None:
            statictime=int(duration*self.samplerate)
            self.eu.values=self.eu.values[statictime:]
            self.theta.values=self.theta.values[statictime:]
            
            self.ts.values=self.ts.values[statictime:]
            self.ts.values-=self.ts.values[0]
            return
        self.bias=np.array([1,0,0.7,0.2,0.5,1.5,0.6,0.6,0.5,0.6,0.63,0.5,8,6.5,5,7])*31*60
        #有提醒
#        self.bias=np.array([6.5*2,6.3,5.3,4.8,5.5,4.46,6.4,4.3,4.5,6.4,5.37,12.5,6.7,7,7,7.5])*60
        #沒提醒
#        self.bias=np.array([3,6,4,9,8,3.5,4,4,4,4,7,7])*60
        
#        self.bias=np.array([[3,0,0,0,0,0],
#                   [6,0,0,0,0,0],
#                   [0,0,0,0,0,0],
#                   [0,0,0,0,0,0],
#                   [0,0,0,7.5,0,0],
#                   [0,0,0,3.5,0,0],#5
#                   [4,0,0,0,0,0],
#                   [4,0,0,0,0,0],
#                   [4,0,0,0,0,0],#8
#                   [0,0,0,4,0,0],
#                   [7,0,0,0,0,0],
#                   [0,0,0,0,0,0],
#                   [7,0,0,0,0,0],
#                   [0,0,0,0,0,0]])*60
        statictime=self.bias[self.imu1.adoptor.subject-1]
        print('start time ',self.bias[self.imu1.adoptor.subject-1])
#        statictime=self.bias[self.imu1.adoptor.subject,self.imu1.adoptor.session]
        statictime=int(statictime)
        self.imu1_body.eu.values=self.imu1_body.eu.values[statictime*31:]
        self.imu1_body.ts.values=self.imu1_body.ts.values[statictime*31:]
        self.imu1_body.ts.values-=self.imu1_body.ts.values[0]
    def detail(self,duration):
        #duration (s)
        if not duration==None:
            statictime=int(duration*self.samplerate)
            self.eu.values=self.eu.values[:-statictime*31]
            self.theta.values=self.theta.values[:-statictime*31]
            self.ts.values=self.ts.values[:-statictime*31]
            self.ts.values-=self.ts.values[0]
            return
        
    def cali_angle_zero(self,start,end,manual=None):
#            self.zeros=self.imu1_body.eu.values[0,0]
#            self.zeros=np.mean(self.imu1_body.eu.values[2000:2300,0])
        if manual is None:
            self.zeros=np.mean(self.imu1_body.eu.values[32*start:32*end,0])
        else:
            self.zeros=manual
        
        self.imu1_body.eu.values=self.imu1_body.eu.values-self.zeros
        print('self.zeros=',self.zeros)
#        else:
#            self.imu1_body.eu.values[:,0]-=zeros
        print('calibrating connector angle')    
    def plt_angle(self,index=None,cali_error=False,delay=0):
        angle_plt=Plot('title',dtype='eu')
        angle_plt.plot_body_angle(self,index=index,cali_error=cali_error,delay=delay)
    def compare_angle(self,index,showlegend=True):
        eu_table=['roll','pitch','yaw']
        angle_plt=Plot(str(self.imu1.table[self.imu1.id])+'_'+str(eu_table[self.vicon.session]),dtype='eu')
        temp_err=[self.rms_eu_s[index],self.rms_eu_d[index]]
        angle_plt.plot_compare_angle(self.imu1,self.vicon,index,temp_err,showlegend)
    def compare_angle_xsens(self,index,showlegend=True):
#        eu_table=['roll','pitch','yaw']
        eu_table=['pitch','roll','yaw']
        angle_plt=Plot(str(self.imu1.table[self.imu1.id])+'_'+str(eu_table[self.imu2.session]),dtype='eu')
        temp_err=[self.rms_eu_s[index],self.rms_eu_d[index]]
        angle_plt.plot_compare_angle(self.imu1,self.imu2,index,temp_err,showlegend)

    def align_sig(self,plt_err=False,index3=[0,1,2]):
        esti_gyr=self.imu1.gyr.values
        esti_eu=self.imu1.eu.values
        esti_ts=self.imu1.ts.values
        ref_ts=self.vicon.ts.values
        ref_eu=self.vicon.eu.values
#        plt.show()
#        plt.title('debug')
#        plt.plot(ref_ts,ref_eu[:,1])
#        plt.plot(esti_ts,esti_eu[:,1])
#        plt.show()
        mask=np.linalg.norm(esti_gyr,axis=1)>=(5)
        
#        plt.show()
#        plt.title('static_dynamic')
#        plt.plot(mask)
#        plt.plot(np.logical_not(mask))
#        plt.show()
#        err_eu1_s=np.zeros([len(esti_eu),1])
#        err_eu2_s=np.zeros([len(esti_eu),1])
#        err_eu3_s=np.zeros([len(esti_eu),1])
#        err_eu1_d=np.zeros([len(esti_eu),1])
#        err_eu2_d=np.zeros([len(esti_eu),1])
#        err_eu3_d=np.zeros([len(esti_eu),1])
        err_eu_d=np.zeros([len(esti_eu),3])
        err_eu_s=np.zeros([len(esti_eu),3])
    #    err_eu1_s,err_eu2_s,err_eu3_s,err_eu1_d,err_eu2_d,err_eu3_d=np.zeros_like(err_eu1_s)
        index2=np.empty([0],dtype=int)
        for i in range(len(esti_ts)):
            index2=np.append(index2,np.searchsorted(ref_ts,esti_ts[i]))
#            print(index2)
            if index2[-1]==len(ref_ts):
#                err_eu1_s[i]=0
#                err_eu2_s[i]=0
#                err_eu3_s[i]=0
#                err_eu1_d[i]=0
#                err_eu2_d[i]=0
#                err_eu3_d[i]=0
                err_eu_s[i]=0
                err_eu_d[i]=0
                index2[-1]=len(ref_ts)-1
            elif not(mask[i]):
#                err_eu1_s[i]=esti_eu[i,0]-ref_eu[index[-1],0]
#                err_eu2_s[i]=esti_eu[i,1]-ref_eu[index[-1],1]
#                err_eu3_s[i]=esti_eu[i,2]-ref_eu[index[-1],2]
#                print(esti_eu[i,0],ref_eu[index2[-1],0])
                err_eu_s[i]=esti_eu[i]-ref_eu[index2[-1]]
#                if err_eu_d[i,0]>10:
#                    print(i,esti_eu[i,0],ref_eu[index2[-1],0])
            else:
#                err_eu1_d[i]=esti_eu[i,0]-ref_eu[index[-1],0]
#                err_eu2_d[i]=esti_eu[i,1]-ref_eu[index[-1],1]
#                err_eu3_d[i]=esti_eu[i,2]-ref_eu[index[-1],2]
#                print(esti_eu[i,0],ref_eu[index2[-1],0])
                err_eu_d[i]=esti_eu[i]-ref_eu[index2[-1]]
#                if err_eu_d[i,0]>10:
#                    print(i,esti_eu[i,0],ref_eu[index2[-1],0])
                    
        if plt_err:
            plt.show()
            plt.plot(esti_ts,err_eu_d[:,index3],label='dynamic')
            plt.plot(esti_ts,err_eu_s[:,index3],label='static')
            plt.show()
        rms_eu_s = np.sqrt(np.mean(err_eu_s**2,axis=0))
        rms_eu_d = np.sqrt(np.mean(err_eu_d**2,axis=0))
        rms_eu_s=np.round(rms_eu_s,2)
        rms_eu_d=np.round(rms_eu_d,2)
        print('rms_eu_s',rms_eu_s) 
        print('rms_eu_d',rms_eu_d)  
        self.rms_eu_s=rms_eu_s
        self.rms_eu_d=rms_eu_d
    def align_sig_xsens(self,plt_err=False,index3=[0,1,2]):
        esti_gyr=self.imu1.gyr.values
        esti_eu=self.imu1.eu.values
        esti_ts=self.imu1.ts.values
        ref_ts=self.imu2.ts.values
        ref_eu=self.imu2.eu.values

        mask=np.linalg.norm(esti_gyr,axis=1)>=(5)

        err_eu_d=np.zeros([len(esti_eu),3])
        err_eu_s=np.zeros([len(esti_eu),3])
    #    err_eu1_s,err_eu2_s,err_eu3_s,err_eu1_d,err_eu2_d,err_eu3_d=np.zeros_like(err_eu1_s)
        index2=np.empty([0],dtype=int)
        for i in range(len(esti_ts)):
            index2=np.append(index2,np.searchsorted(ref_ts,esti_ts[i]))
#            print(index2)
            if index2[-1]==len(ref_ts):
#                err_eu1_s[i]=0
#                err_eu2_s[i]=0
#                err_eu3_s[i]=0
#                err_eu1_d[i]=0
#                err_eu2_d[i]=0
#                err_eu3_d[i]=0
                err_eu_s[i]=0
                err_eu_d[i]=0
                index2[-1]=len(ref_ts)-1
            elif not(mask[i]):
#                err_eu1_s[i]=esti_eu[i,0]-ref_eu[index[-1],0]
#                err_eu2_s[i]=esti_eu[i,1]-ref_eu[index[-1],1]
#                err_eu3_s[i]=esti_eu[i,2]-ref_eu[index[-1],2]
#                print(esti_eu[i,0],ref_eu[index2[-1],0])
                err_eu_s[i]=esti_eu[i]-ref_eu[index2[-1]]
#                if err_eu_d[i,0]>10:
#                    print(i,esti_eu[i,0],ref_eu[index2[-1],0])
            else:
#                err_eu1_d[i]=esti_eu[i,0]-ref_eu[index[-1],0]
#                err_eu2_d[i]=esti_eu[i,1]-ref_eu[index[-1],1]
#                err_eu3_d[i]=esti_eu[i,2]-ref_eu[index[-1],2]
#                print(esti_eu[i,0],ref_eu[index2[-1],0])
                err_eu_d[i]=esti_eu[i]-ref_eu[index2[-1]]
#                if err_eu_d[i,0]>10:
#                    print(i,esti_eu[i,0],ref_eu[index2[-1],0])
                    
        if plt_err:
            plt.show()
            plt.plot(esti_ts,err_eu_d[:,index3],label='dynamic')
            plt.plot(esti_ts,err_eu_s[:,index3],label='static')
            
            plt.legend()
            plt.show()
            plt.plot(self.imu1.ts.values,self.imu1.eu.values[:,inin],label='estimated')
            plt.plot(self.imu2.ts.values,self.imu2.eu.values[:,inin],label='ref')
            plt.legend()
            plt.show()
        
        rms_eu_s = np.sqrt(np.mean(err_eu_s**2,axis=0))
        rms_eu_d = np.sqrt(np.mean(err_eu_d**2,axis=0))
        rms_eu_s=np.round(rms_eu_s,2)
        rms_eu_d=np.round(rms_eu_d,2)
        
#        print('rms_eu_s',rms_eu_s) 
#        print('rms_eu_d',rms_eu_d)  
        self.rms_eu_s=rms_eu_s
        self.rms_eu_d=rms_eu_d
    def align_sig2(self,plt_err=False,index3=[0,1,2]):
        
        esti_gyr=self.imu1.gyr.values
        esti_eu=self.imu1.eu.values
        esti_ts=self.imu1.ts.values
        ref_ts=self.vicon.ts.values
        ref_eu=self.vicon.eu.values
#        plt.show()
#        plt.title('debug')
#        plt.plot(ref_ts,ref_eu[:,index3])
#        plt.plot(esti_ts,esti_eu[:,index3])
#        plt.show()
        mask=np.linalg.norm(esti_gyr,axis=1)>=(5)
#        plt.show()
#        plt.title('static_dynamic')
#        plt.plot(mask,label='dynamic')
#        plt.plot(np.logical_not(mask),label='static')
#        plt.show()
        err_eu_d=np.array([[]]).reshape(0,3)
        err_eu_s=np.array([[]]).reshape(0,3)
        dd=np.zeros([len(esti_eu),3])
        ss=np.zeros([len(esti_eu),3])
        index2=np.empty([0],dtype=int)
        print('mask.shape',mask.shape)
        print('esti_ts.shape',esti_ts.shape)
        print('ref_ts.shape',ref_ts.shape)
        s=[]
        d=[]
        for i in range(len(esti_ts)):
            index2=np.searchsorted(ref_ts,esti_ts[i])
            
            if (index2>=len(ref_ts)) | (index2==0):
                continue
            elif not(mask[i]):
                temp=(esti_eu[i]-ref_eu[index2]).reshape(1,3)
                ss[i]=esti_eu[i]-ref_eu[index2]
                err_eu_s=np.append(err_eu_s,temp,axis=0)
#                s.append(i)
            else:
                temp=(esti_eu[i]-ref_eu[index2]).reshape(1,3)
                err_eu_d=np.append(err_eu_d,temp,axis=0)
                dd[i]=esti_eu[i]-ref_eu[index2]
#                d.append(i)
        if plt_err:
            plt.show()
            plt.plot(esti_ts,dd[:,index3],label='dynamic')
            plt.plot(esti_ts,ss[:,index3],label='static')
            plt.legend()
            plt.show()
        rms_eu_s = np.sqrt(np.mean(err_eu_s**2,axis=0))
        rms_eu_d = np.sqrt(np.mean(err_eu_d**2,axis=0))
#        rms_eu_s = np.sqrt(np.mean(ss[s,:]**2,axis=0))
#        rms_eu_d = np.sqrt(np.mean(dd[d,:]**2,axis=0))
        rms_eu_s=np.round(rms_eu_s,2)
        rms_eu_d=np.round(rms_eu_d,2)
        print('rms_eu_s',rms_eu_s) 
        print('rms_eu_d',rms_eu_d)  
        self.rms_eu_s=rms_eu_s
        self.rms_eu_d=rms_eu_d


if __name__=='__main__':
##    Reader.figure(path='./raw/20190227091323_3.csv',shape=[10000,11],specific=[1,0])
#    Reader.figure(path='./raw/20190227091323_3.csv',specific=[1,0])
#    raw=Reader.export()
    
    print('qqqqqqqqq')
    print(__package__)
    a=__package__
    pass
    
#import subprocess
#subprocess.call(['ls'])
#subprocess.run(["ls", "-l"])
#import os
#os.
#y=os.system('ls')
#os.system("some_command with args")
