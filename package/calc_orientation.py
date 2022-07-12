# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:28:25 2018

@author: 180218
"""
import warnings 

from .madgwickahrs import MadgwickAHRS
from .quaternion import Quaternion
#from pyquaternion import Quaternion

import numpy as np
from .madgwick_c import madgwickfilter
#from madgwick_c import showbias
def changeorder_mag(data):
    data[:,:]=data[:,[1,0,2]]
#    data=data[:,[1,0,2]]
    data[:,2]=-data[:,2]
def changeorder(gyr,acc,mag):#https://github.com/kriswiner/MPU9250/issues/272
    gyr[:,2]=-gyr[:,2]
    gyr[:,1]=-gyr[:,1]
    acc[:,0]=-acc[:,0]
    mag[:,:]=mag[:,[1,0,2]]
    mag[:,1]=-mag[:,1]
def changeorder3(gyr,acc,mag):#https://github.com/kriswiner/MPU9250/issues/272
    gyr[:,2]=-gyr[:,2]
    gyr[:,1]=-gyr[:,1]
    acc[:,0]=-acc[:,0]
    mag[:,:]=mag[:,[1,0,2]]
    mag[:,1]=-mag[:,1]
def changeorder2(gyr,acc):
    gyr[:,2]=-gyr[:,2]
    gyr[:,1]=-gyr[:,1]
    acc[:,0]=-acc[:,0]
def changeorder5(gyr,acc):
    gyr[:,:]=gyr[:,[0,2,1]]
    acc[:,:]=acc[:,[0,2,1]]
#    acc[:,2]=-acc[:,2]
    gyr[:,2]=-gyr[:,2]
    acc[:,1]=-acc[:,1]
    acc[:,0]=-acc[:,0]
def changeorder8(gyr,acc,mag=None):
    acc[:,2]=-acc[:,2]
    acc[:,:]=acc[:,[2,0,1]]
    
    gyr[:,0]=-gyr[:,0]
    gyr[:,1]=-gyr[:,1]
    gyr[:,:]=gyr[:,[2,0,1]]
    if mag is not None:
        mag[:,0]=-mag[:,0]
        mag[:,1]=-mag[:,1]
        mag[:,2]=-mag[:,2]
        mag[:,:]=mag[:,[2,1,0]]
def calc_orientation(gyr,acc,samplerate,align_arr,beta,delta_t,times):
    ref_qq=np.zeros([len(acc),4])
    ref_eu=np.zeros([len(acc),3])
    sampleperiod=1/samplerate
    qq = Quaternion(align_arr)
    madgwick=MadgwickAHRS(sampleperiod=sampleperiod,quaternion=qq,beta=beta)
    count=0
    ll=len(gyr)
    for i in range(0,len(gyr)-1):
        if (i%300)==0:
            print(str(i)+' / '+str(ll)+'  '+str(round(i/ll*100,0))+'%')
        while(count<times):
            madgwick.update_imu(gyr[i],acc[i],delta_t[i])
            count+=1
        count=0
#        ref_qq[i]=madgwick.quaternion.elements
#        ref_eu[i]=madgwick.quaternion.eu_angle
        ref_qq[i,0]=madgwick.quaternion._get_q()[0]
        ref_qq[i,1]=madgwick.quaternion._get_q()[1]
        ref_qq[i,2]=madgwick.quaternion._get_q()[2]
        ref_qq[i,3]=madgwick.quaternion._get_q()[3]
        tempq=Quaternion(ref_qq[i])
        ref_eu[i,:]=tempq.to_euler_angles_by_wiki()
        ref_eu[i]*=180/np.pi
        
    return ref_qq,ref_eu

def calc_orientation_mag(gyr,acc,mag,samplerate,align_arr,beta,delta_t,times):
    ref_qq=np.zeros([len(acc),4])
    ref_eu=np.zeros([len(acc),3])
#    out_qq=np.zeros([len(acc),4])
#    out_eu=np.zeros([len(acc),3])
    sampleperiod=1/samplerate
    qq = align_arr
    madgwick=MadgwickAHRS(sampleperiod=sampleperiod,quaternion=qq,beta=beta)
    rms_mag=np.zeros(len(mag))
    for i in range(len(mag)):
        rms_mag[i]=np.sqrt(np.mean(mag[i,0]**2+mag[i,1]**2+mag[i,2]**2))
    count=0
    ll=len(gyr)
    lastMag=0
    for i in range(0,len(gyr)-1):
        if (i%2000)==0:
            print(str(i)+' / '+str(ll)+'  '+str(round(i/ll*100,1))+'%')
        warnings.warn('mag maxium limit is on')
        if(rms_mag[i]>999):
            print('warning!! magnetometer warning')
            warnings.warn('rms_mag[i]>maxium')
            while(count<times):
                madgwick.update_imu(gyr[i],acc[i],delta_t[i])
                count+=1
            count=0
        else:
            while(count<times):
                if(mag[i][0]!=lastMag):
                    madgwick.update(gyr[i],acc[i],mag[i],delta_t[i])
                    lastMag=mag[i][0]
                else:
                    madgwick.update_imu(gyr[i],acc[i],delta_t[i])
                
                count+=1
            count=0
        ref_qq[i,0]=madgwick.quaternion._get_q()[0]
        ref_qq[i,1]=madgwick.quaternion._get_q()[1]
        ref_qq[i,2]=madgwick.quaternion._get_q()[2]
        ref_qq[i,3]=madgwick.quaternion._get_q()[3]
#        ref_qq[i]=madgwick.quaternion.elements
       
        tempq=Quaternion(ref_qq[i])
#        
        ref_eu[i,:]=tempq.to_euler_angles_by_wiki()
        ref_eu[i]*=180/np.pi
#        ref_eu[i]=madgwick.quaternion.eu_angle
#        
        
    return ref_qq,ref_eu
def calc_orientation_c(gyr,acc,mag,samplerate,align_arr,gyroMeasError,gyroMeasDrift,delta_t,times):
    ref_qq=np.zeros([len(acc),4])
    ref_eu=np.zeros([len(acc),3])
    out_qq=np.zeros([len(acc),4])
    out_eu=np.zeros([len(acc),3])
    sampleperiod=1/samplerate
    qq = [1,0,0,0]
#    sampleperiod,gyroMeasError,gyroMeasDrift,quaternion
    a=gyroMeasError/180*np.pi
    b=gyroMeasDrift/180*np.pi
    madgwick=madgwickfilter(sampleperiod=31,gyroMeasError=a,gyroMeasDrift=b,quaternion=qq)
    rms_mag=np.zeros(len(mag))
    for i in range(len(mag)):
        rms_mag[i]=np.sqrt(np.mean(mag[i,0]**2+mag[i,1]**2+mag[i,2]**2))
    count=0
    for i in range(0,len(gyr)-1):
        if(rms_mag[i]>70):
            while(count<times):
                madgwick.IMUfilterUpdate2(gyr[i],acc[i],delta_t[i])
                count+=1
            count=0
        else:
            while(count<times):
                madgwick.MARGfilterUpdate2(gyr[i],acc[i],mag[i],delta_t[i])
                count+=1
            count=0
        ref_qq[i,0]=madgwick.SEq_1
        ref_qq[i,1]=madgwick.SEq_2
        ref_qq[i,2]=madgwick.SEq_3
        ref_qq[i,3]=madgwick.SEq_4
        tempq=Quaternion(ref_qq[i])
        ref_eu[i,:]=tempq.to_euler_angles_by_wiki()
#        temp_q=Quaternion(madgwick.quaternion._get_q())
#        cq=Quaternion(0,0,0,1).conj() 
#        end_q=cq*temp_q
#        out_qq[i,0]=end_q[0]
#        out_qq[i,1]=end_q[1]
#        out_qq[i,2]=end_q[2]
#        out_qq[i,3]=end_q[3]
#        out_eu[i,:]=end_q.to_euler_angles_by_wiki()
        ref_eu[i]=ref_eu[i]*180/np.pi
#        out_eu[i]*=180/np.pi
#    madgwick.showbias()
    return ref_qq,ref_eu,ref_qq,ref_eu
#def showbias():
#    self.showbias()