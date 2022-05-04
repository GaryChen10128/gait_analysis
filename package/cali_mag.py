# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 19:20:11 2018

@author: 180218
"""
import numpy as np
import matplotlib.pyplot as plt
def cali_mag(mag,esti_mag):
    offset_x=(np.max(mag[:,0])+np.min(mag[:,0]))/2
    offset_y=(np.max(mag[:,1])+np.min(mag[:,1]))/2
    offset_z=(np.max(mag[:,2])+np.min(mag[:,2]))/2
    r_mag_x=(np.max(mag[:,0])-offset_x)/2
    r_mag_y=(np.max(mag[:,1])-offset_y)/2
    r_mag_z=(np.max(mag[:,2])-offset_z)/2
    r_min_x=(np.min(mag[:,0])-offset_x)/2
    r_min_y=(np.min(mag[:,1])-offset_y)/2
    r_min_z=(np.min(mag[:,2])-offset_z)/2
    avg_x=(r_mag_x-r_min_x)/2
    avg_y=(r_mag_y-r_min_y)/2
    avg_z=(r_mag_z-r_min_z)/2
    avg_radius=(avg_x+avg_y+avg_z)/2
    scale_x=avg_radius/avg_x
    scale_y=avg_radius/avg_y
    scale_z=avg_radius/avg_z
    cali_result=plt.figure(figsize=(5,5))

    cali_result=cali_result.add_subplot(111)
    cali_result.set_ylim([-100,100])
    cali_result.set_xlim([-100,100])
    mag[:,0]=(mag[:,0]-offset_x)/scale_x
    mag[:,1]=(mag[:,1]-offset_y)/scale_y
    mag[:,2]=(mag[:,2]-offset_z)/scale_z
    cali_result.scatter(mag[:,0],mag[:,1])
    cali_result.scatter(mag[:,1],mag[:,2])
    cali_result.scatter(mag[:,0],mag[:,2]) 
#    plt.scatter(mag[:,0],mag[:,1])
#    plt.scatter(mag[:,1],mag[:,2])
#    plt.scatter(mag[:,0],mag[:,2]) 
#    plt.grid()
#    cali_result.grid()
    print(offset_x,offset_y,offset_z,scale_x,scale_y,scale_z)
#    plt.show()
    esti_mag[:,0]=(esti_mag[:,0]-offset_x)/scale_x
    esti_mag[:,1]=(esti_mag[:,1]-offset_y)/scale_y
    esti_mag[:,2]=(esti_mag[:,2]-offset_z)/scale_z
    return esti_mag