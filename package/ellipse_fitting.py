# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 11:33:31 2018

@author: 180218
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 29 14:02:44 2018

@author: 180218
"""
#import warnings
#warnings.filterwarnings("error")
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
import copy
#介紹
#程式會讀取path路徑檔案(line:21)，計算該檔案磁力計訊號，校正磁力計參數
#使用方法 
#方法引數esti_mag為待校正參數 (line:20)，為長度l*3的矩陣，l是資料長度)
#iid是要校正的儀器編號
#其它注意事項
#此方法引數為傳址呼叫，所以不用return
def ellipse_fitting2(mag,esti_mag):
#    path='./1207elipse_fitting/20181218112554.csv'
#    df = pd.read_csv(path, index_col=False)
#    mag=df.as_matrix(columns=df.columns[7:10])

#    mag=np.array([Magx,Magy,Magz]).transpose()
#    esti_mag/=norm(esti_mag)
#    mag/=norm(mag)
    
#    
#    mean=np.mean(mag,axis=0)
#    std=np.std(mag,axis=0)
#    threshold=mean+2*std
#    threshold2=mean-2*std
#    print('threshold',threshold)
#    m1=mag[:,0]<threshold[0]
#    m2=mag[:,1]<threshold[1]
#    m3=mag[:,2]<threshold[2]
#    m4=mag[:,0]>threshold2[0]
#    m5=mag[:,1]>threshold2[1]
#    m6=mag[:,2]>threshold2[2]
#    m=np.logical_and(m1,m2)
#    m=np.logical_and(m,m3)
#    m=np.logical_and(m,m4)
#    m=np.logical_and(m,m5)
#    m=np.logical_and(m,m6)
#    mag=mag[m]
#    print('mag.shape',mag.shape)
##    
    
    
    
    print('mag calibrated')
    arr=mag
    D1=np.array([arr[:,0]**2])
    D2=np.array([arr[:,1]**2])
    D3=np.array([arr[:,2]**2])
    D4=np.array([arr[:,0]*arr[:,1]])
    D5=np.array([arr[:,0]*arr[:,2]])
    D6=np.array([arr[:,1]*arr[:,2]])
    D7=np.array([arr[:,0]])
    D8=np.array([arr[:,1]])
    D9=np.array([arr[:,2]])
    D=np.concatenate((D1,D2,D3,D4,D5,D6,D7,D8,D9)).transpose()
#    print('D',D.shape)
    a=np.linalg.inv(np.dot(np.transpose(D),D))
    a=np.dot(a,np.transpose(D))
    a=np.dot(a,np.ones(len(mag)))
    M=np.array([[a[0],a[3]/2,a[4]/2],[a[3]/2,a[1],a[5]/2],[a[4]/2,a[5]/2,a[2]]])
    center=-0.5*np.array([a[6],a[7],a[8]])
    #center=np.transpose(center)
    center=np.dot(center,np.linalg.inv(M))
    SS=np.dot(center,M)
    SS=np.dot(SS,np.transpose(center))+1
    V,U=np.linalg.eig(M)
    n1=V[0]
    n2=V[1]
    n3=V[2]
    np.diagflat([V[0],V[1],V[2]], k=0)
    scale_axis=np.array([(SS/n1)**0.5,(SS/n2)**0.5,(SS/n3)**0.5])
    
    print('center=\n',center)
    print('scale_axis=\n',scale_axis)
    circle_x=np.linspace(-1,1,100)
    #circle_y=np.linspace(1,0,50)
    circle_y=(1-circle_x**2)**0.5
    circle_x=np.concatenate((circle_x,circle_x[::-1]))
    circle_y=np.concatenate((circle_y,-circle_y))
#    circle_z=(1-(circle_x**2+circle_y**2))**0.5
    
    aim=np.linspace(-np.max(arr),np.max(arr),100)
    zeros=np.zeros(len(aim))
    fig = plt.figure(figsize=(10,25))
    ax = fig.add_subplot(411, projection='3d')
    
    ax.scatter(arr[:,0]/scale_axis[0], arr[:,1]/scale_axis[1], arr[:,2]/scale_axis[2], c='r', marker='o',label='uncalibrated')
    ax.set_xlabel('x_axis')
    ax.set_ylabel('y_axis')
    ax.set_zlabel('z_axis')
    ax.set_title('x_y_z sphere', fontsize=20)
    ax.scatter((arr[:,0]-center[0])/scale_axis[0], (arr[:,1]-center[1])/scale_axis[1], (arr[:,2]-center[2])/scale_axis[2], c='b', marker='^',label='calibrated')
    ax.grid()
    ax.legend()
    esti_mag[:,0]=(esti_mag[:,0]-center[0])/scale_axis[0]
    esti_mag[:,1]=(esti_mag[:,1]-center[1])/scale_axis[1]
    esti_mag[:,2]=(esti_mag[:,2]-center[2])/scale_axis[2]
    
    ax2 = fig.add_subplot(412)
    ax2.scatter(arr[:,0]/scale_axis[0], arr[:,1]/scale_axis[1], c='r',label='uncalibrated')
    ax2.set_xlabel('x_axis')
    ax2.set_ylabel('y_axis')
    ax2.set_title('x_y plane', fontsize=20)
    ax2.plot(aim,zeros,color='black')
    ax2.plot(zeros,aim,color='black')
    ax2.scatter((arr[:,0]-center[0])/scale_axis[0], (arr[:,1]-center[1])/scale_axis[1], c='b',label='calibrated')
    ax2.axis([-1.5,1.5,-1.5,1.5])
#    ax2.plot(circle_x,circle_y, c='g',label='calibrated')
    ax2.grid()
    ax2.legend()
    
    
    ax3 = fig.add_subplot(413)
    ax3.set_title('y_z plane', fontsize=20)
    ax3.scatter(arr[:,1]/scale_axis[1], arr[:,2]/scale_axis[2], c='r',label='uncalibrated')
    ax3.set_xlabel('y_axis')
    ax3.set_ylabel('z_axis')
    ax3.plot(aim,zeros,color='black')
    ax3.plot(zeros,aim,color='black')
    ax3.scatter((arr[:,1]-center[1])/scale_axis[1], (arr[:,2]-center[2])/scale_axis[2], c='b',label='calibrated')
    ax3.grid()
    ax3.axis([-1.5,1.5,-1.5,1.5])
    ax3.legend()
    ax4 = fig.add_subplot(414)
    
    ax4.set_title('x_z plane', fontsize=20)
    ax4.plot(aim,zeros,color='black')
    ax4.plot(zeros,aim,color='black')
    ax4.scatter(arr[:,0]/scale_axis[0], arr[:,2]/scale_axis[2], c='r',label='uncalibrated')
    ax4.axis([-1.5,1.5,-1.5,1.5])
    ax4.scatter((arr[:,0]-center[0])/scale_axis[0], (arr[:,2]-center[2])/scale_axis[2], c='b',label='calibrated')
    ax4.plot(aim,zeros,color='black')
    ax4.plot(zeros,aim,color='black')
    ax4.set_xlabel('x_axis')
    ax4.set_ylabel('z_axis')
    ax4.grid()
    ax4.legend()
#    plt.show()
    arr=(arr-center)/scale_axis
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('calibration result')

    ax.scatter(arr[:,0],arr[:,1])
    ax.scatter(arr[:,1],arr[:,2])
    ax.scatter(arr[:,0],arr[:,2])
    ax.grid()
#    ax.set_xlim(-1.7,1.7)
#    ax.set_xscale(-1.5,1.5)
    ax.axis([-1.5,1.5,-1.5,1.5])
    
    return center,scale_axis
#    ax.show()
    # return esti_mag
def ellipse_fitting(esti_mag,path,mag):
#    path='./1207elipse_fitting/20181218112554.csv'
    df = pd.read_csv(path, index_col=False,header=None)
#    mag=df.as_matrix(columns=df.columns[7:10])
    
#    raw=df.as_matrix(columns=df.columns[:])
#    mag=df.as_matrix(columns=df.columns[8:11])
#    mag=mag[raw[:,1]==0]
    
#    mag=np.array([Magx,Magy,Magz]).transpose()
#    esti_mag/=norm(esti_mag)
#    mag/=norm(mag)
    print('mag calibrated')
    arr=mag
    D1=np.array([arr[:,0]**2])
    D2=np.array([arr[:,1]**2])
    D3=np.array([arr[:,2]**2])
    D4=np.array([arr[:,0]*arr[:,1]])
    D5=np.array([arr[:,0]*arr[:,2]])
    D6=np.array([arr[:,1]*arr[:,2]])
    D7=np.array([arr[:,0]])
    D8=np.array([arr[:,1]])
    D9=np.array([arr[:,2]])
    D=np.concatenate((D1,D2,D3,D4,D5,D6,D7,D8,D9))
    
    print(D.shape)
    test=np.dot(D,np.transpose(D))
    print(test.shape)
    a=np.linalg.inv(np.dot(D,np.transpose(D)))
        

    a=np.dot(a,D)
    print('start')
    print(a.shape)
    a=np.dot(a,np.ones(len(mag)))
    print(a.shape)

    print('a.shape',a.shape)
    M=np.array([[a[0],a[3]/2,a[4]/2],[a[3]/2,a[1],a[5]/2],[a[4]/2,a[5]/2,a[2]]])
    center=-0.5*np.array([a[6],a[7],a[8]])
    #center=np.transpose(center)
    center=np.dot(center,np.linalg.inv(M))
    print('center.shape',center.shape)
    SS=np.dot(center,M)
    SS=np.dot(SS,np.transpose(center))+1
    V,U=np.linalg.eig(M)
    
    n1=V[0]
    n2=V[1]
    n3=V[2]
    print(n1,n2,n3)
    print(U)
    
    np.diagflat([V[0],V[1],V[2]], k=0)
    scale_axis=np.array([(SS/n1)**0.5,(SS/n2)**0.5,(SS/n3)**0.5])
    
    print('center=\n',center)
    print('scale_axis=\n',scale_axis)
    circle_x=np.linspace(-1,1,100)
    #circle_y=np.linspace(1,0,50)
    circle_y=(1-circle_x**2)**0.5
    circle_x=np.concatenate((circle_x,circle_x[::-1]))
    circle_y=np.concatenate((circle_y,-circle_y))
#    circle_z=(1-(circle_x**2+circle_y**2))**0.5
    
    aim=np.linspace(-np.max(arr),np.max(arr),100)
    zeros=np.zeros(len(aim))
    fig = plt.figure(figsize=(10,25))
    ax = fig.add_subplot(411, projection='3d')
    
    ax.scatter(arr[:,0]/scale_axis[0], arr[:,1]/scale_axis[1], arr[:,2]/scale_axis[2], c='r', marker='o',label='uncalibrated')
    ax.set_xlabel('x_axis')
    ax.set_ylabel('y_axis')
    ax.set_zlabel('z_axis')
    ax.set_title('x_y_z sphere', fontsize=20)
    ax.scatter((arr[:,0]-center[0])/scale_axis[0], (arr[:,1]-center[1])/scale_axis[1], (arr[:,2]-center[2])/scale_axis[2], c='b', marker='^',label='calibrated')
    ax.grid()
    ax.legend()
#    esti_mag[:,0]=(esti_mag[:,0]-center[0])/scale_axis[0]
#    esti_mag[:,1]=(esti_mag[:,1]-center[1])/scale_axis[1]
#    esti_mag[:,2]=(esti_mag[:,2]-center[2])/scale_axis[2]
    
    ax2 = fig.add_subplot(412)
    ax2.scatter(arr[:,0]/scale_axis[0], arr[:,1]/scale_axis[1], c='r',label='uncalibrated')
    ax2.set_xlabel('x_axis')
    ax2.set_ylabel('y_axis')
    ax2.set_title('x_y plane', fontsize=20)
    ax2.plot(aim,zeros,color='black')
    ax2.plot(zeros,aim,color='black')
    ax2.scatter((arr[:,0]-center[0])/scale_axis[0], (arr[:,1]-center[1])/scale_axis[1], c='b',label='calibrated')
#    ax2.axis([-1.5,1.5,-1.5,1.5])
    ax2.axis([-3.5,3.5,-3.5,3.5])
#    ax2.plot(circle_x,circle_y, c='g',label='calibrated')
    ax2.grid()
    ax2.legend()
    
    
    ax3 = fig.add_subplot(413)
    ax3.set_title('y_z plane', fontsize=20)
    ax3.scatter(arr[:,1]/scale_axis[1], arr[:,2]/scale_axis[2], c='r',label='uncalibrated')
    ax3.set_xlabel('y_axis')
    ax3.set_ylabel('z_axis')
    ax3.plot(aim,zeros,color='black')
    ax3.plot(zeros,aim,color='black')
    ax3.scatter((arr[:,1]-center[1])/scale_axis[1], (arr[:,2]-center[2])/scale_axis[2], c='b',label='calibrated')
    ax3.grid()
    #    ax3.axis([-1.5,1.5,-1.5,1.5])
    ax3.axis([-3.5,3.5,-3.5,3.5])
    ax3.legend()
    ax4 = fig.add_subplot(414)
    ax4.set_title('x_z plane', fontsize=20)
    ax4.plot(aim,zeros,color='black')
    ax4.plot(zeros,aim,color='black')
    ax4.scatter(arr[:,0]/scale_axis[0], arr[:,2]/scale_axis[2], c='r',label='uncalibrated')
    #    ax4.axis([-1.5,1.5,-1.5,1.5])
    ax4.axis([-3.5,3.5,-3.5,3.5])
    ax4.scatter((arr[:,0]-center[0])/scale_axis[0], (arr[:,2]-center[2])/scale_axis[2], c='b',label='calibrated')
    ax4.plot(aim,zeros,color='black')
    ax4.plot(zeros,aim,color='black')
    ax4.set_xlabel('x_axis')
    ax4.set_ylabel('z_axis')
    ax4.grid()
    ax4.legend()
    plt.show()
    esti_mag=(esti_mag-center)/scale_axis
    return esti_mag
#    return center,scale_axis
def ellipse_fitting_aiq_without_t(esti_mag,path,devid):
#    path='./1207elipse_fitting/20181218112554.csv'
    df = pd.read_csv(path, index_col=False,header=None)
#    mag=df.as_matrix(columns=df.columns[7:10])
    
    raw=df.as_matrix(columns=df.columns[:])
    mag=df.as_matrix(columns=df.columns[8:11])
    mag=mag[raw[:,1]==devid]
    
#    err=np.logical_and(mag[:,1]>(-20),mag[:,2]<(30))
#    mag=mag[err]
    
    mean=np.mean(mag,axis=0)
    std=np.std(mag,axis=0)
    threshold=mean+2*std
    threshold2=mean-2*std
    print('threshold',threshold)
    m1=mag[:,0]<threshold[0]
    m2=mag[:,1]<threshold[1]
    m3=mag[:,2]<threshold[2]
    m4=mag[:,0]>threshold2[0]
    m5=mag[:,1]>threshold2[1]
    m6=mag[:,2]>threshold2[2]
    m=np.logical_and(m1,m2)
    m=np.logical_and(m,m3)
    m=np.logical_and(m,m4)
    m=np.logical_and(m,m5)
    m=np.logical_and(m,m6)
    mag=mag[m]
    print('mag.shape',mag.shape)
#    
#    mag=np.array([Magx,Magy,Magz]).transpose()
#    esti_mag/=norm(esti_mag)
#    mag/=norm(mag)
    print('mag calibrated')
    arr=mag
    D1=np.array([arr[:,0]**2])
    D2=np.array([arr[:,1]**2])
    D3=np.array([arr[:,2]**2])
    D4=np.array([arr[:,0]*arr[:,1]])
    D5=np.array([arr[:,0]*arr[:,2]])
    D6=np.array([arr[:,1]*arr[:,2]])
    D7=np.array([arr[:,0]])
    D8=np.array([arr[:,1]])
    D9=np.array([arr[:,2]])
    D=np.concatenate((D1,D2,D3,D4,D5,D6,D7,D8,D9))
    
    print(D.shape)
    test=np.dot(D,np.transpose(D))
    print(test.shape)
    a=np.linalg.inv(np.dot(D,np.transpose(D)))
        

    a=np.dot(a,D)
    print('start')
    print(a.shape)
    a=np.dot(a,np.ones(len(mag)))
    print(a.shape)

    print('a.shape',a.shape)
    M=np.array([[a[0],a[3]/2,a[4]/2],[a[3]/2,a[1],a[5]/2],[a[4]/2,a[5]/2,a[2]]])
    center=-0.5*np.array([a[6],a[7],a[8]])
    #center=np.transpose(center)
    center=np.dot(center,np.linalg.inv(M))
    print('center.shape',center.shape)
    SS=np.dot(center,M)
    SS=np.dot(SS,np.transpose(center))+1
    V,U=np.linalg.eig(M)
    
    n1=V[0]
    n2=V[1]
    n3=V[2]
    print(n1,n2,n3)
    print(U)
    
    np.diagflat([V[0],V[1],V[2]], k=0)
    scale_axis=np.array([(SS/n1)**0.5,(SS/n2)**0.5,(SS/n3)**0.5])

#    try:
#        
#        scale_axis=np.array([(SS/n1)**0.5,(SS/n2)**0.5,(SS/n3)**0.5])
#    except RuntimeWarning:
#        print('catch runtimewarning')
#    print('SS',SS)
#    print('n',n1,n2,n3)
    print('center=\n',center)
    print('scale_axis=\n',scale_axis)
    circle_x=np.linspace(-1,1,100)
    #circle_y=np.linspace(1,0,50)
    circle_y=(1-circle_x**2)**0.5
    circle_x=np.concatenate((circle_x,circle_x[::-1]))
    circle_y=np.concatenate((circle_y,-circle_y))
#    circle_z=(1-(circle_x**2+circle_y**2))**0.5
    
    aim=np.linspace(-np.max(arr),np.max(arr),100)
    zeros=np.zeros(len(aim))
    fig = plt.figure(figsize=(10,25))
    ax = fig.add_subplot(411, projection='3d')
    
    ax.scatter(arr[:,0]/scale_axis[0], arr[:,1]/scale_axis[1], arr[:,2]/scale_axis[2], c='r', marker='o',label='uncalibrated')
    ax.set_xlabel('x_axis')
    ax.set_ylabel('y_axis')
    ax.set_zlabel('z_axis')
    ax.set_title('x_y_z sphere', fontsize=20)
    ax.scatter((arr[:,0]-center[0])/scale_axis[0], (arr[:,1]-center[1])/scale_axis[1], (arr[:,2]-center[2])/scale_axis[2], c='b', marker='^',label='calibrated')
    ax.grid()
    ax.legend()
#    esti_mag[:,0]=(esti_mag[:,0]-center[0])/scale_axis[0]
#    esti_mag[:,1]=(esti_mag[:,1]-center[1])/scale_axis[1]
#    esti_mag[:,2]=(esti_mag[:,2]-center[2])/scale_axis[2]
    
    ax2 = fig.add_subplot(412)
    ax2.scatter(arr[:,0]/scale_axis[0], arr[:,1]/scale_axis[1], c='r',label='uncalibrated')
    ax2.set_xlabel('x_axis')
    ax2.set_ylabel('y_axis')
    ax2.set_title('x_y plane', fontsize=20)
    ax2.plot(aim,zeros,color='black')
    ax2.plot(zeros,aim,color='black')
    ax2.scatter((arr[:,0]-center[0])/scale_axis[0], (arr[:,1]-center[1])/scale_axis[1], c='b',label='calibrated')
    ax2.axis([-1.5,1.5,-1.5,1.5])
#    ax2.plot(circle_x,circle_y, c='g',label='calibrated')
    ax2.grid()
    ax2.legend()
    
    
    ax3 = fig.add_subplot(413)
    ax3.set_title('y_z plane', fontsize=20)
    ax3.scatter(arr[:,1]/scale_axis[1], arr[:,2]/scale_axis[2], c='r',label='uncalibrated')
    ax3.set_xlabel('y_axis')
    ax3.set_ylabel('z_axis')
    ax3.plot(aim,zeros,color='black')
    ax3.plot(zeros,aim,color='black')
    ax3.scatter((arr[:,1]-center[1])/scale_axis[1], (arr[:,2]-center[2])/scale_axis[2], c='b',label='calibrated')
    ax3.grid()
    ax3.axis([-1.5,1.5,-1.5,1.5])
    ax3.legend()
    ax4 = fig.add_subplot(414)
    ax4.set_title('x_z plane', fontsize=20)
    ax4.plot(aim,zeros,color='black')
    ax4.plot(zeros,aim,color='black')
    ax4.scatter(arr[:,0]/scale_axis[0], arr[:,2]/scale_axis[2], c='r',label='uncalibrated')
    ax4.axis([-1.5,1.5,-1.5,1.5])
    ax4.scatter((arr[:,0]-center[0])/scale_axis[0], (arr[:,2]-center[2])/scale_axis[2], c='b',label='calibrated')
    ax4.plot(aim,zeros,color='black')
    ax4.plot(zeros,aim,color='black')
    ax4.set_xlabel('x_axis')
    ax4.set_ylabel('z_axis')
    ax4.grid()
    ax4.legend()
    plt.show()
    
    return center,scale_axis

def ellipse_fitting_wistron(mag):
#    path='./1207elipse_fitting/20181218112554.csv'
    # df = pd.read_csv(path, index_col=False,header=None)
#    mag=df.as_matrix(columns=df.columns[7:10])
    
    # raw=df.as_matrix(columns=df.columns[:])
    # mag=df.as_matrix(columns=df.columns[8:11])
    # mag=mag[raw[:,1]==devid]
    # path='./test_data/Angleasy87E31_SixAxis_210922.csv'
    # devid=0
#-----------------'./aiq_data/D01/01.csv'------------------    
    # df = pd.read_csv(path, index_col=False,header=None)
    # mag=df.values
#    err=np.logical_and(mag[:,1]>(-20),mag[:,2]<(30))
#    mag=mag[err]
    
    # mean=np.mean(mag,axis=0)
    # std=np.std(mag,axis=0)
    # threshold=mean+2*std
    # threshold2=mean-2*std
    # print('threshold',threshold)
    # m1=mag[:,0]<threshold[0]
    # m2=mag[:,1]<threshold[1]
    # m3=mag[:,2]<threshold[2]
    # m4=mag[:,0]>threshold2[0]
    # m5=mag[:,1]>threshold2[1]
    # m6=mag[:,2]>threshold2[2]
    # m=np.logical_and(m1,m2)
    # m=np.logical_and(m,m3)
    # m=np.logical_and(m,m4)
    # m=np.logical_and(m,m5)
    # m=np.logical_and(m,m6)
    # mag=mag[m]
    # print('mag.shape',mag.shape)
#    
#    mag=np.array([Magx,Magy,Magz]).transpose()
#    esti_mag/=norm(esti_mag)
#    mag/=norm(mag)

    
    print('mag calibrated')
    arr=mag
    D1=np.array([arr[:,0]**2])
    D2=np.array([arr[:,1]**2])
    D3=np.array([arr[:,2]**2])
    D4=np.array([arr[:,0]*arr[:,1]])
    D5=np.array([arr[:,0]*arr[:,2]])
    D6=np.array([arr[:,1]*arr[:,2]])
    D7=np.array([arr[:,0]])
    D8=np.array([arr[:,1]])
    D9=np.array([arr[:,2]])
    D=np.concatenate((D1,D2,D3,D4,D5,D6,D7,D8,D9))
    
    print(D.shape)
    test=np.dot(D,np.transpose(D))
    print(test.shape)
    a=np.linalg.inv(np.dot(D,np.transpose(D)))
        

    a=np.dot(a,D)
    print('start')
    print(a.shape)
    a=np.dot(a,np.ones(len(mag)))
    print(a.shape)

    print('a.shape',a.shape)
    M=np.array([[a[0],a[3]/2,a[4]/2],[a[3]/2,a[1],a[5]/2],[a[4]/2,a[5]/2,a[2]]])
    center=-0.5*np.array([a[6],a[7],a[8]])
    #center=np.transpose(center)
    center=np.dot(center,np.linalg.inv(M))
    print('center.shape',center.shape)
    SS=np.dot(center,M)
    SS=np.dot(SS,np.transpose(center))+1
    V,U=np.linalg.eig(M)
    
    n1=V[0]
    n2=V[1]
    n3=V[2]
    print(n1,n2,n3)
    print(U)
    
    np.diagflat([V[0],V[1],V[2]], k=0)
    scale_axis=np.array([(SS/n1)**0.5,(SS/n2)**0.5,(SS/n3)**0.5])

#    try:
#        
#        scale_axis=np.array([(SS/n1)**0.5,(SS/n2)**0.5,(SS/n3)**0.5])
#    except RuntimeWarning:
#        print('catch runtimewarning')
#    print('SS',SS)
#    print('n',n1,n2,n3)
    print('center=\n',center)
    print('scale_axis=\n',scale_axis)
    circle_x=np.linspace(-1,1,100)
    #circle_y=np.linspace(1,0,50)
    circle_y=(1-circle_x**2)**0.5
    circle_x=np.concatenate((circle_x,circle_x[::-1]))
    circle_y=np.concatenate((circle_y,-circle_y))
#    circle_z=(1-(circle_x**2+circle_y**2))**0.5
    
    aim=np.linspace(-np.max(arr),np.max(arr),100)
    zeros=np.zeros(len(aim))
    fig = plt.figure(figsize=(10,25))
    ax = fig.add_subplot(411, projection='3d')
    
    ax.scatter(arr[:,0]/scale_axis[0], arr[:,1]/scale_axis[1], arr[:,2]/scale_axis[2], c='r', marker='o',label='uncalibrated')
    ax.set_xlabel('x_axis')
    ax.set_ylabel('y_axis')
    ax.set_zlabel('z_axis')
    ax.set_title('x_y_z sphere', fontsize=20)
    ax.scatter((arr[:,0]-center[0])/scale_axis[0], (arr[:,1]-center[1])/scale_axis[1], (arr[:,2]-center[2])/scale_axis[2], c='b', marker='^',label='calibrated')
    ax.grid()
    ax.legend()
#    esti_mag[:,0]=(esti_mag[:,0]-center[0])/scale_axis[0]
#    esti_mag[:,1]=(esti_mag[:,1]-center[1])/scale_axis[1]
#    esti_mag[:,2]=(esti_mag[:,2]-center[2])/scale_axis[2]
    
    ax2 = fig.add_subplot(412)
    ax2.scatter(arr[:,0]/scale_axis[0], arr[:,1]/scale_axis[1], c='r',label='uncalibrated')
    ax2.set_xlabel('x_axis')
    ax2.set_ylabel('y_axis')
    ax2.set_title('x_y plane', fontsize=20)
    ax2.plot(aim,zeros,color='black')
    ax2.plot(zeros,aim,color='black')
    ax2.scatter((arr[:,0]-center[0])/scale_axis[0], (arr[:,1]-center[1])/scale_axis[1], c='b',label='calibrated')
    ax2.axis([-1.5,1.5,-1.5,1.5])
#    ax2.plot(circle_x,circle_y, c='g',label='calibrated')
    ax2.grid()
    ax2.legend()
    
    
    ax3 = fig.add_subplot(413)
    ax3.set_title('y_z plane', fontsize=20)
    ax3.scatter(arr[:,1]/scale_axis[1], arr[:,2]/scale_axis[2], c='r',label='uncalibrated')
    ax3.set_xlabel('y_axis')
    ax3.set_ylabel('z_axis')
    ax3.plot(aim,zeros,color='black')
    ax3.plot(zeros,aim,color='black')
    ax3.scatter((arr[:,1]-center[1])/scale_axis[1], (arr[:,2]-center[2])/scale_axis[2], c='b',label='calibrated')
    ax3.grid()
    ax3.axis([-1.5,1.5,-1.5,1.5])
    ax3.legend()
    ax4 = fig.add_subplot(414)
    ax4.set_title('x_z plane', fontsize=20)
    ax4.plot(aim,zeros,color='black')
    ax4.plot(zeros,aim,color='black')
    ax4.scatter(arr[:,0]/scale_axis[0], arr[:,2]/scale_axis[2], c='r',label='uncalibrated')
    ax4.axis([-1.5,1.5,-1.5,1.5])
    ax4.scatter((arr[:,0]-center[0])/scale_axis[0], (arr[:,2]-center[2])/scale_axis[2], c='b',label='calibrated')
    ax4.plot(aim,zeros,color='black')
    ax4.plot(zeros,aim,color='black')
    ax4.set_xlabel('x_axis')
    ax4.set_ylabel('z_axis')
    ax4.grid()
    ax4.legend()
    plt.show()
    
    return center,scale_axis
if __name__ == '__main__':
#    path='./1207elipse_fitting/20181218110723.csv'
    # path='./test_data/Angleasy87E31_SixAxis_210922.csv'
    # path='./test_data/Angleasy87E31_SixAxis_210923.csv'
    # path='./test_data/Angleasy87E31_SixAxis_210923_2.csv'
    # path='./test_data/Angleasy87E31_SixAxis_210923_4.csv'
    # path='./test_data/Angleasy87E31_SixAxis_210923_5.csv'
    
    # path='./test_data/Angleasy735EC_SixAxis_210923.csv'
    # path='./test_data/Angleasy735EC_SixAxis_210923_2.csv'
    # path='./test_data/Angleasy735EC_SixAxis_210923_3.csv'
    # path='./test_data/Angleasy735EC_SixAxis_210923_4.csv'
    # path='./test_data/Angleasy735EC_SixAxis_210923_5.csv'
    # path='./test_data/Angleasy735EC_SixAxis_210923_6.csv'
    # path='./test_data/Angleasy735EC_SixAxis_210923_7.csv'
    # path='./test_data/Angleasy735EC_SixAxis_210923_8.csv'
    # path='./test_data/Angleasy735EC_SixAxis_210923_9.csv'
    # path='./test_data/Angleasy735EC_SixAxis_210923_10.csv'
    # path='./test_data/Angleasy735EC_SixAxis_210923_11.csv'
    # path='./test_data/Angleasy735EC_SixAxis_210923_12.csv'
    
    # path='./test_data/Angleasy62D63_SixAxis_210923_1.csv'
    # path='./test_data/Angleasy62D63_SixAxis_210923_2.csv'
    # path='./test_data/Angleasy62D63_SixAxis_210923_3.csv'
    
    # path='./test_data/Angleasy62D63_SixAxis_210923_4.csv'
    # path='./test_data/Angleasy62D63_SixAxis_210923_5.csv'
    # path='./test_data/Angleasy62D63_SixAxis_210923_2.csv'
    
    # path='./test_data/AngleasyE0798_SixAxis_210924_1.csv'
    # path='./test_data/AngleasyE0798_SixAxis_210924_2.csv'
    path='./test_data/AngleasyE0798_SixAxis_210924_3.csv'
    path='./test_data/AngleasyE0798_SixAxis_210924_6.csv'
    path='./test_data/AngleasyE0798_SixAxis_210924_7.csv'
    
    path='./test_data/AngleasyE0798_SixAxis_210924_9.csv'
    # AngleasyE0798_SixAxis_210924_1
    # path='./aiq_data/D13/23.csv'
#    path='./aiq_data/D03/67.csv'
    # devid=0
#-----------------'./aiq_data/D01/01.csv'------------------    
    df = pd.read_csv(path, index_col=False,header=None)
    raw=df.values
    # raw=raw[raw[:,2]==devid]
    mag=raw[:,7:10]
#    err=np.logical_and(mag[:,1]>(-20),mag[:,2]<(30))
#    mag=mag[err]
#    plt.plot(mag[:,2])
#    plt.plot(mag[:,1])
#    plt.plot(raw[:,1])
#-----------------'./aiq_data/D01/01.csv'------------------
#--------------------1207elipse_fitting---------------------
#    mag=df.as_matrix(columns=df.columns[7:10])
#--------------------1207elipse_fitting---------------------
#    print('mag is')
#    print(mag[:,1])
#    mag=mag[mag[:,1]==4]
#    print(mag)
#    mag=mag/np.array([-1,1,-1])
#    mag-=np.array([100,200,300])
#    (mag[0,0]**2+mag[0,1]**2+mag[0,2]**2)**0.5
#    for i in range(len(mag)):
#        mag[i,:]=mag[i,:]/ norm(mag[i,:])
#    mag /= norm(mag)
#    plt.plot(norm(mag))
#    plt.plot((mag[:,0]**2+mag[:,1]**2+mag[:,2]**2)**0.5)
    plt.scatter(mag[:,0],mag[:,1])
    plt.scatter(mag[:,1],mag[:,2])
    plt.scatter(mag[:,0],mag[:,2])
    plt.show()

    new_mag=copy.deepcopy(mag)
    from cali_mag import cali_mag
    # out=cali_mag(mag,new_mag)
    ellipse_fitting_wistron(mag)
    # plt.show()
    # mag=(mag-center)/scale_axis
    # plt.scatter(mag[:,0],mag[:,1])
    # plt.scatter(mag[:,1],mag[:,2])
    # plt.scatter(mag[:,0],mag[:,2])
    # plt.show()
    
    # err=mag
    # plt.scatter(new_mag[:,0],new_mag[:,1])
    # plt.scatter(new_mag[:,1],new_mag[:,2])
    # plt.scatter(new_mag[:,0],new_mag[:,2])
    # plt.show()
