# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:51:43 2018

@author: 180218
"""
import numpy as np
import pandas as pd
import math
from numpy import linalg as LA
from numpy.linalg import norm
import matplotlib.pyplot as plt
from transforms3d import quaternions
#import calc_mag
#from . import calc_mag
#from calc_mag import calc_mag
from .quaternion import Quaternion
from .simu_filter import vec_filter
def plt_vec(vec,title):
    plt.title(title)
    plt.plot(vec[:,0],label='x')
    plt.plot(vec[:,1],label='y')
    plt.plot(vec[:,2],label='z')
    plt.legend()
    plt.show()
def load_aiq(pathway,dev_id):
    df = pd.read_csv(pathway,header=None)
    df=df.iloc[df.index[df[1]==dev_id],:]
    #df=df.iloc[df[0]>75968,:]
    
    #z=df[0]==-555
    #index=df.where(df[0]==-555)
    
    #index=df.index[df[0]==-555]
    #df=df.loc[38779:,:]
    raw=df.as_matrix(columns=df.columns[2:11])
    esti_acc,esti_gyr,esti_mag=np.split(raw,3,axis=1)
    esti_mag=esti_mag*5000
    esti_ts=df.as_matrix(columns=df.columns[0:1])
    time=esti_ts
    esti_delta_t=np.diff(time[:,0])/1000
#    simu_filter(esti_gyr,esti_acc,esti_mag,75)
    esti_mag[:,0]=-esti_mag[:,0]
    if dev_id==3 :
        print("do nothing")
    elif dev_id==0 or dev_id==1:
        esti_gyr[:,1]=-esti_gyr[:,1]
        esti_acc[:,1]=-esti_acc[:,1]
        esti_mag[:,1]=-esti_mag[:,1]
        esti_acc[:,:]=esti_acc[:,[2,1,0]]
        esti_gyr[:,:]=esti_gyr[:,[2,1,0]]
        esti_mag[:,:]=esti_mag[:,[2,1,0]]
    elif dev_id==5 or dev_id==6:
        esti_gyr[:,[0,2]]=-esti_gyr[:,[0,2]]
        esti_acc[:,[0,2]]=-esti_acc[:,[0,2]]
        esti_mag[:,[0,2]]=-esti_mag[:,[0,2]]
        esti_acc[:,:]=esti_acc[:,[2,0,1]]
        esti_gyr[:,:]=esti_gyr[:,[2,0,1]]
        esti_mag[:,:]=esti_mag[:,[2,0,1]]
    elif dev_id==2 or dev_id==4:   
        esti_gyr[:,0]=-esti_gyr[:,0]
        esti_acc[:,0]=-esti_acc[:,0]
        esti_mag[:,0]=-esti_mag[:,0]
        esti_acc[:,:]=esti_acc[:,[2,1,0]]
        esti_gyr[:,:]=esti_gyr[:,[2,1,0]]
        esti_mag[:,:]=esti_mag[:,[2,1,0]]
    else:
        print('wrong id')
    vec_filter(esti_mag,70,1)
    esti_acc[:,:]=-esti_acc[:,:]
    return esti_ts,esti_gyr,esti_acc,esti_mag,esti_delta_t

def calibrated_mag(uncali_mag,iid):
    path='./data/octopus/0525_experiment/NTU_IMU_GET_DATA2018-5-25 16_13_30.txt'
#    path='./data/octopus/0525_experiment/NTU_IMU_GET_DATA2018-5-25 16_19_10.txt'

#    16_19_10
    f = open(path,'r')
    line = f.readline()
    data=[]
    while line:
        data.append(line.split( ));
        line = f.readline()
    f.close()
    Ax = []
    Ay = []
    Az = []
    Gyrx=[]
    Gyry=[]
    Gyrz=[]
    Magx=[]
    Magy=[]
    Magz=[]
    esti_ts=[]
    for j in range(len(data)):
        if data[j][3]== iid:
            temp_ts=data[j][1].split(':')
#            print(temp_ts)
            h=(float)(temp_ts[0])
            m=(float)(temp_ts[1])
            s=(float)(temp_ts[2])
            temp_esti_t=h*3600+m*60+s+(float)(data[j][2])*0.001
            esti_ts.append(temp_esti_t)
            Gyrx.append(float(data[j][7]))
            Gyry.append(float(data[j][8]))
            Gyrz.append(float(data[j][9]))
            Ax.append(float(data[j][4]))
            Ay.append(float(data[j][5]))
            Az.append(float(data[j][6]))
            Magx.append(float(data[j][10]))
            Magy.append(float(data[j][11]))
            Magz.append(float(data[j][12]))
            
    esti_mag=np.array([Magx,Magy,Magz])
    mag=esti_mag.transpose()
#    esti_delta_t=np.ones(len(esti_acc))*0.01
#    esti_mag/=norm(esti_mag)
    mag/=norm(mag)
    mag=mag[10:-10]
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
    D=np.transpose(D)
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
    ax2.plot(circle_x,circle_y, c='g',label='calibrated')
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

#    return esti_mag
    
def violent_cali_mag(uncali_mag,uu_id):
    #
    path='./data/octopus/0611/cali_mag05.csv'
            
    df = pd.read_csv(path, index_col=False)
    raw=np.zeros([df.shape[0],9])
    u_id=np.zeros([df.shape[0]])
    esti_mag=np.zeros([len(raw),3])
    esti_acc=np.zeros([len(raw),3])
    esti_gyr=np.zeros([len(raw),3])
    raw[:,:]=df.values[:,3:12]
#    raw=raw[u_id==uu_id,:]
#    esti_acc[u_id==2,:]
#    esti_delta_t=np.diff(esti_ts,axis=0)
    esti_acc,esti_gyr,esti_mag=np.split(raw,3,axis=1)
    mag=esti_mag
    print(mag)
    xmin = np.min(mag[:,0])
    ymin = np.min(mag[:,1])
    zmin = np.min(mag[:,2])
    xmax = np.max(mag[:,0])
    ymax = np.max(mag[:,1])
    zmax = np.max(mag[:,2])
    xc = 0.5*(xmax+xmin)
    yc = 0.5*(ymax+ymin)
    zc = 0.5*(zmax+zmin)
    a = 0.5*abs(xmax-xmin)
    b = 0.5*abs(ymax-ymin)
    c = 0.5*abs(zmax-zmin)
    x = mag[:,0]
    y = mag[:,1]
    z = mag[:,2]
    l=len(mag[:,0])
    err=0
    for i in range(l):
        err = err + abs((x[i]-xc)**2/a**2 + (y[i]-yc)**2/b**2+(z[i]-zc)**2/c**2 - 1 )
    print('xc = ',xc,', yc = ',yc,', zc = ',zc,', a = ',a,', b = ',b,', c= ',c,', 初始化InitErr =',err)
    x = mag[:,0]
    y = mag[:,1]
    z = mag[:,2]
    xclast = xc
    yclast = yc
    zclast = zc
    alast = a
    blast = b
    clast = c
    errlast = 100000000000
    print('start calculating')
    for i in range(1,100):
        r=np.random.rand(6,1)
        xcnew = xclast + r[0]-0.5
        ycnew = yclast + r[1]-0.5
        zcnew = zclast + r[2]-0.5
        anew = abs(alast + r[3]-0.5)
        bnew = abs(blast + r[4]-0.5)
        cnew = abs(clast + r[5]-0.5)
        errnew = 0;
        for j in range(1,l):
            errnew = errnew + abs((x[j]-xcnew)**2/anew**2 + (y[j]-ycnew)**2/bnew**2 + (z[j]-zcnew)**2/cnew**2 - 1 )  
        if errnew < errlast:    #有更好的解，接受新解
            xclast = xcnew
            yclast = ycnew
            zclast = zcnew
            alast = anew
            blast = bnew
            clast = cnew
            errlast = errnew
    
    
    print('xc = ',xc,', yc = ',yc,', zc = ',zc,', a = ',a,', b = ',b,', c= ',c,', 最后 InitErr=',err)
    avr = (alast+blast+clast)/3
    newmag=np.zeros([l,3])
    newmag[:,0]=x
    newmag[:,1]=y
    newmag[:,2]=z
    
    print('mx = (mx - ',xclast,')*',alast/avr)
    print('my = (my - ',yclast,')*',blast/avr)
    print('mz = (mz - ',zclast,')*',clast/avr)
    #fig = plt.figure(figsize=(6, 5), dpi=70)
    newmag[:,0]=(newmag[:,0]-xclast)*alast/avr
    newmag[:,1]=(newmag[:,1]-yclast)*blast/avr
    newmag[:,2]=(newmag[:,2]-zclast)*clast/avr

    import matplotlib as mpl
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure(figsize=(6, 16), dpi=70)
    ax = fig.add_subplot(311)
    ax.grid()
    standard=np.linspace(-60,60,100)
    zeros=np.zeros(100)
    ax.plot(zeros, standard, color='black', linewidth=3, linestyle='--')
    ax.plot(standard, zeros, color='black', linewidth=3, linestyle='--')
    ax.scatter(mag[:,0], mag[:,1], label='uncalibrated value')
    ax.scatter(newmag[:,0], newmag[:,1], label='calibrated value')
    ax.set_xlabel('x')
    ax.set_xlabel('y')
    
    ax2 = fig.add_subplot(312)
    ax2.grid()
    ax2.plot(zeros, standard, color='black', linewidth=3, linestyle='--')
    ax2.plot(standard, zeros, color='black', linewidth=3, linestyle='--')
    ax2.scatter(mag[:,1], mag[:,2], label='uncalibrated value')
    ax2.scatter(newmag[:,1], newmag[:,2], label='calibrated value')
    ax2.set_xlabel('y')
    ax2.set_xlabel('z')
    
    ax3 = fig.add_subplot(313)
    ax3.grid()
    ax3.plot(zeros, standard, color='black', linewidth=3, linestyle='--')
    ax3.plot(standard, zeros, color='black', linewidth=3, linestyle='--')
    ax3.scatter(mag[:,0], mag[:,2], label='uncalibrated value')
    ax3.scatter(newmag[:,0], newmag[:,2], label='calibrated value')
    ax3.set_xlabel('x')
    ax3.set_xlabel('z')
    ax.legend()
    plt.show()
    uncali_mag[:,0]=(uncali_mag[:,0]-xclast)*alast/avr
    uncali_mag[:,1]=(uncali_mag[:,1]-yclast)*blast/avr
    uncali_mag[:,2]=(uncali_mag[:,2]-zclast)*clast/avr
    return uncali_mag
def load_txt(data_path):
    data=[]
    with open(data_path,"r") as f:
        c = True
        while(c != ""):
            c = f.readline()[0:-1]
            data.append(c.split("	"))
        data.pop(-1)
        l=len(data)
        ts=np.zeros((l,1))   
        acc=np.zeros((l,3))   
        gyr=np.zeros((l,3))
        mag=np.zeros((l,3))
        for i in range(l):
            ts[i,0]=data[i][0]
            acc[i,0]=data[i][1]
            acc[i,1]=data[i][2]
            acc[i,2]=data[i][3]
            gyr[i,0]=data[i][4]
            gyr[i,1]=data[i][5]
            gyr[i,2]=data[i][6]
            mag[i,0]=data[i][7]
            mag[i,1]=data[i][8]
            mag[i,2]=data[i][9]
        
        ts[0]=0
#        acc=acc*16/32768
        gyr=gyr*np.pi/180
#        mag=mag*4800/8192
#        circulation='C:/Users/180218/Desktop/albertsense/final/albert_calc0.txt'
#        mag=calc_mag.calc_mag_fromtxt(circulation,mag)
        
        
        #acc[:,0]+=-39.49586777
        #acc[:,1]+=3.792637115	
        #acc[:,2]+=31.03230654
        #gyr[:,0]+=157.5860255
        #gyr[:,1]+=-68.6904583
        #gyr[:,2]+=67.42223892
#        acc[:,0]+=0.000714908
#        acc[:,1]+=0.001851874
#        acc[:,2]+=0.015152493
#        gyr[:,0]+=0.020983848
#        gyr[:,1]+=-0.009146687
#        gyr[:,2]+=0.008977814
        print('np.mean(ts)=',np.mean(ts))
        ts=np.cumsum(ts)
    return ts,acc,gyr,mag
def load_octopus(path,dev_id):
    #IMUID       部位
    #0   右肩
    #1   頸後
    #2   頭
    #3   左肩
    #4   中間
    #5   下背

    f = open(path,'r')
    line = f.readline()
    data=[]
    while line:
        data.append(line.split( ));
        line = f.readline()
    f.close()
    Ax = []
    Ay = []
    Az = []
    Gyrx=[]
    Gyry=[]
    Gyrz=[]
    Magx=[]
    Magy=[]
    Magz=[]
    esti_ts=[]
    for j in range(len(data)):
        if data[j][3]== dev_id:
            temp_ts=data[j][1].split(':')
#            print(temp_ts)
            h=(float)(temp_ts[0])
            m=(float)(temp_ts[1])
            s=(float)(temp_ts[2])
            temp_esti_t=h*3600+m*60+s+(float)(data[j][2])*0.001
            esti_ts.append(temp_esti_t)
            Gyrx.append(float(data[j][7]))
            Gyry.append(float(data[j][8]))
            Gyrz.append(float(data[j][9]))
            Ax.append(float(data[j][4]))
            Ay.append(float(data[j][5]))
            Az.append(float(data[j][6]))
            Magx.append(float(data[j][10]))
            Magy.append(float(data[j][11]))
            Magz.append(float(data[j][12]))
    esti_acc=np.array([Ax,Ay,Az]).transpose()
    esti_gyr=np.array([Gyrx,Gyry,Gyrz]).transpose()
    esti_mag=np.array([Magx,Magy,Magz]).transpose()
#    esti_delta_t=np.ones(len(esti_acc))*0.01
    esti_delta_t=np.diff(esti_ts)
    esti_ts=np.zeros([len(esti_delta_t)+1])
    esti_ts[1:]=np.cumsum(esti_delta_t)
    calibrated_mag(esti_mag,dev_id)
    
    return esti_ts,esti_acc,esti_gyr,esti_mag,esti_delta_t

def mag_v(v):
    magv=np.zeros(len(v))
    for i in range(len(v)):
        magv[i]=(v[i,0]**2+v[i,1]**2+v[i,2]**2)**0.5
        v[i]/=magv[i]
    return v
def mag_v1(v):
    return (v[0]**2+v[1]**2+v[2]**2)**0.5


def calc_K(v):
    k=np.array([[v[0,0]-v[1,1]-v[2,2],v[1,0]+v[0,1],v[2,0]+v[0,2],v[1,2]-v[2,1]],
    [v[1,0]+v[0,1],v[1,1]-v[0,0]-v[2,2],v[2,1]+v[1,2],v[2,0]-v[0,2]],
    [v[2,0]+v[0,2],v[2,1]+v[1,2],v[2,2]-v[0,0]-v[1,1],v[0,1]-v[1,0]],
    [v[1,2]-v[2,1],v[2,0]-v[0,2],v[0,1]-v[1,0],v[0,0]+v[1,1]+v[2,2]]])
    k=k*(1/3)
    
    return k
def load_8marker(data_path):
#    names = ['Time (s)', 'Gyroscope X (deg/s)', 'Gyroscope Y (deg/s)', 'Gyroscope Z (deg/s)', 'Accelerometer X (g)', 'Accelerometer Y (g)','Accelerometer Z (g)','Magnetometer X (uT)','Magnetometer Y (uT)','Magnetometer Z (uT)']
    df = pd.read_csv(data_path, index_col=False)
    i_end=np.zeros([len(df.values[:,0]),3])
    i_start=np.copy(i_end)
    j_end=np.copy(i_end)
    j_start=np.copy(i_end)
    k_end=np.copy(i_end)
    k_start=np.copy(i_end)
    l_end=np.copy(i_end)
    l_start=np.copy(i_end)
    
    i_end[:,:]=df.values[:,14:17]  #1
    i_start[:,:]=df.values[:,19:22] #2
    j_end[:,:]=df.values[:,24:27] #3
    j_start[:,:]=df.values[:,29:32] #4
    k_end[:,:]=df.values[:,34:37] #5
    k_start[:,:]=df.values[:,39:42] #6
    l_end[:,:]=df.values[:,44:47] #7
    l_start[:,:]=df.values[:,49:52] #8
    i_end[:,1]=-i_end[:,1]
    i_start[:,1]=-i_start[:,1]    
    j_end[:,1]=-j_end[:,1]
    j_start[:,1]=-j_start[:,1]   
    k_end[:,1]=-k_end[:,1]
    k_start[:,1]=-k_start[:,1]   
    l_end[:,1]=-l_end[:,1]
    l_start[:,1]=-l_start[:,1]   
    return i_end,i_start,j_end,j_start,k_end,k_start,l_end,l_start
def load_4marker(data_path):
#    names = ['Time (s)', 'Gyroscope X (deg/s)', 'Gyroscope Y (deg/s)', 'Gyroscope Z (deg/s)', 'Accelerometer X (g)', 'Accelerometer Y (g)','Accelerometer Z (g)','Magnetometer X (uT)','Magnetometer Y (uT)','Magnetometer Z (uT)']
    df = pd.read_csv(data_path, index_col=False)
    c_end=np.zeros([len(df.values[:,0]),3])
    c_start=np.copy(c_end)
    p_end=np.copy(c_end)
    p_start=np.copy(c_end)
    c_xe=np.zeros([1,3])
    c_ye=np.zeros([1,3])
    p_end[:,:]=df.values[:,14:17]  
#    p_start[:,:]=df.values[:,19:22] #2 -z
#    c_end[:,:]=df.values[:,24:27] #3 +x
#    c_start[:,:]=df.values[:,29:32] #4
##    print(c_end)
#
#    c_p=np.mean((p_end-p_start),axis=0) #c_z g
#    c_p/=mag_v1(c_p)
##    print(c_p)
#    c_c=np.mean((c_end-c_start),axis=0) #c_compass
#    c_c/=mag_v1(c_c)
#    c_xe=c_c-np.inner(c_c,c_p)/(mag_v1(c_p)**2)*c_p
#    c_xe/=mag_v1(c_xe)
#    c_ye[:]=[-(1-c_xe[0]**2-c_p[0]**2)**0.5,
#        -(1-c_xe[1]**2-c_p[1]**2)**0.5,
#        -(1-c_xe[2]**2-c_p[2]**2)**0.5]
##    c_ye=-c_ye
#    c_xe=np.array([c_xe])
#    c_p=np.array([c_p])
    return df#p_end,p_start,c_end,c_start,c_xe,c_ye,c_p
def change_order_q(q):
    w=q[3]
    x=q[0]
    y=q[1]
    z=q[2]
    return [w,x,y,z]
def quaternion_to_euler_angle(w, x, y, z):
    ysqr = y * y
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))
    return X, Y, Z
def load_vicon(data_path):
#    names = ['Time (s)', 'Gyroscope X (deg/s)', 'Gyroscope Y (deg/s)', 'Gyroscope Z (deg/s)', 'Accelerometer X (g)', 'Accelerometer Y (g)','Accelerometer Z (g)','Magnetometer X (uT)','Magnetometer Y (uT)','Magnetometer Z (uT)']
    df = pd.read_csv(data_path, index_col=False)
#    df=df[35:-35,:]
    i_end=np.zeros([len(df.values[:,0]),3])
    i_start=np.copy(i_end)
    j_end=np.copy(i_end)
    j_start=np.copy(i_end)
    k_start=np.copy(i_end)
    k_end=np.copy(i_end)
    temp_mts=np.zeros(len(df.values[:,11]))
    temp_ts=np.zeros(len(df.values[:,10]))
    ref_delta_t=np.zeros(len(df.values[:,11]))
    ref_ts=np.copy(temp_ts)
    temp_mts[:]=df.values[:,11]
    temp_ts[:]=df.values[:,10]
    ref_ts[:]=temp_ts[:]+temp_mts[:]*0.001
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
    j_end[:,:]=df.values[:,14:17]  #1 +x方向8-1
    j_start[:,:]=df.values[:,49:52] #8 
    i_end[:,:]=df.values[:,24:27] #3#
    i_start[:,:]=df.values[:,29:32] #4#    
    k_start[:,:]=df.values[:,19:22] #2  #y 2-5
    k_end[:,:]=df.values[:,34:37] #5#
    j_end=j_end[:,[0,2,1]]
    j_start=j_start[:,[0,2,1]]
    i_end=i_end[:,[0,2,1]]
    i_start=i_start[:,[0,2,1]]
    k_end=k_end[:,[0,2,1]]
    k_start=k_start[:,[0,2,1]]
    c_xm=mag_v(i_end-i_start)
    c_ym=mag_v(j_end-j_start)
    c_zm=mag_v(k_end-k_start)
    Q=np.zeros([len(c_xm),4])
    tempa=np.zeros([3,3])
    for i in range(len(Q)-2):
        tempa=np.vstack((c_xm[i,:],c_ym[i,:],c_zm[i,:]))
        tempa=-tempa.transpose()  
        Q[i,:]=quaternions.mat2quat(tempa)
#    Q[:,:]=df.values[:,6:10]
#    Q=Q[:,[3,0,1,2]]
    Q=Q[:,[0,1,3,2]]
    Q[:,0]=-Q[:,0]
#    Q=Q[:,[0,2,1,3]]
#    Q[:,2]=-Q[:,2]
    ref_eu=np.zeros([len(Q),3])
#    Qb=Quaternion([1,0,0,0])
#    Qc=np.array([[0,0,-0.707,0.707],
#                 [0,0,0.707,0.707],
#                 [0.707,-0.707,0,0],
#                 [-0.707,-0.707,0,0]])
#    Q=np.dot(Q,Qc)
#    Qb=Quaternion(np.mean(Q[100:400,:],axis=0))
#    Qb=Quaternion(0.246265,-0.0168084,0.0457804,0.967974)
#    Qb=Qb.conj() #有加這樣才對 用static測試
    for i in range(1,len(Q)):
        tempq=Quaternion(Q[i,:])
#        tempq=tempq*Qb.conj()
#        Q[i,:]=tempq.__array__()
        ref_eu[i,0],ref_eu[i,1],ref_eu[i,2]=tempq.to_euler_angles_by_wiki()
#        ref_eu[i,0],ref_eu[i,1],ref_eu[i,2]=tempq.to_euler_angles_no_Gimbal()
    
    return ref_ts,c_xm,c_ym,c_zm,Q,ref_eu/np.pi*180,ref_delta_t
def q_unity2earth(q):
    temp=np.zeros(len(q))
    temp=q[:,3]
    q[:,3]=q[:,2]
    q[:,2]=temp
def load_octopus_csv(data_path,uu_id):
    df = pd.read_csv(data_path,header=None)
    dev_id=uu_id
    df=df.iloc[df.index[df[2]==dev_id],:]
    raw=df.as_matrix(columns=df.columns[3:12])
    esti_acc,esti_gyr,esti_mag=np.split(raw,3,axis=1)
    t_ms=df.as_matrix(columns=df.columns[1:2])
    time=df.iloc[:,0].str.replace('[^0-9]',' ').str.split(' ',expand=True).astype(dtype=np.float32).as_matrix()
    if time.shape[1]>3:
        esti_ts=time[:,2]*3600+time[:,3]*60+time[:,4]+t_ms[:,0]*0.001
    else:
        esti_ts=time[:,0]*3600+time[:,1]*60+time[:,2]+t_ms[:,0]*0.001
    
    esti_ts-=esti_ts[0]
    esti_delta_t=np.diff(esti_ts)
    return esti_ts,esti_acc,esti_gyr,esti_mag,esti_delta_t
def load_vicon_octopus(data_path):
    df = pd.read_csv(data_path, index_col=False)
    ref_q=np.zeros([len(df.values[:,0]),4])
    temp_mts=np.zeros(len(df.values[:,11]))
    temp_ts=np.zeros(len(df.values[:,10]))
    ref_delta_t=np.zeros(len(df.values[:,11]))
    ref_ts=np.copy(temp_ts)
    temp_mts[:]=df.values[:,11]
    temp_ts[:]=df.values[:,10]
    ref_ts[:]=temp_ts[:]+temp_mts[:]*0.001
    ref_eu=np.zeros([len(ref_q),3])
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
    c_xm=df.values[:,0]
    c_ym=df.values[:,1]
    c_zm=df.values[:,2]
    ref_q=df.values[:,6:10]
#    ref_q[:,1]=-ref_q[:,1]

#    ref_q=-ref_q

#    ref_q=ref_q[:,[0,2,1,3]]

    ref_q=ref_q[:,[0,1,3,2]]
#    ref_q=ref_q[:,[0,2,1,3]]
    ref_q[:,0]=-ref_q[:,0]
#    ref_q=ref_q[:,[0,2,1,3]]
#    ref_q[:,1]=-ref_q[:,1]
#    ref_q[:,2]=-ref_q[:,2]
#    ref_q[:,3]=-ref_q[:,3]
    for i in range(1,len(ref_q)):
        tempq=Quaternion(ref_q[i])
#        tempq=tempq*Quaternion(ref_q[i-1])
        ref_eu[i,0],ref_eu[i,1],ref_eu[i,2]=tempq.to_euler_angles_no_Gimbal()
        
    return ref_ts,c_xm,c_ym,c_zm,ref_q,ref_eu/np.pi*180,ref_delta_t

def load_vicon_q_eu(data_path):
    df = pd.read_csv(data_path, index_col=False)
    ref_q=np.zeros([len(df.values[:,0]),4])
    temp_mts=np.zeros(len(df.values[:,11]))
    temp_ts=np.zeros(len(df.values[:,10]))
    ref_delta_t=np.zeros(len(df.values[:,11]))
    ref_ts=np.copy(temp_ts)
    temp_mts[:]=df.values[:,11]
    temp_ts[:]=df.values[:,10]
    ref_ts[:]=temp_ts[:]+temp_mts[:]*0.001
    ref_eu=np.zeros([len(ref_q),3])
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
    c_xm=df.values[:,0]
    c_ym=df.values[:,1]
    c_zm=df.values[:,2]
    ref_q=df.values[:,6:10]
#    ref_q[:,1]=-ref_q[:,1]

#    ref_q=-ref_q

#    ref_q=ref_q[:,[0,2,1,3]]
#    ref_q[:,3]=-ref_q[:,3]
#    ref_q=ref_q[:,[0,1,3,2]]
#    ref_q=ref_q[:,[0,2,1,3]]
#    ref_q[:,1]=-ref_q[:,1]
#    ref_q[:,2]=-ref_q[:,2]
#    ref_q[:,3]=-ref_q[:,3]
    for i in range(len(ref_q)):
        tempq=Quaternion(ref_q[i])
        ref_eu[i,0],ref_eu[i,1],ref_eu[i,2]=tempq.to_euler_angles_no_Gimbal()
        
    return ref_ts,c_xm,c_ym,c_zm,ref_q,ref_eu/np.pi*180,ref_delta_t

def id_maxium(data):
    index=0
    temp=0
    for i in range(len(data)):
        if data[i]>temp:
            temp=data[i]
            index=i
    return index
def load_white(data_path):
    df = pd.read_csv(data_path)
    l=len(df.values[:,1])
    ts=np.zeros([l,1])
    delta_t=np.zeros([l,1])
    acc=np.zeros([l,3])
    gyr=np.zeros([l,3])
    mag=np.zeros([l,3])
    
#    esti_eu=np.zeros([l,3])
#    esti_quater=np.zeros([l,4])
    ts=df.values[:,1]
    acc[:,0]=df.values[:,2]
    acc[:,1]=df.values[:,3]
    acc[:,2]=df.values[:,4]
    acc=acc/2048
    
    gyr[:,0]=df.values[:,5]
    gyr[:,1]=df.values[:,6]
    gyr[:,2]=df.values[:,7]
    gyr=gyr*250/32768/180*np.pi
    mag[:,0]=df.values[:,8]
    mag[:,1]=df.values[:,9]
    mag[:,2]=df.values[:,10]
#    esti_eu[:,0]=df.values[:,10]
#    esti_eu[:,1]=df.values[:,11]
#    esti_eu[:,2]=df.values[:,12]
#    esti_quater[:,0]=df.values[:,13]
#    esti_quater[:,1]=df.values[:,14]
#    esti_quater[:,2]=df.values[:,15]
#    esti_quater[:,3]=df.values[:,16]
    delta_t=np.diff(ts)/10000
#    gyr=gyr/np.pi*180
#    print('calc_mag')
    
#    circulation='C:/Users/180218/Desktop/ngimu/table_zc2/NGIMU - 0029C6FC/sensors.csv'
#    circulation='C:/Users/180218/Desktop/ngimu/Separator_MAG/NGIMU - 0029C6FC/sensors.csv'
    
#    circulation='C:/Users/180218/Desktop/ngimu/bike_slow_nomag/NGIMU - 0029C6FC/sensors.csv'
#    circulation='C:/Users/180218/Desktop/ngimu/calc_rightangle/NGIMU - 0029C6FC/sensors.csv'
#    circulation='C:/Users/180218/Desktop/ngimu/madgwick_z/NGIMU - 0029C6FC/sensors.csv'
#    
#    mag=calc_mag.calc_mag(circulation,mag)
    
#    mag[:,0]=(mag[:,0]-36.2109375)*1.6273007686084142
#    mag[:,1]=(mag[:,1]+43.453125)*1.4668626646611058
#    mag[:,2]=(mag[:,2]+67.515625)*1.4209422465559873
#    print('comple calc_mag')
#    return ts,acc,gyr,mag,esti_eu,esti_quater,delta_t
    return ts,acc,gyr,mag,delta_t
def load_iii(data_path):
    df = pd.read_csv(data_path, index_col=False)
    l=len(df.values[:,1])

    ts=np.zeros([l,1])
    delta_t=np.zeros([l,1])
    acc=np.zeros([l,3])
    gyr=np.zeros([l,3])
    mag=np.zeros([l,3])
    esti_eu=np.zeros([l,3])
    esti_quater=np.zeros([l,4])
    ts=df.values[:,0]
    acc[:,0]=df.values[:,1]
    acc[:,1]=df.values[:,2]
    acc[:,2]=df.values[:,3]
    gyr[:,0]=df.values[:,4]
    gyr[:,1]=df.values[:,5]
    gyr[:,2]=df.values[:,6]
    mag[:,0]=df.values[:,7]
    mag[:,1]=df.values[:,8]
    mag[:,2]=df.values[:,9]
    esti_eu[:,0]=df.values[:,10]
    esti_eu[:,1]=df.values[:,11]
    esti_eu[:,2]=df.values[:,12]
    esti_quater[:,0]=df.values[:,13]
    esti_quater[:,1]=df.values[:,14]
    esti_quater[:,2]=df.values[:,15]
    esti_quater[:,3]=df.values[:,16]
    delta_t=df.values[:,17]
#    Qb=Quaternion(np.mean(esti_quater[200:800,:],axis=0))
#    for i in range(len(esti_quater)):
#        tempq=Quaternion(esti_quater[i,:])
#        tempq=tempq*Qb.conj()
#        esti_quater[i,:]=tempq._get_q()
#        esti_eu[i,0],esti_eu[i,1],esti_eu[i,2]=tempq.to_euler_angles_by_wiki()
#    esti_eu=esti_eu/np.pi*180
#    gyr=gyr/np.pi*180
#    print('calc_mag')
#    circulation='C:/Users/180218/Desktop/ngimu/table_zc2/NGIMU - 0029C6FC/sensors.csv'
#    circulation='C:/Users/180218/Desktop/ngimu/Separator_MAG/NGIMU - 0029C6FC/sensors.csv'
#    circulation='C:/Users/180218/Desktop/ngimu/bike_slow_nomag/NGIMU - 0029C6FC/sensors.csv'
#    circulation='C:/Users/180218/Desktop/ngimu/calc_rightangle/NGIMU - 0029C6FC/sensors.csv'
#    circulation='C:/Users/180218/Desktop/ngimu/madgwick_z/NGIMU - 0029C6FC/sensors.csv'
#    mag=calc_mag.calc_mag(circulation,mag)
#    mag[:,0]=(mag[:,0]-36.2109375)*1.6273007686084142
#    mag[:,1]=(mag[:,1]+43.453125)*1.4668626646611058
#    mag[:,2]=(mag[:,2]+67.515625)*1.4209422465559873
#    print('comple calc_mag')
    return ts,acc,gyr,mag,esti_eu,esti_quater,delta_t
def load_csv_quaternion(data_path):
#    names = ['W', 'X', 'Y', 'Z']
    df = pd.read_csv(data_path)
    ts=df.values[:,0]
    l=len(ts)
    ref_quaternion=np.zeros([l,4])
    ref_eu=np.zeros([l,3])
    ref_quaternion[:,0]=df.values[:,1]
#    ref_quaternion[:,1]=df.values[:,2]
#    ref_quaternion[:,2]=df.values[:,3]
#    ref_quaternion[:,3]=df.values[:,4]
    ref_quaternion[:,1]=df.values[:,2]
    ref_quaternion[:,2]=df.values[:,3]
    ref_quaternion[:,3]=df.values[:,4]
#    for i in range(l):
#        ref_eu[i,0],ref_eu[i,1],ref_eu[i,2]=to_euler_angles_by_wiki(ref_quaternion[i,0],ref_quaternion[i,1],ref_quaternion[i,2],ref_quaternion[i,3])
#    ref_eu=ref_eu*180/np.pi
    for i in range(l):
        ref_eu[i,0],ref_eu[i,1],ref_eu[i,2]=quaternion_to_euler_angle(ref_quaternion[i,0],ref_quaternion[i,1],ref_quaternion[i,2],ref_quaternion[i,3])
    return ref_quaternion,ref_eu

def to_euler_angles_by_wiki(q1,q2,q3,q4):
    pitch=np.arcsin(2 * (q1 * q3 -  q4 * q2))
    roll=np.arctan2(2 * (q1 * q2 +  q3 * q4), 1- 2 * (q2 ** 2 +  q3 ** 2))
    yaw=np.arctan2(2 * (q1 * q4 + q2 * q3), 1- 2 * (q3 ** 2 + q4 ** 2))
    pitch=(pitch/np.pi)*180
    roll=(roll/np.pi)*180
    yaw=(yaw/np.pi)*180
    return roll, pitch, yaw
def quaternion_to_euler_angle(w, x, y, z):
    ysqr = y * y
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))
    return X, Y, Z
def load_csv_cmpmag(data_path,circumpath):
#    names = ['Time (s)', 'Gyroscope X (deg/s)', 'Gyroscope Y (deg/s)', 'Gyroscope Z (deg/s)', 'Accelerometer X (g)', 'Accelerometer Y (g)','Accelerometer Z (g)','Magnetometer X (uT)','Magnetometer Y (uT)','Magnetometer Z (uT)']
    df = pd.read_csv(data_path)
    ts=df.values[:,0]
    l=len(ts)
    acc=np.zeros([l,3])
    gyr=np.zeros([l,3])
    mag=np.zeros([l,3])
    acc[:,0]=df.values[:,4]
    acc[:,1]=df.values[:,5]
    acc[:,2]=df.values[:,6]
    gyr[:,0]=df.values[:,1]
    gyr[:,1]=df.values[:,2]
    gyr[:,2]=df.values[:,3]
    mag[:,0]=df.values[:,7]
    mag[:,1]=df.values[:,8]
    mag[:,2]=df.values[:,9]
    uncali_mag=np.copy(mag)
    gyr=gyr*np.pi/180
    print('calc_mag')
#    circumpath='C:/Users/180218/Desktop/home_mag/NGIMU - 0029C6FC/sensors.csv'
    calimag=calc_mag.calc_mag(circumpath,mag)
    print('comple calc_mag')
    return ts,acc,gyr,calimag,uncali_mag

