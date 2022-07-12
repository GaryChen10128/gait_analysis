import ahrs
from ahrs.common.orientation import q_prod, q_conj, acc2q, am2q, q2R, q_rot
import pyquaternion
import ximu_python_library.xIMUdataClass as xIMU
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from package.temp_package import *
%matplotlib qt
# %matplotlib inline
filePath = 'datasets/spiralStairs'
startTime = 0
stopTime = 500
samplePeriod = 1/256


Writer.figure(path='0502csvWriteTest.csv')
from enum import Enum

# acc paramters
#setting acc's calibration parameters
acc_bias=np.array([0.01184237,-0.02746599,0.02606265])
acc_scale=np.array([[0.95566679,-0.0249658,-0.03582581],[-0.0249658,0.95940553,0.01238245],[-0.03582581,0.01238245,0.97620943]])

# [[ 0.95566679 -0.0249658  -0.03582581]
#  [-0.0249658   0.95940553  0.01238245]
#  [-0.03582581  0.01238245  0.97620943]]
class BIAS(Enum):
    GYR_X = 0
    GYR_Y = 0
    GYR_Z = 0
    # GYR_X = -0.74
    # GYR_Y = 0.26
    # GYR_Z = 0.067
    # GYR_X = -1.9
    # GYR_Y = 0.35
    # GYR_Z = 0.067
samplePeriod = 1/256
xIMUdata = xIMU.xIMUdataClass(filePath, 'InertialMagneticSampleRate', 1/samplePeriod)

#--------------------plot setting--------------------
plot_vel=False

plot_pos=False
plot_q=True

plot_acc=False
#----------------------------------------------------
#------------------origin parameters-----------------
# samplePeriod=1/256
# hpcutoff=0.001
# lpcutoff=5
# order=1
# threshold=0.05
useMahony=True
useMag=False
# reprocess=True
#----------------------------------------------------



#------------------my parameters-----------------
samplePeriod=1/200
hpcutoff=0.001
lpcutoff=5
order=1
threshold=0.05
useMahony=1 ##是否適用mahony重算quaternion
useMag=0 ##是否使用磁力計
useMadgwick=0
singlelevel=1
caliACC=0
mahonyKp=1
mahonyKi=0
dontswitch=0 ##############important
mahonyrate=samplePeriod
#----------------------------------------------------


pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/nmagr1/Roy62E24_9axis_220627.csv' #r1r2

# 直線"有"干擾
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/onhand/NoDistortion/straight/Roy62E24_9axis_220701.csv' #r1r2
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/onFoot/NoDistortion/straight/Roy62E24_9axis_220701.csv' #r1r2
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/waist/NoDistortion/straight/Roy62E24_9axis_220701.csv' #r1r2


# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/onhand/Distortion/straight/Roy62E24_9axis_220701.csv' #r1r2
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/onFoot/Distortion/straight/Roy62E24_9axis_220701.csv' #r1r2
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/waist/Distortion/straight/Roy62E24_9axis_220701.csv' #r1r2


#方
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/onhand/NoDistortion/square/Roy62E24_9axis_220701.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/onFoot/NoDistortion/square/Roy62E24_9axis_220701.csv' #r2
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/waist/NoDistortion/square/Roy62E24_9axis_220701.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/onhand/Distortion/square/Roy62E24_9axis_220701.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/onFoot/Distortion/square/Roy62E24_9axis_220701.csv' #r2 
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/waist/Distortion/square/Roy62E24_9axis_220701.csv' #r2 

# 常直線"有"干擾
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/longDistance/Distortion/onHand/Roy62E24_9axis_220701.csv' #r2 

pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/longDistance/Distortion/onFoot/Roy62E24_9axis_220701.csv' #r2 
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/longDistance/Distortion/onWaist/Roy62E24_9axis_220701.csv' #r2 
# 常直線"沒"干擾
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/longDistance/NoDistortion/onFoot/Roy62E24_9axis_220701.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/longDistance/NoDistortion/onWaist/Roy62E24_9axis_220701.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/longDistance/NoDistortion/onHand/Roy62E24_9axis_220701.csv' #r2 

#靜置table
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/staticTableGameRoy0704/staticTableGameRoy0704/Roy62E24_9axis_220704.csv' #r1r2


# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/longDistance/Distortion/onHand/Roy62E24_9axis_220701.csv' #r2 
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220707mahony12_7noMagSquare/Roy62E24_9axis_220707.csv' #r2 


# Roy62E24_9axis_220617rollpitchyaw
Reader.figure(path=pathway)
raw=Reader.export()
acc=raw[:,1:4]
if caliACC:
    for i in range(len(acc)):
        acc[i]=acc_scale.dot(acc[i]-acc_bias)

gyrX=raw[:,4]-BIAS.GYR_X.value
gyrY=raw[:,5]-BIAS.GYR_Y.value
gyrZ=raw[:,6]-BIAS.GYR_Z.value


# magMagnitude=(magX**2+magY**2+magZ**2)**0.5
# plt.title('magMagnitude')
# plt.plot(magMagnitude)
# plt.show()


raw[:,7]=(raw[:,7]-74.11734393)/208.01637665
raw[:,8]=(raw[:,8]-25.96085566)/159.03207377
raw[:,9]=(raw[:,9]+6.0178284)/115.74314187
# raw[:,7]=(raw[:,7])/208.01637665-74.11734393
# raw[:,8]=(raw[:,8])/159.03207377-25.96085566
# raw[:,9]=(raw[:,9])/115.74314187+6.0178284
acc=raw[:,1:4]
gyr=raw[:,4:7]
mag=raw[:,7:10]
magX=raw[:,7]
magY=raw[:,8]
magZ=raw[:,9]
# center=
#  [74.11734393 25.96085566 -6.0178284 ]
# scale_axis=
#  [208.01637665 159.03207377 115.74314187]
gyrA=gyr
# plt.plot(gyrA)
ts = raw[:,0]
# ts[ts<ts[0]]+=65536

# indexSel = np.all([ts>=startTime,ts<=stopTime], axis=0)


pathway2=pathway.replace("_9axis_", "_quater_")
Reader.figure(path=pathway2)
raw2=Reader.export()
ts2=raw2[:,0]

print('this is ',mag)
fig = plt.figure(figsize=(7, 7))
plt.title('mag_xy_plane')
plt.scatter(mag[:,0],mag[:,1])
# plt.legend(['gz'])
plt.show()


fig = plt.figure(figsize=(7, 7))
plt.title('gz')
plt.plot(gyrZ)
plt.legend(['gz'])
plt.show()
fig = plt.figure(figsize=(7, 7))
plt.title('delt ts & ts2')
plt.plot(np.diff(ts))
plt.plot(np.diff(ts2))
plt.legend(['deltTs'],['deltTs2'])
plt.show()
# ts2[ts2<ts2[0]]+=65536

if ts[0]>=ts2[0]:
    startTime=ts[0]
else:
    startTime=ts2[0]
if len(ts2)>len(ts):
    datalengh=len(ts)
else:
    datalengh=len(ts2)
ind=np.where(ts ==startTime)[0][0]
ind2=np.where(ts ==startTime)[0][0]
ts=ts[ind:ind+datalengh]
ts2=ts2[ind2:ind2+datalengh]

ts=np.arange(0,5*datalengh,5)
ts2=np.arange(0,5*datalengh,5)
plt.show()

fig = plt.figure(figsize=(7, 7))
plt.title('ts & ts2')
plt.plot(ts)
plt.plot(ts2)
plt.legend(['ts_9axis'],['ts_qua'])
plt.show()


# newIndex=np.searchsorted(ts2,ts)
# newIndex[newIndex==ts2.shape[0]]=ts2.shape[0]-1
# print('newIndex shape',newIndex.shape)

accX=acc[ind:ind+datalengh,0]
accY=acc[ind:ind+datalengh,1]
accZ=acc[ind:ind+datalengh,2]
# accX=raw[ind:ind+datalengh,0]
# accY=raw[ind:ind+datalengh,1]
# accZ=raw[ind:ind+datalengh,2]
acc=acc[ind:ind+datalengh]
accA=acc
gyrX=raw[ind:ind+datalengh,4]-BIAS.GYR_X.value
gyrY=raw[ind:ind+datalengh,5]-BIAS.GYR_Y.value
gyrZ=raw[ind:ind+datalengh,6]-BIAS.GYR_Z.value
magX=raw[ind:ind+datalengh,7]
magY=raw[ind:ind+datalengh,8]
magZ=raw[ind:ind+datalengh,9]

# acc_bk=raw[ind:ind+datalengh,[0:3]]

# gyr_bk=raw[ind:ind+datalengh,[3:6]]
# mag_bk=raw[ind:ind+datalengh,[6:9]]
quat=raw2[ind2:ind2+datalengh,1:5]
accX=acc[:,0]
accY=acc[:,1]
accZ=acc[:,2]

fig = plt.figure(figsize=(7, 7))
plt.title('accZ')
plt.plot(accZ)
plt.show()
accA=acc
# Compute accelerometer magnitude
acc_mag = np.sqrt(accX*accX+accY*accY+accZ*accZ)
mag_mag = np.sqrt(magX*magX+magY*magY+magZ*magZ)
fig = plt.figure(figsize=(7, 7))
plt.title('acc magnitude')
plt.grid()
plt.plot(ts,acc_mag)
plt.show()
fig = plt.figure(figsize=(7, 7))
plt.title('acc_z')
plt.grid()
plt.plot(ts,acc[ind:ind+datalengh,2])
plt.grid()
plt.show()
fig = plt.figure(figsize=(7, 7))
plt.title('mag magnitude')
plt.grid()
plt.plot(ts,mag_mag)
plt.show()
# HP filter accelerometer data
filtCutOff = hpcutoff #default 0,001
b, a = signal.butter(order, (2*filtCutOff)/(1/samplePeriod), 'highpass') #1
acc_magFilt = signal.filtfilt(b, a, acc_mag, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
#acc_magFilt = signal.filtfilt(b, a, acc_mag)
acc_magFilt1=np.copy(acc_magFilt)

# Compute absolute value
acc_magFilt = np.abs(acc_magFilt)
acc_magAbs=np.copy(acc_magFilt)
print('acc_magAbs',acc_magAbs)

# LP filter accelerometer data
filtCutOff = lpcutoff  #default 5
b, a = signal.butter(order, (2*filtCutOff)/(1/samplePeriod), 'lowpass') #1
acc_magFilt = signal.filtfilt(b, a, acc_magFilt, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))

#acc_magFilt = signal.filtfilt(b, a, acc_magFilt)

# Threshold detection
# print('acc_magFilt',acc_magFilt)
stationary = acc_magFilt < threshold #default 0.05
# stationary = acc_magFilt < -999999999999 #default 0.05
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
ax1.plot(ts,gyrX,c='r',linewidth=0.5)
ax1.plot(ts,gyrY,c='g',linewidth=0.5)
ax1.plot(ts,gyrZ,c='b',linewidth=0.5)
ax1.set_title("gyroscope")
ax1.set_xlabel("time (s)")
ax1.set_ylabel("angular velocity (degrees/s)")
ax1.legend(["x","y","z"])
ax2.plot(accX,c='r',linewidth=0.5)
ax2.plot(accY,c='g',linewidth=0.5)
ax2.plot(accZ,c='b',linewidth=0.5)
ax2.plot(acc_magFilt,c='k',linestyle=":",linewidth=1)
ax2.plot(stationary,c='k')
ax2.set_title("accelerometer")
ax2.set_xlabel("time (s)")
ax2.set_ylabel("acceleration (g)")
ax2.legend(["x","y","z"])
plt.show(block=False)


# initial convergence
initPeriod = 2
# indexSel = ts<=ts[0]+initPeriod
# a,b,c=Quaternion(1,0,0,0).to_euler_angles_by_wiki()
mahony = ahrs.filters.Mahony(Kp=1, Ki=mahonyKi,KpInit=1, frequency=1/mahonyrate)

if useMahony:
    gyr=np.zeros(3, dtype=np.float64)
    acc = np.array([np.mean(accX), np.mean(accY), np.mean(accZ)])
    quat  = np.zeros((ts.size, 4), dtype=np.float64)
    eu  = np.zeros((ts.size, 3), dtype=np.float64)
    
    q = np.array([1.0,0.0,0.0,0.0], dtype=np.float64)
    # for i in range(0, 2000):
    #     q = mahony.updateIMU(q, gyr=gyr, acc=acc)
    # For all data
    lastmag=-1
    for t in range(0,ts.size):
        if(stationary[t]):
            mahony.Kp = 1
            # mahony.Kp = mahonyKp
        else:
            mahony.Kp = 0
            # mahony.Kp = mahonyKp
        gyr = np.array([gyrX[t],gyrY[t],gyrZ[t]])*np.pi/180
        acc = np.array([accX[t],accY[t],accZ[t]])
        # mag = np.array([magX[t],magY[t],magZ[t]])
        # gyr = np.array([gyrX[t],gyrY[t],gyrZ[t]])*np.pi/180
        # acc = np.array([accX[t],accY[t],accZ[t]])
        # gyr = np.array([gyrY[t],-gyrX[t],gyrZ[t]])*np.pi/180
        # acc = np.array([accY[t],-accX[t],accZ[t]])
        mag = np.array([magX[t],-magY[t],-magZ[t]])
        # mag = np.array([magX[t],-magY[t],-magZ[t]])
        # mag = np.array([-magY[t],-magX[t],-magZ[t]])
        if useMag:
            # quat[t,:]=mahony.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
            
            if mag[2]!=lastmag:
                quat[t,:]=mahony.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
                lastmag=mag[2]
            else:
                quat[t,:]=mahony.updateIMU(q,gyr=gyr,acc=acc)
            if dontswitch:
                quat[t,:]=mahony.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
        else:
            quat[t,:]=mahony.updateIMU(q,gyr=gyr,acc=acc)
        qq=Quaternion(quat[t,:])
        eu[t,:]=qq.to_euler_angles_by_wiki()
        eu[t,:]=eu[t,:]/np.pi*180
# if useMahony:
#     gyr=np.zeros(3, dtype=np.float64)
#     acc = np.array([np.mean(accX), np.mean(accY), np.mean(accZ)])
#     quat  = np.zeros((ts.size, 4), dtype=np.float64)
#     eu2  = np.zeros((ts.size, 3), dtype=np.float64)
    
#     q = np.array([1.0,0.0,0.0,0.0], dtype=np.float64)
#     for i in range(0, 2000):
#         q = mahony.updateIMU(q, gyr=gyr, acc=acc)
#     # For all data
#     lastmag=-1
#     for t in range(0,ts.size):
#         if(stationary[t]):
#             mahony.Kp = 5
#         else:
#             mahony.Kp = 0
#         gyr = np.array([gyrX[t],gyrY[t],gyrZ[t]])*np.pi/180
#         acc = np.array([accX[t],accY[t],accZ[t]])
#         mag = np.array([magX[t],-magY[t],-magZ[t]])
#         # mag = np.array([-magY[t],-magX[t],-magZ[t]])
#         if useMag:
#             # quat[t,:]=mahony.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
#             if mag[2]!=lastmag:
#                 quat[t,:]=mahony.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
#                 lastmag=mag[2]
#             else:
#                 quat[t,:]=mahony.updateIMU(q,gyr=gyr,acc=acc)
#         else:
#             quat[t,:]=mahony.updateIMU(q,gyr=gyr,acc=acc)
#         qq=Quaternion(quat[t,:])
#         eu2[t,:]=qq.to_euler_angles_by_wiki()
#         eu2[t,:]=eu2[t,:]/np.pi*180
# if useMahony:
#     gyr=np.zeros(3, dtype=np.float64)
#     acc = np.array([np.mean(accX), np.mean(accY), np.mean(accZ)])
#     quat  = np.zeros((ts.size, 4), dtype=np.float64)
#     eu3  = np.zeros((ts.size, 3), dtype=np.float64)
    
#     q = np.array([1.0,0.0,0.0,0.0], dtype=np.float64)
#     for i in range(0, 2000):
#         q = mahony.updateIMU(q, gyr=gyr, acc=acc)
#     # For all data
#     lastmag=-1
#     for t in range(0,ts.size):
#         if(stationary[t]):
#             mahony.Kp = 1
#         else:
#             mahony.Kp = 0
#         gyr = np.array([gyrX[t],gyrY[t],gyrZ[t]])*np.pi/180
#         acc = np.array([accX[t],accY[t],accZ[t]])
#         mag = np.array([magX[t],-magY[t],-magZ[t]])
#         if useMag:
#             # quat[t,:]=mahony.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
#             if mag[2]!=lastmag:
#                 quat[t,:]=mahony.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
#                 lastmag=mag[2]
#             else:
#                 quat[t,:]=mahony.updateIMU(q,gyr=gyr,acc=acc)
#         else:
#             quat[t,:]=mahony.updateIMU(q,gyr=gyr,acc=acc)
#         qq=Quaternion(quat[t,:])
#         eu3[t,:]=qq.to_euler_angles_by_wiki()
#         eu3[t,:]=eu3[t,:]/np.pi*180
# x=np.linspace(0,int(0.005*eu.shape[0]),eu.shape[0])
# plt.show()
# fig = plt.figure(figsize=(7, 7))
# plt.title('eu')
# plt.plot(x,eu[:,2]+0.3)
# plt.plot(x,eu2[:,2])
# plt.plot(x,eu3[:,2]-0.3)
# plt.legend(['kp=10','kp=5','kp=1'])
# plt.grid()
# plt.show()
# plt.title('eu')

# 123
# return
# plt.plot(eu[:,0],c='r')
# plt.plot(eu[:,1],c='g')
# plt.plot(x[13500:],eu[13500:,2],c='b')
# plt.legend(["yaw"])
# plt.legend(["roll","pitch","yaw"])
# plt.show()
if useMadgwick:
    # acc=acc_bk
    # gyr=gyr_bk
    # mag=mag_bk
    quat  = np.zeros((ts.size, 4), dtype=np.float64)
    eu=np.zeros((ts.size, 3), dtype=np.float64)
    delt_t=np.ones((acc.shape[0], 1), dtype=np.float64)*samplePeriod

    align_arr=Quaternion(1,0,0,0)
    beta=0.5
    # mag[:,1]=-mag[:,1]
    # mag[:,2]=-mag[:,2]
    
    ##NED system conversion
    acc=acc[:,[1,0,2]]
    acc[:,2]=-acc[:,2]
    gyr=gyr[:,[1,0,2]]
    gyr[:,2]=-gyr[:,2]
    mag=mag[:,[1,0,2]]
    mag[:,0]=-mag[:,0]
    
    # self.qq,self.eu=calc_orientation_mag(self.gyr.values/180*np.pi,self.acc.values,self.mag.values,100,Quaternion(1,0,0,0),self.beta,self.delta_ts.values/1,1)
    quat,eu=calc_orientation_mag(gyr/180*np.pi,acc,mag,1/samplePeriod,align_arr,beta,delt_t/1,1)
    # calc_orientation_mag(gyr/180*np.pi,acc,mag,1/samplePeriod,align_arr,beta,samplePeriod,1)

# -------------------------------------------------------------------------
# Compute translational accelerations

# Rotate body accelerations to Earth frame


acc = []
debug_c=0
for x,y,z,q in zip(accX,accY,accZ,quat):
    acc.append(q_rot(q_conj(q), np.array([x, y, z])))
    # print(debug_c,np.array([x, y, z]))
    # print(debug_c,q)
    # print(debug_c,q_rot(q_conj(q), np.array([x, y, z])))
    # debug_c+=1
# return
newAcc=np.copy(acc)
acc = np.array(acc)
acc = acc - np.array([0,0,1])
acc = acc * 9.81
accNoG=np.copy(acc)
# Compute translational velocities

vel = np.zeros(acc.shape)
for t in range(1,vel.shape[0]):
    vel[t,:] = vel[t-1,:] + acc[t,:]*samplePeriod
    if stationary[t] == True:
        vel[t,:] = np.zeros(3)

# Compute integral drift during non-stationary periods
velDrift = np.zeros(vel.shape)
stationaryStart = np.where(np.diff(stationary.astype(int)) == -1)[0]+1
stationaryEnd = np.where(np.diff(stationary.astype(int)) == 1)[0]+1
for i in range(0,stationaryEnd.shape[0]):
    try:
        driftRate = vel[stationaryEnd[i]-1,:] / (stationaryEnd[i] - stationaryStart[i])
        enum = np.arange(0,stationaryEnd[i]-stationaryStart[i])
        drift = np.array([enum*driftRate[0], enum*driftRate[1], enum*driftRate[2]]).T
        velDrift[stationaryStart[i]:stationaryEnd[i],:] = drift
    except:
        pass
    
if plot_vel:
    fig = plt.figure(figsize=(10, 5))
    plt.plot(vel[:,0],linewidth=0.5)
    plt.plot(vel[:,1],linewidth=0.5)
    plt.plot(vel[:,2],linewidth=0.5)
    
    plt.plot(velDrift[:,0],linewidth=0.5)
    plt.plot(velDrift[:,1],linewidth=0.5)
    plt.plot(velDrift[:,2],linewidth=0.5)
    plt.legend(["vel_x","vel_y","vel_z","velDrift_x","velDrift_y","velDrift_z"])
    plt.title("vel & Drift")
    plt.xlabel("time (s)")
    plt.ylabel("vel & Drift (m/s)")
    plt.show(block=False)
# Remove integral drift
vel_speed=vel
vel = vel - velDrift


# fig = plt.figure(figsize=(10, 5))
# plt.plot(vel[:,0],c='r',linewidth=0.5)
# plt.plot(vel[:,1],c='g',linewidth=0.5)
# plt.plot(vel[:,2],c='b',linewidth=0.5)
# plt.legend(["x","y","z"])
# plt.title("velocity")
# plt.xlabel("time (s)")
# plt.ylabel("velocity (m/s)")
# plt.show(block=False)

if plot_q:
    fig = plt.figure(figsize=(10, 5))
    plt.plot(quat[:,0],c='r',linewidth=0.5)
    plt.plot(quat[:,1],c='g',linewidth=0.5)
    plt.plot(quat[:,2],c='b',linewidth=0.5)
    plt.plot(quat[:,3],c='y',linewidth=0.5)
    plt.legend(["w","x","y","z"])
    plt.title("quaternion")
    plt.xlabel("time (s)")
    plt.ylabel("quaternion")
    plt.show(block=False)
if plot_acc:
    fig = plt.figure(figsize=(10, 5))
    plt.plot(accA[:,0],c='r',linewidth=0.5)
    plt.plot(accA[:,1],c='g',linewidth=0.5)
    plt.plot(accA[:,2],c='b',linewidth=0.5)
    plt.legend(["ax","ay","az"])
    plt.title("acc")
    plt.xlabel("time (s)")
    plt.ylabel("g/s^2")
    plt.show(block=False)


mk=accA[:,0]==-0.000488
mk2=accA[:,1]==-0.000488
mk3=accA[:,2]==-0.000488
accZ[mk3]=-1
fig = plt.figure(figsize=(10, 5))
plt.title("acc-0.000488")

plt.plot(mk,c='r',linewidth=0.5)
plt.plot(mk2,c='g',linewidth=0.5)
plt.plot(mk3,c='b',linewidth=0.5)

plt.show(block=False)
# -------------------------------------------------------------------------
# Compute translational position
pos = np.zeros(vel.shape)
pos_realtime = np.zeros(vel.shape)
# pos_accCali = np.zeros(vel.shape)
for t in range(1,pos_realtime.shape[0]):
    pos_realtime[t,:] = pos_realtime[t-1,:] + vel_speed[t,:]*samplePeriod
    if singlelevel:
        if stationary[t] == True or pos_realtime[t,2]<0:
            pos_realtime[t,2] = 0
for t in range(1,pos.shape[0]):
    pos[t,:] = pos[t-1,:] + vel[t,:]*samplePeriod
    if singlelevel:
        if stationary[t] == True or pos[t,2]<0:
            pos[t,2] = 0

if  plot_pos:
    fig = plt.figure(figsize=(10, 5))
    plt.plot(pos[:,0],c='r',linewidth=0.5)
    plt.plot(pos[:,1],c='g',linewidth=0.5)
    plt.plot(pos[:,2],c='b',linewidth=0.5)
    plt.legend(["x","y","z"])
    plt.title("position")
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.show(block=False)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(pos_realtime[:,0],c='r',linewidth=0.5)
    plt.plot(pos_realtime[:,1],c='g',linewidth=0.5)
    plt.plot(pos_realtime[:,2],c='b',linewidth=0.5)
    plt.legend(["x","y","z"])
    plt.title("position_realtime")
    plt.xlabel("time (s)")
    plt.ylabel("position (m)")
    plt.show(block=False)
# -------------------------------------------------------------------------
# Plot 3D foot trajectory

posPlot = pos
quatPlot = quat

extraTime = 20
onesVector = np.ones(int(extraTime*(1/samplePeriod)))

# Create 6 DOF animation
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d') # Axe3D object
ax.plot(posPlot[:,0],posPlot[:,1],posPlot[:,2])
ax.plot(pos_realtime[:,0],pos_realtime[:,1],pos_realtime[:,2])

ax.set_title("trajectory")
ax.set_xlabel("x position (m)")
ax.set_ylabel("y position (m)")
ax.set_zlabel("z position (m)")
plt.legend(['reprocess','realtime'])
plt.show(block=False)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111) # Axe3D object
ax.plot(posPlot[:,0],posPlot[:,1])
ax.plot(pos_realtime[:,0],pos_realtime[:,1])

plt.legend(['reprocess','realtime'])
ax.set_title("2d trajectory")
ax.set_xlabel("x position (m)")
ax.set_ylabel("y position (m)")
ax.grid()
plt.show(block=False)





fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d') # Axe3D object
ax.plot(posPlot[:,0],posPlot[:,1],posPlot[:,2])
# ax.plot(pos_accCali[:,0],pos_accCali[:,1],pos_accCali[:,2])
ax.set_title("trajectory")
ax.set_xlabel("x position (m)")
ax.set_ylabel("y position (m)")
ax.set_zlabel("z position (m)")
plt.legend(['posPlot'])
plt.show(block=False)


fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d') # Axe3D object
ax.plot(pos_realtime[:,0],pos_realtime[:,1],pos_realtime[:,2])
ax.set_title("trajectory_rel")
ax.set_xlabel("x position (m)")
ax.set_ylabel("y position (m)")
ax.set_zlabel("z position (m)")
# plt.legend(['posPlot','pos_accCali'])
plt.show(block=False)
# xs=np.array([0,1,2,3,4,5])
# ys=np.array([0,1,2,3,4,5])
# xs=posPlot[:,0]
# ys=posPlot[:,1]


# myPlotAnimate=Animation(xs,ys)
# myPlotAnimate=Animation(posPlot[:,0],posPlot[:,1])

# for i in range(len(posPlot[:,0])):
#     time.sleep(0.1)
#     myPlotAnimate.animate(i)
# myPlotAnimate.animateGenerate()


# from matplotlib.animation import FuncAnimation
# fig = plt.figure()
# ax = plt.axes()#limits were     arbitrary
# #line = ax.plot([],[])
# line, = ax.plot([], [], lw=2)

# # initialization function: plot the background of each frame
# def init():
#     line.set_data([], [])
#     return line,


# def animate(i):
#     x = xs[i]
#     y = ys[i]
#     # y1 = real_vec[i] 
#     # y2 = modulus_vec[i]
#     line.set_data(x,y)
#     # line.set_data(x,y1)
#     # line.set_data(x,y2) 
#     return line,

# animation_object = FuncAnimation(fig, animate, init_func= init, frames =xs.shape[0] ,interval = 30, blit = True)

# #turnn this line on to save as mp4
# animation_object.save("name.mp4", fps = 100)
# plt.show()








# fig = plt.figure(figsize=(6, 8))

# ax1 = fig.add_subplot(3,1,1)
# ax2 = fig.add_subplot(3,1,2)
# ax3 = fig.add_subplot(3,1,3)
# # plt.plot(raw[:,9:10]) 
# # plt.plot(newAcc[:,2])#acc_magFilt

# ax1.plot(raw[:,7:8],linewidth=0.5)
# ax1.plot(newAcc[:,0],linewidth=0.5)
# ax1.set_title("new acc_x")
# ax1.set_xlabel("n")
# ax1.set_ylabel("acceleration (g)")
# ax2.plot(raw[:,8:9],linewidth=0.5)
# ax2.plot(newAcc[:,1],linewidth=0.5)
# ax2.set_title("new acc_y")
# ax2.set_xlabel("n")
# ax2.set_ylabel("acceleration (g)")
# ax3.plot(raw[:,9:10],linewidth=0.5)
# ax3.plot(newAcc[:,2],linewidth=0.5)
# ax3.set_title("new acc_z")
# ax3.set_xlabel("n")
# ax3.set_ylabel("acceleration (g)")
# #compare acc
# plt.show(block=False)




#-------------plot add magnitude calculation-------------
# fig = plt.figure(figsize=(10, 5))
# plt.plot(acc_magFilt1,c='r',linewidth=0.5)
# plt.plot(raw[:,16],c='g',linewidth=0.5)
# plt.title('acc_magnitude 1 (hp)')
# plt.grid()
# plt.legend(["reference","estimated"])

# fig = plt.figure(figsize=(10, 5))
# plt.plot(acc_magAbs,c='r',linewidth=0.5)
# plt.plot(raw[:,17],c='g',linewidth=0.5)
# plt.title('acc_magnitude 1 (abs)')
# plt.legend(["reference","estimated"])
# plt.grid()
# fig = plt.figure(figsize=(10, 5))
# plt.plot(acc_magFilt,c='r',linewidth=0.5)
# plt.plot(raw[:,18],c='g',linewidth=0.5)
# plt.plot(np.ones(len(acc_magFilt))*0.05,c='b',linewidth=0.5)

# plt.title('acc_magnitude 1 (lp)')
# plt.grid()
# plt.legend(["reference","estimated"])
#---------------------------------------------------------









#-------------plot velocity  and position calculation-------------
# fig = plt.figure(figsize=(6, 8))
# ax1 = fig.add_subplot(3,1,1)
# ax2 = fig.add_subplot(3,1,2)
# ax3 = fig.add_subplot(3,1,3)
# # plt.plot(raw[:,10:11]) 
# # plt.plot(pos[:,0])#acc_magFilt

# ax1.plot(raw[:,10],c='g',linewidth=0.5)
# ax1.plot(vel[:,0],c='r',linewidth=0.5)
# ax1.legend(["eestimated","referenc"])
# ax1.set_title("vel_x")
# # ax1.set_xlabel("n")
# ax1.set_ylabel("velocity (m/s)")
# ax2.plot(raw[:,11],c='g',linewidth=0.5)
# ax2.plot(vel[:,1],c='r',linewidth=0.5)
# ax2.legend(["eestimated","referenc"])
# ax2.set_title("vel_y")
# # ax2.set_xlabel("n")
# ax2.set_ylabel("velocity (m/s)")
# ax3.plot(raw[:,12],c='g',linewidth=0.5)
# ax3.plot(vel[:,2],c='r',linewidth=0.5)
# ax3.legend(["eestimated","referenc"])
# ax3.set_title("vel_z")
# ax3.set_xlabel("n")
# ax3.set_ylabel("velocity (m/s)")
# #compare acc
# plt.show(block=False)
#---------------------------------------------------------


#-------------plot velocity  and position calculation-------------
# fig = plt.figure(figsize=(6, 8))
# ax1 = fig.add_subplot(3,1,1)
# ax2 = fig.add_subplot(3,1,2)
# ax3 = fig.add_subplot(3,1,3)
# ax1.plot(raw[:,13],c='g',linewidth=0.5)
# ax1.plot(pos[:,0],c='r',linewidth=0.5)
# ax1.legend(["eestimated","referenc"])
# ax1.set_title("pos_x")
# # ax1.set_xlabel("n")
# ax1.set_ylabel("position_x (m)")
# ax2.plot(raw[:,14],c='g',linewidth=0.5)
# ax2.plot(pos[:,2],c='r',linewidth=0.5)
# ax2.legend(["eestimated","referenc"])
# ax2.set_title("pos_y")
# # ax2.set_xlabel("n")
# ax2.set_ylabel("position_y (m)")
# ax3.plot(raw[:,15],c='g',linewidth=0.5)
# ax3.plot(pos[:,1],c='r',linewidth=0.5)
# ax3.legend(["eestimated","referenc"])
# ax3.set_title("pos_z")
# ax3.set_xlabel("n")
# ax3.set_ylabel("position_z (m)")
# #compare acc
# plt.show(block=False)
#---------------------------------------------------------

# plt.show()



# if __name__ == "__main__":
#     main()
