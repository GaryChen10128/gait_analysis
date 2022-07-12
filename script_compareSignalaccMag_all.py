import ahrs
from ahrs.common.orientation import q_prod, q_conj, acc2q, am2q, q2R, q_rot
import pyquaternion
import ximu_python_library.xIMUdataClass as xIMU
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from package.temp_package import *
# filePath = 'datasets/straightLine'
# startTime = 6
# stopTime = 26
# samplePeriod = 1/256

# filePath = 'datasets/stairsAndCorridor'
# startTime = 5
# stopTime = 53
# samplePeriod = 1/256
%matplotlib qt
filePath = 'datasets/spiralStairs'
startTime = 0
stopTime = 500
samplePeriod = 1/256

# def main():

Writer.figure(path='0502csvWriteTest.csv')



samplePeriod = 1/256
xIMUdata = xIMU.xIMUdataClass(filePath, 'InertialMagneticSampleRate', 1/samplePeriod)
#------------------origin parameters-----------------

# samplePeriod=1/256
# hpcutoff=0.001
# lpcutoff=5
# order=1
# threshold=0.05
#----------------------------------------------------
# velprocess=True


#------------------my parameters-----------------
samplePeriod=1/200
hpcutoff=0.001
lpcutoff=5
order=1
threshold=0.05
velprocess=True
useMahony=True
#----------------------------------------------------

derate=1

time = xIMUdata.CalInertialAndMagneticData.Time
gyrX = xIMUdata.CalInertialAndMagneticData.gyroscope[:,0]
gyrY = xIMUdata.CalInertialAndMagneticData.gyroscope[:,1]
gyrZ = xIMUdata.CalInertialAndMagneticData.gyroscope[:,2]
accX = xIMUdata.CalInertialAndMagneticData.accelerometer[:,0]
accY = xIMUdata.CalInertialAndMagneticData.accelerometer[:,1]
accZ = xIMUdata.CalInertialAndMagneticData.accelerometer[:,2]

indexSel = np.all([time>=startTime,time<=stopTime], axis=0)
time = time[indexSel]
gyrX = gyrX[indexSel]
gyrY = gyrY[indexSel]
gyrZ = gyrZ[indexSel]
accX = accX[indexSel]
accY = accY[indexSel]
accZ = accZ[indexSel]
time = time[::derate]
gyrX = gyrX[::derate]
gyrY = gyrY[::derate]
gyrZ = gyrZ[::derate]
accX = accX[::derate]
accY = accY[::derate]
accZ = accZ[::derate]

#pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/213244077_positionCalculator.csv' ##currentbest 1order
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/130420630_positionCalculator.csv' ##currentbest 4 order
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/162014682_positionCalculator.csv' ##better

# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/162415414_positionCalculator.csv' ##better
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/163247570_positionCalculator.csv' ##better
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/KaptureAFD72_9axis_220606.csv' 
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/KaptureAFD72_9axis_220606walk3.csv' 
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/KaptureAFD72_9axis_220608outSquare3.csv' 

Reader.figure(path=pathway)
raw=Reader.export()
# raw=raw[:-3]
accX=raw[:,1]
accY=raw[:,2]
accZ=raw[:,3]
gyrX=raw[:,4]
gyrY=raw[:,5]
gyrZ=raw[:,6]
ts = raw[:,0]
indexSel = np.all([ts>=startTime,ts<=stopTime], axis=0)
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/KaptureAFD72_quater_220606walk3.csv' 
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/KaptureAFD72_quater_220608outSquare3.csv' 

Reader.figure(path=pathway)
raw=Reader.export()
ts2=raw[:,0]*1000+raw[:,1]
plt.plot(ts2)

newIndex=np.searchsorted(ts2,ts)
# newIndex2=np.searchsorted(ts2,ts)
# plt.plot(newIndex)
# plt.plot(newIndex2)
# plt.show()
newIndex[newIndex==ts2.shape[0]]=ts2.shape[0]-1
print('newIndex shape',newIndex.shape)
quat=raw[newIndex,2:6]
# quat=raw[:,[5,2,3,4]]
# print('C707d',raw.shape)
## the parameter is being change:
# order
# 3dbcutoff
#
# Compute accelerometer magnitude
acc_mag = np.sqrt(accX*accX+accY*accY+accZ*accZ)
# print('acc_mag',acc_mag)
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
indexSel = ts<=ts[0]+initPeriod
gyr=np.zeros(3, dtype=np.float64)
acc = np.array([np.mean(accX[indexSel]), np.mean(accY[indexSel]), np.mean(accZ[indexSel])])
mahony = ahrs.filters.Mahony(Kp=1, Ki=0,KpInit=1, frequency=1/samplePeriod)
if useMahony:
    quat  = np.zeros((ts.size, 4), dtype=np.float64)
    q = np.array([1.0,0.0,0.0,0.0], dtype=np.float64)
    for i in range(0, 2000):
        q = mahony.updateIMU(q, gyr=gyr, acc=acc)
    # For all data
    for t in range(0,ts.size):
        if(stationary[t]):
            mahony.Kp = 0.5
        else:
            mahony.Kp = 0
        gyr = np.array([gyrX[t],gyrY[t],gyrZ[t]])*np.pi/180
        acc = np.array([accX[t],accY[t],accZ[t]])
        quat[t,:]=mahony.updateIMU(q,gyr=gyr,acc=acc)

# -------------------------------------------------------------------------
# Compute translational accelerations

# Rotate body accelerations to Earth frame
# quat=raw[:,3:7]

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
# acc[:,2] = acc[:,2] - 9.81

# acc_offset = np.zeros(3)
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

# Remove integral drift
if velprocess:
    vel = vel - velDrift
fig = plt.figure(figsize=(10, 5))
plt.plot(vel[:,0],c='r',linewidth=0.5)
plt.plot(vel[:,1],c='g',linewidth=0.5)
plt.plot(vel[:,2],c='b',linewidth=0.5)
plt.legend(["x","y","z"])
plt.title("velocity")
plt.xlabel("time (s)")
plt.ylabel("velocity (m/s)")
plt.show(block=False)


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
# -------------------------------------------------------------------------
# Compute translational position
pos = np.zeros(vel.shape)
for t in range(1,pos.shape[0]):
    pos[t,:] = pos[t-1,:] + vel[t,:]*samplePeriod

fig = plt.figure(figsize=(10, 5))
plt.plot(pos[:,0],c='r',linewidth=0.5)
plt.plot(pos[:,1],c='g',linewidth=0.5)
plt.plot(pos[:,2],c='b',linewidth=0.5)
plt.legend(["x","y","z"])
plt.title("position")
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
# min_, max_ = np.min(np.min(posPlot,axis=0)), np.max(np.max(posPlot,axis=0))
# ax.set_xlim(min_,max_)
# ax.set_ylim(min_,max_)
# ax.set_zlim(min_,max_)
ax.set_title("trajectory")
ax.set_xlabel("x position (m)")
ax.set_ylabel("y position (m)")
ax.set_zlabel("z position (m)")
plt.show(block=False)

# plt.show()
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111) # Axe3D object
ax.plot(posPlot[:,0],posPlot[:,1])
# min_, max_ = np.min(np.min(posPlot,axis=0)), np.max(np.max(posPlot,axis=0))
# ax.set_xlim(min_,max_)
# ax.set_ylim(min_,max_)
# ax.set_zlim(min_,max_)
ax.set_title("2d trajectory")
ax.set_xlabel("x position (m)")
ax.set_ylabel("y position (m)")
ax.grid()
# ax.set_zlabel("z position (m)")
plt.show(block=False)



fig = plt.figure(figsize=(10, 5))
plt.plot(velDrift[:,0],c='r',linewidth=0.5)
plt.plot(velDrift[:,1],c='g',linewidth=0.5)
plt.plot(velDrift[:,2],c='b',linewidth=0.5)
plt.legend(["velDrift_x","velDrift_y","velDrift_z"])
plt.title("velDrift")
plt.xlabel("time (s)")
plt.ylabel("velDrift (m/s)")


plt.show(block=False)



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
