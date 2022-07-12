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
# %matplotlib inline
filePath = 'datasets/spiralStairs'
startTime = 0
stopTime = 500
samplePeriod = 1/256

# def main():

Writer.figure(path='0502csvWriteTest.csv')
from enum import Enum
# class BIAS(Enum):
#     GYR_X = 0
#     GYR_Y = 0
#     GYR_Z = 0
#     # GYR_X = -0.457909625
#     # GYR_Y = 0.226978555
#     # GYR_Z = -0.238626755
# acc paramters
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
#------------------origin parameters-----------------

# samplePeriod=1/256
# hpcutoff=0.001
# lpcutoff=5
# order=1
# threshold=0.05
velprocess=True
useMahony=True
useMag=False
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
useMag=True
useMadgwick=False
singlelevel=True
mahonyrate=samplePeriod
#----------------------------------------------------

derate=1

# time = xIMUdata.CalInertialAndMagneticData.Time
gyrX = xIMUdata.CalInertialAndMagneticData.gyroscope[:,0]
gyrY = xIMUdata.CalInertialAndMagneticData.gyroscope[:,1]
gyrZ = xIMUdata.CalInertialAndMagneticData.gyroscope[:,2]
accX = xIMUdata.CalInertialAndMagneticData.accelerometer[:,0]
accY = xIMUdata.CalInertialAndMagneticData.accelerometer[:,1]
accZ = xIMUdata.CalInertialAndMagneticData.accelerometer[:,2]
sampleACC=accX
sampleGYR=gyrX
# indexSel = np.all([time>=startTime,time<=stopTime], axis=0)
# time = time[indexSel]
# gyrX = gyrX[indexSel]
# gyrY = gyrY[indexSel]
# gyrZ = gyrZ[indexSel]
# accX = accX[indexSel]
# accY = accY[indexSel]
# accZ = accZ[indexSel]
# time = time[::derate]
# gyrX = gyrX[::derate]
# gyrY = gyrY[::derate]
# gyrZ = gyrZ[::derate]
# accX = accX[::derate]
# accY = accY[::derate]
# accZ = accZ[::derate]

#pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/213244077_positionCalculator.csv' ##currentbest 1order
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/130420630_positionCalculator.csv' ##currentbest 4 order
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/162014682_positionCalculator.csv' ##better

# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/162415414_positionCalculator.csv' ##better
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/163247570_positionCalculator.csv' ##better
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/KaptureAFD72_9axis_220606.csv' 
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/KaptureAFD72_9axis_220606walk3.csv' 
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/KaptureAFD72_9axis_220608outSquare3.csv' 
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/RoyAFD72_9axis_220608royOutWalk.csv' 
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/Roy6BA93_9axis_220609walk2.csv' 
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/RoyAFD72_9axis_220613walk2.csv' 
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/Roy6BA93_9axis_220613walk5.csv' 




pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/Roy6BA93_9axis_220613walk5.csv' 
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/Roy62E24_9axis_220614walknomag.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/w8/Roy62E24_9axis_220614.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/w9/Roy62E24_9axis_220620.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/w10/Roy62E24_9axis_220620.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/1w/Roy62E24_9axis_220621.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/w11/Roy62E24_9axis_220621.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/w12/Roy62E24_9axis_220621.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/w13/Roy62E24_9axis_220622_1.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/w14/Roy62E24_9axis_220622_1111.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/w15/Roy62E24_9axis_220622_2222.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/RecordStraight/strsightAgain/Roy6BA93_9axis_220622.csv' 

# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/RecordStraight/straightAgainNomag4/Roy6BA93_9axis_220622.csv'
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220623StraightRecord/s2mag/Roy62E24_9axis_220623.csv'
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220623StraightRecord/s3_3/Roy62E24_9axis_220623.csv'
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/roytdk42686hz100static_8Froom7min/Roy62E24_9axis_220624.csv'
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220627Record200static/Roy62E24_9axis_220627.csv'
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220627bodyTest/220627stom/Roy62E24_9axis_220627.csv'
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220627bodyTest/220227back/Roy62E24_9axis_220627.csv'
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220627wallstreet/nmagr2/Roy62E24_9axis_220627.csv' #r1r2
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220627wallstreet/magr1/Roy62E24_9axis_220627.csv'
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/220627stom/Roy62E24_9axis_220627.csv' #r1r2
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/backstreet1nomag/Roy62E24_9axis_220627.csv' #r1r2
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/backstreet4mag200/Roy62E24_9axis_220627.csv' #r1r2
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/backstreet5mag200/Roy62E24_9axis_220627.csv' #r1r2
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/nmagr1/Roy62E24_9axis_220627.csv' #r1r2
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/nmagr2/Roy62E24_9axis_220627.csv' #r1r2
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/magr1/Roy62E24_9axis_220627.csv' #r1r2
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/wordmagr1/Roy62E24_9axis_220627.csv' #r1r2
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/wordmagr2/Roy62E24_9axis_220627.csv' #r1r2
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220607_froomdusk/froombacknomaground3/Roy62E24_9axis_220627.csv' #r1r2
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220607_froomdusk/froombacknomag3/Roy62E24_9axis_220627.csv' #r1r2


# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220623StraightRecord/s4_3nomag/Roy62E24_9axis_220623.csv'

# 轉角放置干擾
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/magworddistortion/0628magdistor6angle/Roy62E24_9axis_220628.csv'
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/magworddistortion/0628magdistor5angle/Roy62E24_9axis_220628.csv'
# 直線放置干擾
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/magworddistortion/0628magdistor3/Roy62E24_9axis_220628.csv'
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/magworddistortion/0628magdistor4/Roy62E24_9axis_220628.csv'


#沒有干擾
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/magworddistortion/0628magnodistor2/Roy62E24_9axis_220628.csv'
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/magworddistortion/0628magnodistor/Roy62E24_9axis_220628.csv'


Reader.figure(path=pathway)
raw=Reader.export()
# raw=raw[::2]
# raw=raw[:-3]
acc=raw[:,1:4]
# for i in range(len(acc)):
#     acc[i]=acc_scale.dot(acc[i]-acc_bias)
accX=acc[:,0]
accY=acc[:,1]
accZ=acc[:,2]
gyrX=raw[:,4]-BIAS.GYR_X.value
gyrY=raw[:,5]-BIAS.GYR_Y.value
gyrZ=raw[:,6]-BIAS.GYR_Z.value
magX=raw[:,7]
magY=raw[:,8]
magZ=raw[:,9]
plt.title('accZ')
plt.plot(accZ)
plt.show()
# magMagnitude=(magX**2+magY**2+magZ**2)**0.5
# plt.title('magMagnitude')
# plt.plot(magMagnitude)
# plt.show()
acc=raw[:,1:4]
gyr=raw[:,4:7]
mag=raw[:,7:10]
gyrA=gyr
# plt.plot(gyrA)
ts = raw[:,0]
ts[ts<ts[0]]+=65536
plt.title('ts')
plt.plot(ts)
plt.show()
# indexSel = np.all([ts>=startTime,ts<=stopTime], axis=0)
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/KaptureAFD72_quater_220606walk3.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/KaptureAFD72_quater_220608outSquare3.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/RoyAFD72_quater_220608royOutWalk.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/Roy6BA93_quater_220609walk2.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/RoyAFD72_quater_220613walk2.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/Roy6BA93_quater_220613walk5.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/Roy6BA93_quater_220613walk5.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/Roy62E24_quater_220614walknomag.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/w8/Roy62E24_quater_220614.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/w9/Roy62E24_quater_220620.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/w10/Roy62E24_quater_220620.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/1w/Roy62E24_quater_220621.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/w11/Roy62E24_quater_220621.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/w12/Roy62E24_quater_220621.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/w13/Roy62E24_quater_220622_1.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/w14/Roy62E24_quater_220622_1111.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/w15/Roy62E24_quater_220622_2222.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/RecordStraight/strsightAgain/Roy6BA93_quater_220622.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/RecordStraight/straightAgainNomag4/Roy6BA93_quater_220622.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220623StraightRecord/s2mag/Roy62E24_quater_220623.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220623StraightRecord/s3_3/Roy62E24_quater_220623.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/roytdk42686hz100static_8Froom7min/Roy62E24_quater_220624.csv' 
pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220627Record200static/Roy62E24_quater_220627.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220627bodyTest/220227back/Roy62E24_quater_220627.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220627bodyTest/220627stom/Roy62E24_quater_220627.csv' 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220627wallstreet/nmagr2/Roy62E24_quater_220627.csv' #r2 

# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220627wallstreet/magr1/Roy62E24_quater_220627.csv' 

# 3F experiment
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/220627stom/Roy62E24_quater_220627.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/backstreet1nomag/Roy62E24_quater_220627.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/backstreet4mag200/Roy62E24_quater_220627.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/backstreet5mag200/Roy62E24_quater_220627.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/nmagr1/Roy62E24_quater_220627.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/nmagr2/Roy62E24_quater_220627.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/magr1/Roy62E24_quater_220627.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/wordmagr1/Roy62E24_quater_220627.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/wordmagr2/Roy62E24_quater_220627.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220623StraightRecord/s4_3nomag/Roy62E24_quater_220623.csv' 

# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220607_froomdusk/froombacknomaground3/Roy62E24_quater_220627.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220607_froomdusk/froombacknomag3/Roy62E24_quater_220627.csv' #r2 

#轉角放置干擾
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/magworddistortion/0628magdistor6angle/Roy62E24_quater_220628.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/magworddistortion/0628magdistor5angle/Roy62E24_quater_220628.csv' #r2 

#直線干擾
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/magworddistortion/0628magdistor3/Roy62E24_quater_220628.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/magworddistortion/0628magdistor4/Roy62E24_quater_220628.csv' #r2 

#沒有干擾
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/magworddistortion/0628magnodistor2/Roy62E24_quater_220628.csv' #r2 
# pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/magworddistortion/0628magnodistor/Roy62E24_quater_220628.csv' #r2 


Reader.figure(path=pathway)
raw=Reader.export()
# raw=raw[::2]
ts2=raw[:,0]
ts2[ts2<ts2[0]]+=65536
plt.title('ts2')
plt.plot(ts2)
plt.show()
newIndex=np.searchsorted(ts2,ts)
# newIndex2=np.searchsorted(ts2,ts)
# plt.plot(newIndex)
# plt.plot(newIndex2)
# plt.show()
newIndex[newIndex==ts2.shape[0]]=ts2.shape[0]-1
print('newIndex shape',newIndex.shape)
quat=raw[newIndex,1:5]
# quat=raw[:,[5,2,3,4]]
# print('C707d',raw.shape)
## the parameter is being change:
# order
# 3dbcutoff
#
# Compute accelerometer magnitude
acc_mag = np.sqrt(accX*accX+accY*accY+accZ*accZ)

mag_magnitude = np.sqrt(magX*magX+magY*magY+magZ*magZ)
fig = plt.figure(figsize=(7, 7))
plt.title('mag magnitude')
plt.grid()
plt.plot(ts,mag_magnitude)
plt.show()
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
mahony = ahrs.filters.Mahony(Kp=4, Ki=0,KpInit=1, frequency=1/mahonyrate)
if useMahony:
    gyr=np.zeros(3, dtype=np.float64)
    acc = np.array([np.mean(accX), np.mean(accY), np.mean(accZ)])
    quat  = np.zeros((ts.size, 4), dtype=np.float64)
    eu  = np.zeros((ts.size, 3), dtype=np.float64)
    
    q = np.array([1.0,0.0,0.0,0.0], dtype=np.float64)
    for i in range(0, 2000):
        q = mahony.updateIMU(q, gyr=gyr, acc=acc)
    # For all data
    lastmag=-1
    for t in range(0,ts.size):
        if(stationary[t]):
            mahony.Kp = 1
        else:
            mahony.Kp = 0
        gyr = np.array([gyrX[t],gyrY[t],gyrZ[t]])*np.pi/180
        acc = np.array([accX[t],accY[t],accZ[t]])
        # mag = np.array([magX[t],magY[t],magZ[t]])
        # gyr = np.array([gyrX[t],gyrY[t],gyrZ[t]])*np.pi/180
        # acc = np.array([accX[t],accY[t],accZ[t]])
        # gyr = np.array([gyrY[t],-gyrX[t],gyrZ[t]])*np.pi/180
        # acc = np.array([accY[t],-accX[t],accZ[t]])
        # mag = np.array([magX[t],magY[t],magZ[t]])
        mag = np.array([-magX[t],-magY[t],magZ[t]])
        # mag = np.array([-magY[t],-magX[t],-magZ[t]])
        if useMag:
            # quat[t,:]=mahony.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
            if mag[2]!=lastmag:
                quat[t,:]=mahony.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
                lastmag=mag[2]
            else:
                quat[t,:]=mahony.updateIMU(q,gyr=gyr,acc=acc)
        else:
            quat[t,:]=mahony.updateIMU(q,gyr=gyr,acc=acc)
        qq=Quaternion(quat[t,:])
        eu[t,:]=qq.to_euler_angles_by_wiki()
        eu[t,:]=eu[t,:]/np.pi*180
eulog=eu[-2000:]
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
    quat  = np.zeros((ts.size, 4), dtype=np.float64)
    eu=np.zeros((ts.size, 3), dtype=np.float64)
    delt_t=np.ones((ts.size, 1), dtype=np.float64)*samplePeriod
    # acc=np.zeros((ts.size, 3), dtype=np.float64)
    # gyr=np.zeros((ts.size, 3), dtype=np.float64)
    # mag=np.zeros((ts.size, 3), dtype=np.float64)

    # q = np.array([1.0,0.0,0.0,0.0], dtype=np.float64)
    # for t in range(0,ts.size):
    #     if(stationary[t]):
    #         mahony.Kp = 0.5
    #     else:
    #         mahony.Kp = 0
    #     gyr = np.array([gyrX[t],gyrY[t],gyrZ[t]])*np.pi/180
    #     acc = np.array([accX[t],accY[t],accZ[t]])
    #     mag = np.array([magX[t],magZ[t],magY[t]])
    #         quat[t,:]=mahony.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
    #         quat[t,:]=mahony.updateIMU(q,gyr=gyr,acc=acc)
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
pos_speed = np.zeros(vel.shape)
# pos_accCali = np.zeros(vel.shape)
for t in range(1,pos_speed.shape[0]):
    pos_speed[t,:] = pos_speed[t-1,:] + vel_speed[t,:]*samplePeriod
    if singlelevel:
        if stationary[t] == True:
            pos_speed[t,2] = 0
    # if pos_speed[t,2]<0:
    #     pos_speed[t,2]=0

for t in range(1,pos.shape[0]):
    pos[t,:] = pos[t-1,:] + vel[t,:]*samplePeriod
    if singlelevel:
        if stationary[t] == True:
            pos[t,2] = 0
            
# for t in range(1,pos.shape[0]):
#     pos_accCali[t,:] = pos_accCali[t-1,:] + vel[t,:]*samplePeriod
#     if singlelevel:
#         if stationary[t] == True:
#             pos_accCali[t,2] = 0


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
plt.plot(pos_speed[:,0],c='r',linewidth=0.5)
plt.plot(pos_speed[:,1],c='g',linewidth=0.5)
plt.plot(pos_speed[:,2],c='b',linewidth=0.5)
plt.legend(["x","y","z"])
plt.title("position_speed")
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
ax.plot(pos_speed[:,0],pos_speed[:,1],pos_speed[:,2])

ax.set_title("trajectory")
ax.set_xlabel("x position (m)")
ax.set_ylabel("y position (m)")
ax.set_zlabel("z position (m)")
plt.legend(['reprocess','realtime'])
plt.show(block=False)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111) # Axe3D object
ax.plot(posPlot[:,0],posPlot[:,1])
ax.plot(pos_speed[:,0],pos_speed[:,1])

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
plt.legend(['posPlot','pos_accCali'])
plt.show(block=False)

# xs=np.array([0,1,2,3,4,5])
# ys=np.array([0,1,2,3,4,5])
xs=posPlot[:,0]
ys=posPlot[:,1]


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
