import ahrs
from ahrs.common.orientation import q_prod, q_conj, acc2q, am2q, q2R, q_rot
import pyquaternion
import ximu_python_library.xIMUdataClass as xIMU
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from package.temp_package import *
from enum import Enum

def plotData(quat,vel,pos,vel_realtime,pos_realtime,velDrift,tagNote,eu):
    if plot_advanced_data:
        fig = plt.figure(figsize=(7, 7))
        plt.title('mag_xy_plane')
        plt.scatter(mag[:,0],mag[:,1])
        plotHorizonal=(np.max(mag[:,0])-np.min(mag[:,0]))/2
        plotVertical=(np.max(mag[:,1])-np.min(mag[:,1]))/2
        plt.axvline(x=0, ymin=-plotHorizonal, ymax=plotHorizonal,c='r')
        plt.axhline(y=0, xmin=-plotVertical, xmax=plotVertical,c='r')
        plt.grid()
        plt.show()
        
        fig = plt.figure(figsize=(7, 7))
        plt.title('accZ')
        plt.plot(ts9axis,accZ)
        plt.xlabel('ts (s)')
        plt.show()
        
        fig = plt.figure(figsize=(7, 7))
        plt.title('delt ts & tsQuaternion')
        plt.plot(np.diff(ts9axis))
        plt.plot(np.diff(tsQuaternion))
        plt.legend(['deltTs','delttsQuaternion'])
        plt.xlabel('n')
        plt.show()
        plt.show()
        fig = plt.figure(figsize=(7, 7))
        plt.title('ts_9axis & ts_Quaternion')
        plt.plot(ts9axis)
        plt.plot(tsQuaternion)
        plt.legend(['ts_9axis','ts_qua'])
        plt.xlabel('n')
        plt.show()
        fig = plt.figure(figsize=(7, 7))
        plt.title('gz')
        plt.plot(gyrZ)
        plt.legend(['gz'])
        plt.xlabel('n')
        plt.show()
        fig = plt.figure(figsize=(7, 7))
        plt.title('acc magnitude')
        plt.grid()
        plt.plot(ts9axis,acc_mag)
        plt.xlabel('ts (s)')
        plt.show()
        
        # plot acc_z (optional)
        fig = plt.figure(figsize=(7, 7))
        plt.title('acc_z')
        plt.grid()
        plt.plot(ts9axis,acc[ind:ind+datalengh,2])
        plt.grid()
        plt.xlabel('ts (s)')
        plt.show()
        
        # plot mag's magnitude (optional)
        fig = plt.figure(figsize=(7, 7))
        plt.title('mag magnitude')
        plt.grid()
        plt.plot(ts9axis,mag_mag)
        plt.xlabel('ts (s)')
        plt.show()
    
    if plot_raw6axis:   
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        ax1.plot(ts9axis,gyrX,c='r',linewidth=0.5)
        ax1.plot(ts9axis,gyrY,c='g',linewidth=0.5)
        ax1.plot(ts9axis,gyrZ,c='b',linewidth=0.5)
        
        ax1.set_title("gyroscope")
        # ax1.set_xlabel("time (s)")
        ax1.set_ylabel("angular velocity (degrees/s)")
        ax1.legend(["x","y","z"])
        ax2.plot(ts9axis,accX,c='r',linewidth=0.5)
        ax2.plot(ts9axis,accY,c='g',linewidth=0.5)
        ax2.plot(ts9axis,accZ,c='b',linewidth=0.5)
        ax2.plot(acc_magFilt,c='k',linestyle=":",linewidth=1)
        ax2.plot(stationary,c='k')
        ax2.set_title("accelerometer")
        ax2.set_xlabel("time (s)")
        ax2.set_ylabel("acceleration (g)")
        ax2.legend(["x","y","z"])
        plt.grid()
        plt.show(block=False)
    if plot_eu:      
        fig = plt.figure(figsize=(10, 5))
        plt.plot(ts9axis,eu[:,0],c='r')
        plt.plot(ts9axis,eu[:,1],c='g')
        plt.plot(ts9axis,eu[:,2],c='b')
        plt.grid()
        plt.legend(["roll","pitch","yaw"])
        plt.show()
    
    # -------------------------------------------------------------------------
    if plot_q:
        fig = plt.figure(figsize=(10, 5))
        plt.plot(ts9axis,quat[:,0],c='r',linewidth=0.5)
        plt.plot(ts9axis,quat[:,1],c='g',linewidth=0.5)
        plt.plot(ts9axis,quat[:,2],c='b',linewidth=0.5)
        plt.plot(ts9axis,quat[:,3],c='y',linewidth=0.5)
        plt.legend(["w","x","y","z"])
        plt.title("quaternion"+tagNote)
        plt.xlabel("time (s)")
        plt.ylabel("quaternion")
        plt.show(block=False)
    # if useMahony and plot_q:
    #     fig = plt.figure(figsize=(10, 5))
    #     plt.plot(ts9axis,quat_raw[:,0],linewidth=0.5)
    #     plt.plot(ts9axis,quat_raw[:,1],linewidth=0.5)
    #     plt.plot(ts9axis,quat_raw[:,2],linewidth=0.5)
    #     plt.plot(ts9axis,quat_raw[:,3],linewidth=0.5)
        
    #     plt.plot(ts9axis,quat_mahony[:,0],linewidth=0.5)
    #     plt.plot(ts9axis,quat_mahony[:,1],linewidth=0.5)
    #     plt.plot(ts9axis,quat_mahony[:,2],linewidth=0.5)
    #     plt.plot(ts9axis,quat_mahony[:,3],linewidth=0.5)
    #     plt.legend(["qw","qx","qy","qz","qw_mah","qx_mah","qy_mah","qz_mah"])
    #     plt.title("quaternion_csv_raw")
    #     plt.xlabel("time (s)")
    #     plt.ylabel("quaternion")
    #     plt.show(block=False)
    # -------------------------------------------------------------------------
    if plot_acc:
        fig = plt.figure(figsize=(10, 5))
        plt.plot(ts9axis,accA[:,0],c='r',linewidth=0.5)
        plt.plot(ts9axis,accA[:,1],c='g',linewidth=0.5)
        plt.plot(ts9axis,accA[:,2],c='b',linewidth=0.5)
        plt.legend(["ax","ay","az"])
        plt.title("acc")
        plt.xlabel("time (s)")
        plt.ylabel("g/s^2")
        plt.grid()
        plt.show(block=False)
        
    # -------------------------------------------------------------------------
    if plot_advanced_data:
        mk=accA[:,0]==-0.000488
        mk2=accA[:,1]==-0.000488
        mk3=accA[:,2]==-0.000488
        accZ[mk3]=-1
        fig = plt.figure(figsize=(10, 5))
        plt.title("acc_z=0.000488")
        plt.plot(ts9axis,mk,c='r',linewidth=0.5)
        plt.plot(ts9axis,mk2,c='g',linewidth=0.5)
        plt.plot(ts9axis,mk3,c='b',linewidth=0.5)
        plt.xlabel("time (s)")
        plt.grid()
        plt.show(block=False)
    
    # -------------------------------------------------------------------------
    if plot_vel:
        fig = plt.figure(figsize=(10, 5))
        plt.plot(ts9axis,vel_realtime[:,0],linewidth=0.5)
        plt.plot(ts9axis,vel_realtime[:,1],linewidth=0.5)
        plt.plot(ts9axis,vel_realtime[:,2],linewidth=0.5)
        
        plt.plot(ts9axis,velDrift[:,0],linewidth=0.5)
        plt.plot(ts9axis,velDrift[:,1],linewidth=0.5)
        plt.plot(ts9axis,velDrift[:,2],linewidth=0.5)
        plt.legend(["vel_x","vel_y","vel_z","velDrift_x","velDrift_y","velDrift_z"])
        plt.title("vel & Drift")
        plt.xlabel("time (s)")
        plt.ylabel("vel & Drift (m/s)")
        plt.grid()
        plt.show(block=False)
                
    # -------------------------------------------------------------------------
    if  plot_pos:
        fig = plt.figure(figsize=(10, 5))
        plt.plot(ts9axis,pos[:,0],c='r',linewidth=0.5)
        plt.plot(ts9axis,pos[:,1],c='g',linewidth=0.5)
        plt.plot(ts9axis,pos[:,2],c='b',linewidth=0.5)
        plt.legend(["x","y","z"])
        plt.title("position")
        plt.xlabel("time (s)")
        plt.ylabel("position (m)")
        plt.show(block=False)
        fig = plt.figure(figsize=(10, 5))
        plt.plot(ts9axis,pos_realtime[:,0],c='r',linewidth=0.5)
        plt.plot(ts9axis,pos_realtime[:,1],c='g',linewidth=0.5)
        plt.plot(ts9axis,pos_realtime[:,2],c='b',linewidth=0.5)
        plt.legend(["x","y","z"])
        plt.title("position_realtime")
        plt.xlabel("time (s)")
        plt.ylabel("position (m)")
        plt.grid()
        plt.show(block=False)
    # -------------------------------------------------------------------------
    
    
    
    #------------------------------------------------------------------------
    # Create 6 DOF animation
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d') # Axe3D object
    ax.plot(pos[:,0],pos[:,1],pos[:,2])
    ax.plot(pos_realtime[:,0],pos_realtime[:,1],pos_realtime[:,2])
    ax.set_title("trajectory"+tagNote)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_zlabel("z position (m)")
    plt.legend(['reprocess','realtime'])
    ax.grid()
    plt.show(block=False)
    
    if plot_advanced_data:
    #------------------------------------------------------------------------
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111) # Axe3D object
        ax.plot(pos[:,0],pos[:,1])
        ax.plot(pos_realtime[:,0],pos_realtime[:,1])
        plt.legend(['reprocess','realtime'])
        ax.set_title("2d trajectory"+tagNote)
        ax.set_xlabel("x position (m)")
        ax.set_ylabel("y position (m)")
        ax.grid()
        plt.show(block=False)
    
    
    if plot_advanced_data:
    # Plot 3D foot trajectory
    #------------------------------------------------------------------------
    
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d') # Axe3D object
        ax.plot(pos[:,0],pos[:,1],pos[:,2])
        # ax.plot(pos_accCali[:,0],pos_accCali[:,1],pos_accCali[:,2])
        ax.set_title("trajectory"+tagNote)
        ax.set_xlabel("x position (m)")
        ax.set_ylabel("y position (m)")
        ax.set_zlabel("z position (m)")
        plt.legend(['posPlot'])
        ax.grid()
        plt.show(block=False)
    
    #------------------------------------------------------------------------
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d') # Axe3D object
    ax.plot(pos_realtime[:,0],pos_realtime[:,1],pos_realtime[:,2])
    ax.set_title("trajectory_rel"+tagNote)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_zlabel("z position (m)")
    
    # plt.legend(['posPlot','pos_accCali'])
    
    import matplotlib.patches as patches
    
    # ax.add_patch(
    #      patches.Rectangle(
    #         (0, 0,1 ),
    #         1,
    #         1,
    #         1,
    #         edgecolor = 'blue',
    #         facecolor = 'red',
    #         fill=False      
    #      ) )
    # ax.grid()
    plt.show(block=False)
    
    #------------------------------------------------------------------------
    
    
    
    
    
    
    






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
# ax3.plot(linearAcc[:,2],linewidth=0.5)
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
def algorithm(acc, gyr, mag, quat, useMag, useMahony, useMadgwick):
    # main alrightm as below
    # HP filter accelerometer data
    filtCutOff = hpcutoff #default 0,001
    b, a = signal.butter(order, (2*filtCutOff)/(1/samplePeriod), 'highpass') #1
    acc_magFilt = signal.filtfilt(b, a, acc_mag, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    # acc_magFilt = signal.filtfilt(b, a, acc_mag)
    acc_magFilt1=np.copy(acc_magFilt)
    
    # Compute absolute value
    acc_magFilt = np.abs(acc_magFilt)
    acc_magAbs=np.copy(acc_magFilt)
    
    # LP filter accelerometer data
    filtCutOff = lpcutoff  #default 5
    b, a = signal.butter(order, (2*filtCutOff)/(1/samplePeriod), 'lowpass') #1
    acc_magFilt = signal.filtfilt(b, a, acc_magFilt, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))
    # acc_magFilt = signal.filtfilt(b, a, acc_magFilt)
    
    # Threshold detection
    stationary = acc_magFilt < threshold #default 0.05
    
    
    
    # initial convergence
    # a,b,c=Quaternion(1,0,0,0).to_euler_angles_by_wiki()
    mahony = ahrs.filters.Mahony(Kp=1, Ki=mahonyKi,KpInit=1, frequency=1/samplePeriod)
    quat_raw=quat
    eu  = np.zeros((ts9axis.size, 3), dtype=np.float64)
    if useMahony:
    
        gyr=np.zeros(3, dtype=np.float64)
        acc = np.array([np.mean(accX), np.mean(accY), np.mean(accZ)])
        quat  = np.zeros((ts9axis.size, 4), dtype=np.float64)
        # eu  = np.zeros((ts9axis.size, 3), dtype=np.float64)
        q = np.array([1.0,0.0,0.0,0.0], dtype=np.float64)
        lastmag=-1
        for t in range(0,ts9axis.size):
            if(stationary[t]):
                mahony.Kp = 1
                # mahony.Kp = mahonyKp
            else:
                mahony.Kp = 0
                # mahony.Kp = mahonyKp
            gyr = np.array([gyrX[t],gyrY[t],gyrZ[t]])*np.pi/180
            acc = np.array([accX[t],accY[t],accZ[t]])
            mag = np.array([magX[t],-magY[t],-magZ[t]])
            
            # gyr = np.array([gyrY[t],gyrX[t],-gyrZ[t]])*np.pi/180
            # acc = np.array([accY[t],accX[t],-accZ[t]])
            # mag = np.array([-magY[t],magX[t],magZ[t]])
                        
            # gyr = np.array([gyrX[t],-gyrY[t],-gyrZ[t]])*np.pi/180
            # acc = np.array([accX[t],-accY[t],-accZ[t]])
            # mag = np.array([magX[t],magY[t],magZ[t]])
            
            
            # gyr = np.array([-gyrY[t],-gyrX[t],-gyrZ[t]])*np.pi/180
            # acc = np.array([-accY[t],-accX[t],-accZ[t]])
            # mag = np.array([magY[t],magX[t],-magZ[t]])
            
            # gyr = np.array([-gyrX[t],gyrY[t],-gyrZ[t]])*np.pi/180
            # acc = np.array([-accX[t],accY[t],-accZ[t]])
            # mag = np.array([magX[t],-magY[t],-magZ[t]])
            
            # gyr = np.array([-gyrX[t],gyrY[t],-gyrZ[t]])*np.pi/180
            # acc = np.array([-accX[t],accY[t],-accZ[t]])
            # mag = np.array([-magX[t],-magY[t],magZ[t]])
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
            # eu[t,:]=qq.to_euler123()
            # 
            eu[t,:]=eu[t,:]/np.pi*180
            quat_mahony=quat
    
    
    
    if useMadgwick:
        quat_madgwick  = np.zeros((ts.size, 4), dtype=np.float64)
        eu=np.zeros((ts.size, 3), dtype=np.float64)
        delt_t=np.ones((acc.shape[0], 1), dtype=np.float64)*samplePeriod
        align_arr=Quaternion(1,0,0,0)
        beta=0.5
        ##NED system conversion
        acc=acc[:,[1,0,2]]
        acc[:,2]=-acc[:,2]
        gyr=gyr[:,[1,0,2]]
        gyr[:,2]=-gyr[:,2]
        mag=mag[:,[1,0,2]]
        mag[:,0]=-mag[:,0]
        quat_madgwick,eu=calc_orientation_mag(gyr/180*np.pi,acc,mag,1/samplePeriod,align_arr,beta,delt_t/1,1)
    
    # -------------------------------------------------------------------------
    # Compute translational accelerations
    # Rotate body accelerations to Earth frame
    acc = []
    debug_c=0
    for x,y,z,q in zip(accX,accY,accZ,quat):
        acc.append(q_rot(q_conj(q), np.array([x, y, z])))
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
        
    
        
    # Remove integral drift
    vel_realtime=vel
    vel = vel - velDrift
    
    # -------------------------------------------------------------------------
    # Compute translational position
    pos = np.zeros(vel.shape)
    pos_realtime = np.zeros(vel.shape)
    # pos_accCali = np.zeros(vel.shape)
    for t in range(1,pos_realtime.shape[0]):
        pos_realtime[t,:] = pos_realtime[t-1,:] + vel_realtime[t,:]*samplePeriod
        if singlelevel:
            if stationary[t] == True or pos_realtime[t,2]<0:
                pos_realtime[t,2] = 0
    for t in range(1,pos.shape[0]):
        pos[t,:] = pos[t-1,:] + vel[t,:]*samplePeriod
        if singlelevel:
            if stationary[t] == True or pos[t,2]<0:
                pos[t,2] = 0
    return quat,vel,pos,vel_realtime,pos_realtime,velDrift,acc_magFilt,stationary,eu

if __name__=='__main__':
        
    #-------------raw 9axis's csv data format------------
    #  0   1   2   3   4   5   6   7   8   9
    #  ts  ax  ay  az  gx  gy  gz  mx  my  mz
    
    
    
    #----------raw Quaternion's csv data format----------
    #   0   1   2   3   4   5   6   7   8   9  10  11  12
    #  ts  qw  qx  qy  qz  roll pitch yaw  qw (mag)  qx  qy  qz
    #   0   1:5                     6:9                    10:14                 14:17
    #  ts   quaternion (mag off)    euler (mag off)        quaternion (mag on)   quaternion (mag off)
    
    
    #set plot inline or plot on new window (optional)
    %matplotlib qt
    # %matplotlib inline
    
    #--------setting acc's calibration parameters--------
    acc_bias=np.array([0.01184237,-0.02746599,0.02606265])
    acc_scale=np.array([[0.95566679,-0.0249658,-0.03582581],[-0.0249658,0.95940553,0.01238245],[-0.03582581,0.01238245,0.97620943]])
    
    #--------setting gyr's calibration parameters--------
    class BIAS(Enum):
        GYR_X = -0.74
        GYR_Y = 0.26
        GYR_Z = 0.067
        # GYR_X = -1.9
        # GYR_Y = 0.35
        # GYR_Z = 0.067
    #--------------------plot setting--------------------
    plot_vel=0 # plot velocity's data
    plot_pos=0 # plot postion's data
    plot_q=1 # plot quaternion's data
    plot_acc=0 # plot acc's data
    plot_eu=0 # plot euler_angle's data
    plot_raw6axis=0
    plot_advanced_data=0
    #----------------------------------------------------
    
    
    
    #----------------algorithm parameters---------------- #
    hpcutoff=0.001
    lpcutoff=5
    order=1
    threshold=0.05
    singlelevel=1
    mahonyKp=1
    mahonyKi=0
    dontswitch=0 #important
    
    #----------------customized parameters----------------
    samplePeriod=1/200 #device/rawdata's sample period
    useMahony=0 #是否適用mahony重算quaternion
    useMadgwick=0 #是否使用Madgwick計算Quaternion
    useMag=1 #是否使用磁力計計算Quaternion
    #--------------calibration flag setting--------------
    caliACC=0 #use acc's calibration parameter or not
    caliMAG=0 #use mag's calibration parameter or not
    caliGYR=0 #use gyr's calibration parameter or not
    
    #----------------------------------------------------
    
    pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/nmagr1/Roy62E24_9axis_220627.csv' #r1r2
    
    # 直線"有"干擾
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/onhand/NoDistortion/straight/Roy62E24_9axis_220701.csv' #r1r2
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/onFoot/NoDistortion/straight/Roy62E24_9axis_220701.csv' #r1r2
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/waist/NoDistortion/straight/Roy62E24_9axis_220701.csv' #r1r2
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/onhand/Distortion/straight/Roy62E24_9axis_220701.csv' #r1r2
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/onFoot/Distortion/straight/Roy62E24_9axis_220701.csv' #r1r2
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/waist/Distortion/straight/Roy62E24_9axis_220701.csv' #r1r2
    
    #直線無干擾
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/onhand/NoDistortion/square/Roy62E24_9axis_220701.csv' #r2 
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/onFoot/NoDistortion/square/Roy62E24_9axis_220701.csv' #r2
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/waist/NoDistortion/square/Roy62E24_9axis_220701.csv' #r2 
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/onhand/Distortion/square/Roy62E24_9axis_220701.csv' #r2 
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/onFoot/Distortion/square/Roy62E24_9axis_220701.csv' #r2 
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/waist/Distortion/square/Roy62E24_9axis_220701.csv' #r2 
    
    # 常直線"有"干擾
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/longDistance/Distortion/onHand/Roy62E24_9axis_220701.csv' #r2 
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/longDistance/Distortion/onFoot/Roy62E24_9axis_220701.csv' #r2 
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/longDistance/Distortion/onWaist/Roy62E24_9axis_220701.csv' #r2 
    # 常直線"沒"干擾
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/longDistance/NoDistortion/onFoot/Roy62E24_9axis_220701.csv' #r2 
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/longDistance/NoDistortion/onWaist/Roy62E24_9axis_220701.csv' #r2 
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/longDistance/NoDistortion/onHand/Roy62E24_9axis_220701.csv' #r2 
    
    #靜置table
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/staticTableGameRoy0704/staticTableGameRoy0704/Roy62E24_9axis_220704.csv' #r1r2
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/0701outdoor/longDistance/Distortion/onHand/Roy62E24_9axis_220701.csv' #r2 
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/220707mahony12_7noMagSquare/Roy62E24_9axis_220707.csv' #r2 
    
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/nmagr1/Roy62E24_9axis_220627.csv' #r1r2
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/nmagr2/Roy62E24_9axis_220627.csv' #r1r2
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/magr1/Roy62E24_9axis_220627.csv' #r1r2
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/wordmagr1/Roy62E24_9axis_220627.csv' #r1r2
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/3FOutsideExperiment/wordmagr2/Roy62E24_9axis_220627.csv' #r1r2
    
    # pathway=Reader.get_UpperPath('.')+'/Gait-Tracking-With-x-IMU-Python-master/Roy62E24_9axis_220617rollpitchyaw.csv'  ##大阪 記得改mahony rate

    Reader.figure(path=pathway)
    raw9axis=Reader.export()
    acc=raw9axis[:,1:4]
    if caliACC:
        for i in range(len(acc)):
            acc[i]=acc_scale.dot(acc[i]-acc_bias)
    
    # magMagnitude=(magX**2+magY**2+magZ**2)**0.5
    # plt.title('magMagnitude')
    # plt.plot(magMagnitude)
    # plt.show()
    if caliMAG:
        raw9axis[:,7]=(raw9axis[:,7]-74.11734393)/208.01637665
        raw9axis[:,8]=(raw9axis[:,8]-25.96085566)/159.03207377
        raw9axis[:,9]=(raw9axis[:,9]+6.0178284)/115.74314187
    acc=raw9axis[:,1:4]
    gyr=raw9axis[:,4:7]
    mag=raw9axis[:,7:10]
    magX=raw9axis[:,7]
    magY=raw9axis[:,8]
    magZ=raw9axis[:,9]
    gyrA=gyr
    ts9axis = raw9axis[:,0]
    
    
    
    pathway_quaternion=pathway.replace("_9axis_", "_quater_")
    Reader.figure(path=pathway_quaternion)
    rawQuaternion=Reader.export()
    tsQuaternion=rawQuaternion[:,0]
    
    
    # ---------------reprocess ts for 2 reasons----------------
    # 1. ts would be forced become 0 when mag suffer distortion
    # 2. ts would be reseted to zero if the value increased more then 65536
    if ts9axis[0]>=tsQuaternion[0]:
        startTime=ts9axis[0]
    else:
        startTime=tsQuaternion[0]
    if len(tsQuaternion)>len(ts9axis):
        datalengh=len(ts9axis)
    else:
        datalengh=len(tsQuaternion)
    ind=np.where(ts9axis ==startTime)[0][0]
    ind2=np.where(ts9axis ==startTime)[0][0]
    ts9axis=ts9axis[ind:ind+datalengh]
    tsQuaternion=tsQuaternion[ind2:ind2+datalengh]
    dt=samplePeriod #ts (s)=> ts(ms)
    ts9axis=np.arange(0,dt*datalengh,dt)
    tsQuaternion=np.arange(0,dt*datalengh,dt)
    # ---------------------------------------------------------
    
    
    # -----------align 9axis and quaternion by new ts----------
    # data lenght should be trimed for 2 reasons
    # 1. raw data of 9axis and quternion's data may not be the same
    # 2. raw data of 9axis and quternion's start time may not be the same 
    accY=acc[ind:ind+datalengh,1]
    accZ=acc[ind:ind+datalengh,2]
    # accX=raw[ind:ind+datalengh,0]
    # accY=raw[ind:ind+datalengh,1]
    # accZ=raw[ind:ind+datalengh,2]
    acc=acc[ind:ind+datalengh]
    accA=acc
    gyrX=raw9axis[ind:ind+datalengh,4]
    gyrY=raw9axis[ind:ind+datalengh,5]
    gyrZ=raw9axis[ind:ind+datalengh,6]
    magX=raw9axis[ind:ind+datalengh,7]
    magY=raw9axis[ind:ind+datalengh,8]
    magZ=raw9axis[ind:ind+datalengh,9]
    quat=rawQuaternion[ind2:ind2+datalengh,1:5]
    # ---------------------------------------------------------
    if caliGYR: #optional
        gyrX=raw9axis[:,4]-BIAS.GYR_X.value
        gyrY=raw9axis[:,5]-BIAS.GYR_Y.value
        gyrZ=raw9axis[:,6]-BIAS.GYR_Z.value
    accX=acc[:,0]
    accY=acc[:,1]
    accZ=acc[:,2]
    accA=acc
    # Compute accelerometer magnitude (optional)
    acc_mag = np.sqrt(accX*accX+accY*accY+accZ*accZ)
    mag_mag = np.sqrt(magX*magX+magY*magY+magZ*magZ)
                
    quat_royMahony,vel_royMahony,pos_royMahony,vel_royMahony_realtime,pos_royMahony_realtime,velDrift_royMahony,acc_magFilt,stationary,eu=algorithm(acc,gyr,mag,quat,useMag=0,useMahony=0,useMadgwick=0)
    quat_pyMahony,vel_pyyMahony,pos_pyMahony,vel_pyMahony_realtime,pos_pyMahony_realtime,velDrift_pyMahony,acc_magFilt,stationary,eu=algorithm(acc,gyr,mag,quat,useMag=0,useMahony=1,useMadgwick=0)
    
    plotData(quat_royMahony,vel_royMahony,pos_royMahony,vel_royMahony_realtime,pos_royMahony_realtime,velDrift_royMahony,' (disable mag and calulated by roy`s quaternion)',eu)
    plotData(quat_pyMahony,vel_pyyMahony,pos_pyMahony,vel_pyMahony_realtime,pos_pyMahony_realtime,velDrift_pyMahony,' (disable mag and calulated by python_mahony`s quaternion)',eu)
    
    quat_royMahony,vel_royMahony,pos_royMahony,vel_royMahony_realtime,pos_royMahony_realtime,velDrift_royMahony,acc_magFilt,stationary,eu=algorithm(acc,gyr,mag,quat,useMag=1,useMahony=0,useMadgwick=0)
    quat_pyMahony,vel_pyyMahony,pos_pyMahony,vel_pyMahony_realtime,pos_pyMahony_realtime,velDrift_pyMahony,acc_magFilt,stationary,eu=algorithm(acc,gyr,mag,quat,useMag=1,useMahony=1,useMadgwick=0)
    
    
    plotData(quat_royMahony,vel_royMahony,pos_royMahony,vel_royMahony_realtime,pos_royMahony_realtime,velDrift_royMahony,' (use mag to calulated by roy`s quaternion)',eu)
    plotData(quat_pyMahony,vel_pyyMahony,pos_pyMahony,vel_pyMahony_realtime,pos_pyMahony_realtime,velDrift_pyMahony,' (use mag to calulated by python_mahony`s quaternion)',eu)
    