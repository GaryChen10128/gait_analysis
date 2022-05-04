# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 14:07:30 2021

@author: 11005080
"""


import warnings
import numpy as np
from numpy.linalg import norm
#from pyquaternion import Quaternion
from .quaternion import Quaternion

class KalmanAHRS:
    samplePeriod = 1/256
    quaternion = Quaternion(1, 0, 0, 0)
    beta = 1
    g=9.8
    flag=1
    p=0
    q=0
    r=0
    ax,ay,az=0,0,0
    mx,my,mz=0,0,0
    Va=0

    lpfGyr = 0.7
    lpfAcc = 0.9
    lpfMag = 0.4
    lpfVa  = 0.7
    x = np.zeros([7, 1]);
    
    cAtt0 = 0.001
    cBias0 = 0.0001
    p=np.array([[],[],[],[]])
    P = np.diag([cAtt0**2,cAtt0**2,cAtt0**2,cAtt0**2,cBias0**2,cBias0**2,cBias0**2])
    
    nProcAtt  = 0.00005;
    nProcBias = 0.000001;
    Q = diag([nProcAtt**2,nProcAtt**2,nProcAtt**2,nProcAtt**2,nProcBias**2,nProcBias**2,nProcBias**2]);
    
    nMeasAcc = 0.05;
    nMeasMag = 0.02;
    Q = diag([nMeasAcc**2,nMeasAcc**2,nMeasAcc**2,nMeasMag**2,nMeasMag**2,nMeasMag**2]);

    R = diag([[1 1 1] * nMeasAcc, [1 1 1] * nMeasMag] .^ 2);
    x[1]= 1.0;
    def __init__(self, sampleperiod=None, quaternion=None, beta=None):
        """
        Initialize the class with the given parameters.
        :param sampleperiod: The sample period
        :param quaternion: Initial quaternion
        :param beta: Algorithm gain beta
        :return:
        """
        if sampleperiod is not None:
            self.samplePeriod = sampleperiod
        if quaternion is not None:
            self.quaternion = quaternion
        if beta is not None:
            self.beta = beta

    def update(self, gyroscope, accelerometer, magnetometer,delte_t):
        """
        Perform one update step with data from a AHRS sensor array
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        :param magnetometer: A three-element array containing the magnetometer data. Can be any unit since a normalized value is used.
        :return:
        """
        q = self.quaternion

        gyroscope = np.array(gyroscope, dtype=float).flatten()
        accelerometer = np.array(accelerometer, dtype=float).flatten()
        magnetometer = np.array(magnetometer, dtype=float).flatten()

        # Normalise accelerometer measurement
        if norm(accelerometer) is 0:
            warnings.warn("accelerometer is zero")
            return
        accelerometer /= norm(accelerometer)

        # Normalise magnetometer measurement
        if norm(magnetometer) is 0:
            warnings.warn("magnetometer is zero")
            return
        magnetometer /= norm(magnetometer)
#        print(magnetometer[0], magnetometer[1], magnetometer[2])
#        print(Quaternion(0, magnetometer[0], magnetometer[1], magnetometer[2]))
#        return 
    
#        h = q * (Quaternion(0, magnetometer[0], magnetometer[1], magnetometer[2]) * q.conjugate)
        h = q * (Quaternion(0, magnetometer[0], magnetometer[1], magnetometer[2]) * q.conj())
        
#        h=h.elements
        b = np.array([0, norm(h[1:3]), 0, h[3]])

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2],
            2*b[1]*(0.5 - q[2]**2 - q[3]**2) + 2*b[3]*(q[1]*q[3] - q[0]*q[2]) - magnetometer[0],
            2*b[1]*(q[1]*q[2] - q[0]*q[3]) + 2*b[3]*(q[0]*q[1] + q[2]*q[3]) - magnetometer[1],
            2*b[1]*(q[0]*q[2] + q[1]*q[3]) + 2*b[3]*(0.5 - q[1]**2 - q[2]**2) - magnetometer[2]
        ])
        j = np.array([
            [-2*q[2],                  2*q[3],                  -2*q[0],                  2*q[1]],
            [2*q[1],                   2*q[0],                  2*q[3],                   2*q[2]],
            [0,                        -4*q[1],                 -4*q[2],                  0],
            [-2*b[3]*q[2],             2*b[3]*q[3],             -4*b[1]*q[2]-2*b[3]*q[0], -4*b[1]*q[3]+2*b[3]*q[1]],
            [-2*b[1]*q[3]+2*b[3]*q[1], 2*b[1]*q[2]+2*b[3]*q[0], 2*b[1]*q[1]+2*b[3]*q[3],  -2*b[1]*q[0]+2*b[3]*q[2]],
            [2*b[1]*q[2],              2*b[1]*q[3]-4*b[3]*q[1], 2*b[1]*q[0]-4*b[3]*q[2],  2*b[1]*q[1]]
        ])
        step = j.T.dot(f)
        step /= norm(step)  # normalise step magnitude

        # Compute rate of change of quaternion
        qdot = (q * Quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])) * 0.5 - self.beta * step.T
        
        # Integrate to yield quaternion
#        q += qdot * self.samplePeriod
        q += qdot * delte_t
        self.quaternion = Quaternion(q / norm(q))  # normalise quaternion
#        self.quaternion=q.normalised
#        print(self.quaternion)
    def setbeata(self,newbeta):
        self.beta=newbeta
    def update_imu(self, gyroscope, accelerometer,delte_t):
        """
        Perform one update step with data from a IMU sensor array
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        """
        q = self.quaternion

        gyroscope = np.array(gyroscope, dtype=float).flatten()
        accelerometer = np.array(accelerometer, dtype=float).flatten()

        # Normalise accelerometer measurement
        if norm(accelerometer) is 0:
            warnings.warn("accelerometer is zero")
            return
        accelerometer /= norm(accelerometer)

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2]
        ])
        j = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0]
        ])
        step = j.T.dot(f)
        step /= norm(step)  # normalise step magnitude

        # Compute rate of change of quaternion
        qdot = (q * Quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])) * 0.5-self.beta * step.T
#        qdot = (q * Quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])) * 0.5-self.beta * step.T
#        print('this is mag...')
        
        q += qdot * delte_t
#        print(q[:])
#        tempq=q.elements
#        tempq+= qdot * delte_t
#        q=Quaternion(tempq)
#        q=Quaternion(qdot)
        self.quaternion = Quaternion(q / norm(q))  # normalise quaternion
#        self.quaternion=q.normalised
    def update_imu2(self, gyroscope, accelerometer,delte_t):
        """
        Perform one update step with data from a IMU sensor array
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        """
        q = self.quaternion

        gyroscope = np.array(gyroscope, dtype=float).flatten()
        accelerometer = np.array(accelerometer, dtype=float).flatten()

        # Normalise accelerometer measurement
        if norm(accelerometer) is 0:
            warnings.warn("accelerometer is zero")
            return
        accelerometer /= norm(accelerometer)
        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2]
        ])
        j = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0]
        ])
        step = j.T.dot(f)
        step /= norm(step)  # normalise step magnitude

        # Compute rate of change of quaternion
        qdot = (q * Quaternion(0, gyroscope[0], gyroscope[1], gyroscope[2])) * 0.5 - self.beta * step.T

        # Integrate to yield quaternion
#        q += qdot * self.samplePeriod
        q += qdot * delte_t
        self.quaternion = Quaternion(q / norm(q))  # normalise quaternion
#        self.quaternion=q.normalised
        