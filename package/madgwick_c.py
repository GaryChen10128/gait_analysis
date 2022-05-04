# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 19:05:34 2018

@author: 180218
"""

#// Math library required for `sqrt'
#include <math.h>
import numpy as np
import matplotlib.pyplot as plt
#// System constants
#define 
class madgwickfilter:   
    def __init__(self,sampleperiod,gyroMeasError,gyroMeasDrift,quaternion):
        self.deltat=sampleperiod
        #sampling period in seconds (shown as 1 ms)
        self.gyroMeasError=np.pi * (gyroMeasError / 180.0) 
        #gyroscope measurement error in rad/s (shown as 5 deg/s)
        self.gyroMeasDrift=np.pi * (gyroMeasDrift / 180.0) 
        #gyroscope measurement error in rad/s/s (shown as 0.2f deg/s/s)
        self.beta=(3.0 / 4.0)**0.5 * self.gyroMeasError 
        #compute beta
        self.zeta=(3.0 / 4.0)**0.5 * self.gyroMeasDrift
        #compute zeta
#        a_x=0
#        a_y=0
#        a_z=0      
#        #accelerometer measurements
#        w_x=0
#        w_y=0
#        w_z=0
#        #gyroscope measurements in rad/s
#        m_x=0
#        m_y=0
#        m_z=0#magnetometer measurements
        self.SEq_1=quaternion[0]
        self.SEq_2=quaternion[1]
        self.SEq_3=quaternion[2]
        self.SEq_4=quaternion[3]
        #estimated orientation quaternion elements with initial conditions
        self.b_x = 1
        self.b_z = 0 
        # reference direction of flux in earth frame
        self.w_bx = 0
        self.w_by = 0
        self.w_bz = 0
        self.debug0=[]
        #estimate gyroscope biases error
    def MARGfilterUpdate2(self,gyr, acc, mag, delt_t):
        '''SEqDot_omega_1, SEqDot_omega_2, SEqDot_omega_3, SEqDot_omega_4; #quaternion rate from gyroscopes elements
            f_1, f_2, f_3, f_4, f_5, f_6; // objective function elements
            J_11or24, J_12or23, J_13or22, J_14or21, J_32, J_33, // objective function Jacobian elements
            J_41, J_42, J_43, J_44, J_51, J_52, J_53, J_54, J_61, J_62, J_63, J_64; //
            SEqHatDot_1, SEqHatDot_2, SEqHatDot_3, SEqHatDot_4; // estimated direction of the gyroscope error
            w_err_x, w_err_y, w_err_z; // estimated direction of the gyroscope error (angular)
            h_x, h_y, h_z; // computed flux in the earth frame
            // axulirary variables to avoid reapeated calcualtions'''
        w_x, w_y, w_z=gyr[0],gyr[1],gyr[2]
        a_x, a_y, a_z=acc[0],acc[1],acc[2]
        m_x, m_y, m_z=mag[0],mag[1],mag[2]
        self.deltat=delt_t
        halfSEq_1 = 0.5 * self.SEq_1
        halfSEq_2 = 0.5 * self.SEq_2
        halfSEq_3 = 0.5 * self.SEq_3
        halfSEq_4 = 0.5 * self.SEq_4
        twoSEq_1 = 2.0 * self.SEq_1
        twoSEq_2 = 2.0 * self.SEq_2
        twoSEq_3 = 2.0 * self.SEq_3
        twoSEq_4 = 2.0 * self.SEq_4
        twob_x = 2.0 * self.b_x
        twob_z = 2.0 * self.b_z
        twob_xSEq_1 = 2.0 * self.b_x * self.SEq_1
        twob_xSEq_2 = 2.0 * self.b_x * self.SEq_2
        twob_xSEq_3 = 2.0 * self.b_x * self.SEq_3
        twob_xSEq_4 = 2.0 * self.b_x * self.SEq_4
        twob_zSEq_1 = 2.0 * self.b_z * self.SEq_1
        twob_zSEq_2 = 2.0 * self.b_z * self.SEq_2
        twob_zSEq_3 = 2.0 * self.b_z * self.SEq_3
        twob_zSEq_4 = 2.0 * self.b_z * self.SEq_4
        SEq_1SEq_2=0
        SEq_1SEq_3 = self.SEq_1 * self.SEq_3
        SEq_1SEq_4=0
        SEq_2SEq_3=0
        SEq_2SEq_4 = self.SEq_2 * self.SEq_4
        SEq_3SEq_4=0
        twom_x = 2.0 * m_x
        twom_y = 2.0 * m_y
        twom_z = 2.0 * m_z
#        // normalise the accelerometer measurement
        norm = (a_x * a_x + a_y * a_y + a_z * a_z)**0.5
        a_x /= norm
        a_y /= norm
        a_z /= norm
        norm = (m_x * m_x + m_y * m_y + m_z * m_z)**0.5
        m_x /= norm
        m_y /= norm
        m_z /= norm
#            // compute the objective function and Jacobian
        f_1 = twoSEq_2 * self.SEq_4 - twoSEq_1 * self.SEq_3 - a_x
        f_2 = twoSEq_1 * self.SEq_2 + twoSEq_3 * self.SEq_4 - a_y
        f_3 = 1.0 - twoSEq_2 * self.SEq_2 - twoSEq_3 * self.SEq_3 - a_z
        f_4 = twob_x * (0.5 - self.SEq_3 * self.SEq_3 - self.SEq_4 *self.SEq_4) + twob_z * (SEq_2SEq_4 - SEq_1SEq_3) - m_x
        f_5 = twob_x * (self.SEq_2 * self.SEq_3 - self.SEq_1 * self.SEq_4) + twob_z * (self.SEq_1 * self.SEq_2 + self.SEq_3 * self.SEq_4) - m_y
        f_6 = twob_x * (SEq_1SEq_3 + SEq_2SEq_4) + twob_z * (0.5 - self.SEq_2 * self.SEq_2 - self.SEq_3 * self.SEq_3) - m_z
        J_11or24 = twoSEq_3 #J_11 negated in matrix multiplication 
        J_12or23 = 2.0 * self.SEq_4
        J_13or22 = twoSEq_1 #J_12 negated in matrix multiplication
        J_14or21 = twoSEq_2
        J_32 = 2.0 * J_14or21#negated in matrix multiplication 
        J_33 = 2.0 * J_11or24#negated in matrix multiplication
        J_41 = twob_zSEq_3#negated in matrix multiplication 
        J_42 = twob_zSEq_4
        J_43 = 2.0 * twob_xSEq_3 + twob_zSEq_1#negated in matrix multiplication 
        J_44 = 2.0 * twob_xSEq_4 - twob_zSEq_2#negated in matrix multiplication 
        J_51 = twob_xSEq_4 - twob_zSEq_2#negated in matrix multiplication 
        J_52 = twob_xSEq_3 + twob_zSEq_1
        J_53 = twob_xSEq_2 + twob_zSEq_4
        J_54 = twob_xSEq_1 - twob_zSEq_3#negated in matrix multiplication 
        J_61 = twob_xSEq_3
        J_62 = twob_xSEq_4 - 2.0 * twob_zSEq_2
        J_63 = twob_xSEq_1 - 2.0 * twob_zSEq_3
        J_64 = twob_xSEq_2
            #compute the gradient (matrix multiplication)
        SEqHatDot_1 = J_14or21 * f_2 - J_11or24 * f_1 - J_41 * f_4 - J_51 * f_5 + J_61 * f_6
        SEqHatDot_2 = J_12or23 * f_1 + J_13or22 * f_2 - J_32 * f_3 + J_42 * f_4 + J_52 * f_5 + J_62 * f_6
        SEqHatDot_3 = J_12or23 * f_2 - J_33 * f_3 - J_13or22 * f_1 - J_43 * f_4 + J_53 * f_5 + J_63 * f_6
        SEqHatDot_4 = J_14or21 * f_1 + J_11or24 * f_2 - J_44 * f_4 - J_54 * f_5 + J_64 * f_6
        #// normalise the gradient to estimate direction of the gyroscope error
        norm = (SEqHatDot_1 * SEqHatDot_1 + SEqHatDot_2 * SEqHatDot_2 + SEqHatDot_3 * SEqHatDot_3 + SEqHatDot_4 * SEqHatDot_4)**0.5
        SEqHatDot_1 = SEqHatDot_1 / norm
        SEqHatDot_2 = SEqHatDot_2 / norm
        SEqHatDot_3 = SEqHatDot_3 / norm
        SEqHatDot_4 = SEqHatDot_4 / norm
        #// compute angular estimated direction of the gyroscope error
        w_err_x = twoSEq_1 * SEqHatDot_2 - twoSEq_2 * SEqHatDot_1 - twoSEq_3 * SEqHatDot_4 + twoSEq_4 * SEqHatDot_3
        w_err_y = twoSEq_1 * SEqHatDot_3 + twoSEq_2 * SEqHatDot_4 - twoSEq_3 * SEqHatDot_1 - twoSEq_4 * SEqHatDot_2
        w_err_z = twoSEq_1 * SEqHatDot_4 - twoSEq_2 * SEqHatDot_3 + twoSEq_3 * SEqHatDot_2 - twoSEq_4 * SEqHatDot_1
#           // compute and remove the gyroscope baises
        self.w_bx += w_err_x * self.deltat * self.zeta
        self.w_by += w_err_y * self.deltat * self.zeta
        self.w_bz += w_err_z * self.deltat * self.zeta
        w_x -= self.w_bx
        w_y -= self.w_by
        w_z -= self.w_bz
#            // compute the quaternion rate measured by gyroscopes
        SEqDot_omega_1 = -halfSEq_2 * w_x - halfSEq_3 * w_y - halfSEq_4 * w_z
        SEqDot_omega_2 = halfSEq_1 * w_x + halfSEq_3 * w_z - halfSEq_4 * w_y
        SEqDot_omega_3 = halfSEq_1 * w_y - halfSEq_2 * w_z + halfSEq_4 * w_x
        SEqDot_omega_4 = halfSEq_1 * w_z + halfSEq_2 * w_y - halfSEq_3 * w_x
#            // compute then integrate the estimated quaternion rate
        self.SEq_1 += (SEqDot_omega_1 - (self.beta * SEqHatDot_1)) * self.deltat
        self.SEq_2 += (SEqDot_omega_2 - (self.beta * SEqHatDot_2)) * self.deltat
        self.SEq_3 += (SEqDot_omega_3 - (self.beta * SEqHatDot_3)) * self.deltat
        self.SEq_4 += (SEqDot_omega_4 - (self.beta * SEqHatDot_4)) * self.deltat
#            // normalise quaternion
        norm = (self.SEq_1 * self.SEq_1 + self.SEq_2 * self.SEq_2 + self.SEq_3 * self.SEq_3 + self.SEq_4 * self.SEq_4)**0.5
        self.SEq_1 /= norm
        self.SEq_2 /= norm
        self.SEq_3 /= norm
        self.SEq_4 /= norm
#            // compute flux in the earth frame
        SEq_1SEq_2 = self.SEq_1 * self.SEq_2#recompute axulirary variables
        SEq_1SEq_3 = self.SEq_1 * self.SEq_3
        SEq_1SEq_4 = self.SEq_1 * self.SEq_4
        SEq_3SEq_4 = self.SEq_3 * self.SEq_4
        SEq_2SEq_3 = self.SEq_2 * self.SEq_3
        SEq_2SEq_4 = self.SEq_2 * self.SEq_4
        h_x = twom_x * (0.5 - self.SEq_3 * self.SEq_3 - self.SEq_4 * self.SEq_4) + twom_y * (SEq_2SEq_3 - SEq_1SEq_4) + twom_z * (SEq_2SEq_4 + SEq_1SEq_3)
        h_y = twom_x * (SEq_2SEq_3 + SEq_1SEq_4) + twom_y * (0.5 - self.SEq_2 * self.SEq_2 - self.SEq_4 * self.SEq_4) + twom_z * (SEq_3SEq_4 - SEq_1SEq_2)
        h_z = twom_x * (SEq_2SEq_4 - SEq_1SEq_3) + twom_y * (SEq_3SEq_4 + SEq_1SEq_2) + twom_z * (0.5 - self.SEq_2 * self.SEq_2 - self.SEq_3 * self.SEq_3)
#            // normalise the flux vector to have only components in the x and z
        self.b_x = ((h_x * h_x) + (h_y * h_y))**0.5
        self.b_z = h_z
        self.debug0.append(self.w_bx)
    def MARGfilterUpdate(self,w_x, w_y, w_z, a_x, a_y, a_z, m_x, m_y, m_z):
        #  // local system variables
#            SEqDot_omega_1, SEqDot_omega_2, SEqDot_omega_3, SEqDot_omega_4 #quaternion rate from gyroscopes elements
#float f_1, f_2, f_3, f_4, f_5, f_6; // objective function elements
#float J_11or24, J_12or23, J_13or22, J_14or21, J_32, J_33, // objective function Jacobian elements
#J_41, J_42, J_43, J_44, J_51, J_52, J_53, J_54, J_61, J_62, J_63, J_64; //
#float SEqHatDot_1, SEqHatDot_2, SEqHatDot_3, SEqHatDot_4; // estimated direction of the gyroscope error
#float w_err_x, w_err_y, w_err_z; // estimated direction of the gyroscope error (angular)
#float h_x, h_y, h_z; // computed flux in the earth frame
#// axulirary variables to avoid reapeated calcualtions
        halfSEq_1 = 0.5 * self.SEq_1
        halfSEq_2 = 0.5 * self.SEq_2
        halfSEq_3 = 0.5 * self.SEq_3
        halfSEq_4 = 0.5 * self.SEq_4
        twoSEq_1 = 2.0 * self.SEq_1
        twoSEq_2 = 2.0 * self.SEq_2
        twoSEq_3 = 2.0 * self.SEq_3
        twoSEq_4 = 2.0 * self.SEq_4
        twob_x = 2.0 * self.b_x
        twob_z = 2.0 * self.b_z
        twob_xSEq_1 = 2.0 * self.b_x * self.SEq_1
        twob_xSEq_2 = 2.0 * self.b_x * self.SEq_2
        twob_xSEq_3 = 2.0 * self.b_x * self.SEq_3
        twob_xSEq_4 = 2.0 * self.b_x * self.SEq_4
        twob_zSEq_1 = 2.0 * self.b_z * self.SEq_1
        twob_zSEq_2 = 2.0 * self.b_z * self.SEq_2
        twob_zSEq_3 = 2.0 * self.b_z * self.SEq_3
        twob_zSEq_4 = 2.0 * self.b_z * self.SEq_4
        SEq_1SEq_2=0
        SEq_1SEq_3 = self.SEq_1 * self.SEq_3
        SEq_1SEq_4=0
        SEq_2SEq_3=0
        SEq_2SEq_4 = self.SEq_2 * self.SEq_4
        SEq_3SEq_4=0
        twom_x = 2.0 * m_x
        twom_y = 2.0 * m_y
        twom_z = 2.0 * m_z
#        // normalise the accelerometer measurement
        norm = (a_x * a_x + a_y * a_y + a_z * a_z)**0.5
        a_x /= norm
        a_y /= norm
        a_z /= norm
        norm = (m_x * m_x + m_y * m_y + m_z * m_z)**0.5
        m_x /= norm
        m_y /= norm
        m_z /= norm
#            // compute the objective function and Jacobian
        f_1 = twoSEq_2 * self.SEq_4 - twoSEq_1 * self.SEq_3 - a_x
        f_2 = twoSEq_1 * self.SEq_2 + twoSEq_3 * self.SEq_4 - a_y
        f_3 = 1.0 - twoSEq_2 * self.SEq_2 - twoSEq_3 * self.SEq_3 - a_z
        f_4 = twob_x * (0.5 - self.SEq_3 * self.SEq_3 - self.SEq_4 *self.SEq_4) + twob_z * (SEq_2SEq_4 - SEq_1SEq_3) - m_x
        f_5 = twob_x * (self.SEq_2 * self.SEq_3 - self.SEq_1 * self.SEq_4) + twob_z * (self.SEq_1 * self.SEq_2 + self.SEq_3 * self.SEq_4) - m_y
        f_6 = twob_x * (SEq_1SEq_3 + SEq_2SEq_4) + twob_z * (0.5 - self.SEq_2 * self.SEq_2 - self.SEq_3 * self.SEq_3) - m_z
        J_11or24 = twoSEq_3 #J_11 negated in matrix multiplication
        J_12or23 = 2.0 * self.SEq_4
        J_13or22 = twoSEq_1; #J_12 negated in matrix multiplication
        J_14or21 = twoSEq_2
        J_32 = 2.0 * J_14or21#negated in matrix multiplication
        J_33 = 2.0 * J_11or24#negated in matrix multiplication
        J_41 = twob_zSEq_3#negated in matrix multiplication
        J_42 = twob_zSEq_4
        J_43 = 2.0 * twob_xSEq_3 + twob_zSEq_1#negated in matrix multiplication
        J_44 = 2.0 * twob_xSEq_4 - twob_zSEq_2#negated in matrix multiplication
        J_51 = twob_xSEq_4 - twob_zSEq_2#negated in matrix multiplication
        J_52 = twob_xSEq_3 + twob_zSEq_1
        J_53 = twob_xSEq_2 + twob_zSEq_4
        J_54 = twob_xSEq_1 - twob_zSEq_3#negated in matrix multiplication
        J_61 = twob_xSEq_3
        J_62 = twob_xSEq_4 - 2.0 * twob_zSEq_2
        J_63 = twob_xSEq_1 - 2.0 * twob_zSEq_3
        J_64 = twob_xSEq_2
            #compute the gradient (matrix multiplication)
        SEqHatDot_1 = J_14or21 * f_2 - J_11or24 * f_1 - J_41 * f_4 - J_51 * f_5 + J_61 * f_6
        SEqHatDot_2 = J_12or23 * f_1 + J_13or22 * f_2 - J_32 * f_3 + J_42 * f_4 + J_52 * f_5 + J_62 * f_6
        SEqHatDot_3 = J_12or23 * f_2 - J_33 * f_3 - J_13or22 * f_1 - J_43 * f_4 + J_53 * f_5 + J_63 * f_6
        SEqHatDot_4 = J_14or21 * f_1 + J_11or24 * f_2 - J_44 * f_4 - J_54 * f_5 + J_64 * f_6
        #// normalise the gradient to estimate direction of the gyroscope error
        norm = (SEqHatDot_1 * SEqHatDot_1 + SEqHatDot_2 * SEqHatDot_2 + SEqHatDot_3 * SEqHatDot_3 + SEqHatDot_4 * SEqHatDot_4)**0.5
        SEqHatDot_1 = SEqHatDot_1 / norm
        SEqHatDot_2 = SEqHatDot_2 / norm
        SEqHatDot_3 = SEqHatDot_3 / norm
        SEqHatDot_4 = SEqHatDot_4 / norm
        #// compute angular estimated direction of the gyroscope error
        w_err_x = twoSEq_1 * SEqHatDot_2 - twoSEq_2 * SEqHatDot_1 - twoSEq_3 * SEqHatDot_4 + twoSEq_4 * SEqHatDot_3
        w_err_y = twoSEq_1 * SEqHatDot_3 + twoSEq_2 * SEqHatDot_4 - twoSEq_3 * SEqHatDot_1 - twoSEq_4 * SEqHatDot_2
        w_err_z = twoSEq_1 * SEqHatDot_4 - twoSEq_2 * SEqHatDot_3 + twoSEq_3 * SEqHatDot_2 - twoSEq_4 * SEqHatDot_1
#           // compute and remove the gyroscope baises
        self.w_bx += w_err_x * self.deltat * self.zeta
        self.w_by += w_err_y * self.deltat * self.zeta
        self.w_bz += w_err_z * self.deltat * self.zeta
#        self.w_x -= self.w_bx
#        self.w_y -= self.w_by
#        self.w_z -= self.w_bz
        w_x -= self.w_bx
        w_y -= self.w_by
        w_z -= self.w_bz
#            // compute the quaternion rate measured by gyroscopes
        SEqDot_omega_1 = -halfSEq_2 * w_x - halfSEq_3 * w_y - halfSEq_4 * w_z
        SEqDot_omega_2 = halfSEq_1 * w_x + halfSEq_3 * w_z - halfSEq_4 * w_y
        SEqDot_omega_3 = halfSEq_1 * w_y - halfSEq_2 * w_z + halfSEq_4 * w_x
        SEqDot_omega_4 = halfSEq_1 * w_z + halfSEq_2 * w_y - halfSEq_3 * w_x
#            // compute then integrate the estimated quaternion rate
        self.SEq_1 += (SEqDot_omega_1 - (self.beta * SEqHatDot_1)) * self.deltat
        self.SEq_2 += (SEqDot_omega_2 - (self.beta * SEqHatDot_2)) * self.deltat
        self.SEq_3 += (SEqDot_omega_3 - (self.beta * SEqHatDot_3)) * self.deltat
        self.SEq_4 += (SEqDot_omega_4 - (self.beta * SEqHatDot_4)) * self.deltat
#            // normalise quaternion
        norm = (self.SEq_1 * self.SEq_1 + self.SEq_2 * self.SEq_2 + self.SEq_3 * self.SEq_3 + self.SEq_4 * self.SEq_4)**0.5
        self.SEq_1 /= norm
        self.SEq_2 /= norm
        self.SEq_3 /= norm
        self.SEq_4 /= norm
#            // compute flux in the earth frame
        SEq_1SEq_2 = self.SEq_1 * self.SEq_2#recompute axulirary variables
        SEq_1SEq_3 = self.SEq_1 * self.SEq_3
        SEq_1SEq_4 = self.SEq_1 * self.SEq_4
        SEq_3SEq_4 = self.SEq_3 * self.SEq_4
        SEq_2SEq_3 = self.SEq_2 * self.SEq_3
        SEq_2SEq_4 = self.SEq_2 * self.SEq_4
        h_x = twom_x * (0.5 - self.SEq_3 * self.SEq_3 - self.SEq_4 * self.SEq_4) + twom_y * (SEq_2SEq_3 - SEq_1SEq_4) + twom_z * (SEq_2SEq_4 + SEq_1SEq_3)
        h_y = twom_x * (SEq_2SEq_3 + SEq_1SEq_4) + twom_y * (0.5 - self.SEq_2 * self.SEq_2 - self.SEq_4 * self.SEq_4) + twom_z * (SEq_3SEq_4 - SEq_1SEq_2)
        h_z = twom_x * (SEq_2SEq_4 - SEq_1SEq_3) + twom_y * (SEq_3SEq_4 + SEq_1SEq_2) + twom_z * (0.5 - self.SEq_2 * self.SEq_2 - self.SEq_3 * self.SEq_3)
#            // normalise the flux vector to have only components in the x and z
        self.b_x = ((h_x * h_x) + (h_y * h_y))**0.5
        self.b_z = h_z;
    def to_euler123(self):
        roll = np.arctan2(-2*(self.SEq_3*self.SEq_4 - self.SEq_1*self.SEq_2), self.SEq_1**2 - self.SEq_2**2 - self.SEq_3**2 + self.SEq_4**2)
        pitch = np.arcsin(2*(self.SEq_2*self.SEq_4 + self.SEq_1*self.SEq_2))
        yaw = np.arctan2(-2*(self.SEq_2*self.SEq_3 - self.SEq_1*self.SEq_4), self.SEq_1**2 + self.SEq_2**2 - self.SEq_3**2 - self.SEq_4**2)
        return roll, pitch, yaw
    def to_eulermagwick(self):
        roll = np.arctan2(2*self.SEq_2*self.SEq_3 - 2*self.SEq_1*self.SEq_4, 2*self.SEq_1**2 + 2*self.SEq_2**2 - 1)
        pitch = -np.arcsin(2*self.SEq_2*self.SEq_4 + 2*self.SEq_1*self.SEq_3)
        yaw = np.arctan2(2*self.SEq_3*self.SEq_4 - 2*self.SEq_1*self.SEq_2, 2*self.SEq_1**2 + 2*self.SEq_4**2 - 1)
        return roll, pitch, yaw
    def IMUfilterUpdate(self,w_x, w_y, w_z, a_x, a_y, a_z):
        halfSEq_1 = 0.5 * self.SEq_1
        halfSEq_2 = 0.5 * self.SEq_2
        halfSEq_3 = 0.5 * self.SEq_3
        halfSEq_4 = 0.5 * self.SEq_4
        twoSEq_1 = 2.0 * self.SEq_1
        twoSEq_2 = 2.0 * self.SEq_2
        twoSEq_3 = 2.0 * self.SEq_3
        norm = (a_x * a_x + a_y * a_y + a_z * a_z)**0.5
        a_x /= norm
        a_y /= norm
        a_z /= norm
        f_1 = twoSEq_2 * self.SEq_4 - twoSEq_1 * self.SEq_3 - a_x
        f_2 = twoSEq_1 * self.SEq_2 + twoSEq_3 * self.SEq_4 - a_y
        f_3 = 1.0 - twoSEq_2 * self.SEq_2 - twoSEq_3 * self.SEq_3 - a_z
        J_11or24 = twoSEq_3 #J_11 negated in matrix multiplication
        J_12or23 = 2.0 * self.SEq_4
        J_13or22 = twoSEq_1 #J_12 negated in matrix multiplication
        J_14or21 = twoSEq_2
        J_32 = 2.0 * J_14or21 #negated in matrix multiplication
        J_33 = 2.0 * J_11or24
        SEqHatDot_1 = J_14or21 * f_2 - J_11or24 * f_1
        SEqHatDot_2 = J_12or23 * f_1 + J_13or22 * f_2 - J_32 * f_3
        SEqHatDot_3 = J_12or23 * f_2 - J_33 * f_3 - J_13or22 * f_1
        SEqHatDot_4 = J_14or21 * f_1 + J_11or24 * f_2
        norm = (SEqHatDot_1 * SEqHatDot_1 + SEqHatDot_2 * SEqHatDot_2 + SEqHatDot_3 * SEqHatDot_3 + SEqHatDot_4 * SEqHatDot_4)**0.5
        SEqHatDot_1 /= norm
        SEqHatDot_2 /= norm
        SEqHatDot_3 /= norm
        SEqHatDot_4 /= norm
        
        
        SEqDot_omega_1 = -halfSEq_2 * w_x - halfSEq_3 * w_y - halfSEq_4 * w_z
        SEqDot_omega_2 = halfSEq_1 * w_x + halfSEq_3 * w_z - halfSEq_4 * w_y
        SEqDot_omega_3 = halfSEq_1 * w_y - halfSEq_2 * w_z + halfSEq_4 * w_x
        SEqDot_omega_4 = halfSEq_1 * w_z + halfSEq_2 * w_y - halfSEq_3 * w_x
#        // Compute then integrate the estimated quaternion derrivative
        self.SEq_1 += (SEqDot_omega_1 - (self.beta * SEqHatDot_1)) * self.deltat
        self.SEq_2 += (SEqDot_omega_2 - (self.beta * SEqHatDot_2)) * self.deltat
        self.SEq_3 += (SEqDot_omega_3 - (self.beta * SEqHatDot_3)) * self.deltat
        self.SEq_4 += (SEqDot_omega_4 - (self.beta * SEqHatDot_4)) * self.deltat
#        // Normalise quaternion
        norm = (self.SEq_1 * self.SEq_1 + self.SEq_2 * self.SEq_2 + self.SEq_3 * self.SEq_3 + self.SEq_4 * self.SEq_4)**0.5
        self.SEq_1 /= norm
        self.SEq_2 /= norm
        self.SEq_3 /= norm
        self.SEq_4 /= norm
    def IMUfilterUpdate2(self,w_x, w_y, w_z, a_x, a_y, a_z,delt_t):
        self.deltat=delt_t
        halfSEq_1 = 0.5 * self.SEq_1
        halfSEq_2 = 0.5 * self.SEq_2
        halfSEq_3 = 0.5 * self.SEq_3
        halfSEq_4 = 0.5 * self.SEq_4
        twoSEq_1 = 2.0 * self.SEq_1
        twoSEq_2 = 2.0 * self.SEq_2
        twoSEq_3 = 2.0 * self.SEq_3
        norm = (a_x * a_x + a_y * a_y + a_z * a_z)**0.5
        a_x /= norm
        a_y /= norm
        a_z /= norm
        f_1 = twoSEq_2 * self.SEq_4 - twoSEq_1 * self.SEq_3 - a_x
        f_2 = twoSEq_1 * self.SEq_2 + twoSEq_3 * self.SEq_4 - a_y
        f_3 = 1.0 - twoSEq_2 * self.SEq_2 - twoSEq_3 * self.SEq_3 - a_z
        J_11or24 = twoSEq_3 #J_11 negated in matrix multiplication
        J_12or23 = 2.0 * self.SEq_4
        J_13or22 = twoSEq_1 #J_12 negated in matrix multiplication
        J_14or21 = twoSEq_2
        J_32 = 2.0 * J_14or21 #negated in matrix multiplication
        J_33 = 2.0 * J_11or24
        SEqHatDot_1 = J_14or21 * f_2 - J_11or24 * f_1
        SEqHatDot_2 = J_12or23 * f_1 + J_13or22 * f_2 - J_32 * f_3
        SEqHatDot_3 = J_12or23 * f_2 - J_33 * f_3 - J_13or22 * f_1
        SEqHatDot_4 = J_14or21 * f_1 + J_11or24 * f_2
        norm = (SEqHatDot_1 * SEqHatDot_1 + SEqHatDot_2 * SEqHatDot_2 + SEqHatDot_3 * SEqHatDot_3 + SEqHatDot_4 * SEqHatDot_4)**0.5
        SEqHatDot_1 /= norm
        SEqHatDot_2 /= norm
        SEqHatDot_3 /= norm
        SEqHatDot_4 /= norm
        
        
        SEqDot_omega_1 = -halfSEq_2 * w_x - halfSEq_3 * w_y - halfSEq_4 * w_z
        SEqDot_omega_2 = halfSEq_1 * w_x + halfSEq_3 * w_z - halfSEq_4 * w_y
        SEqDot_omega_3 = halfSEq_1 * w_y - halfSEq_2 * w_z + halfSEq_4 * w_x
        SEqDot_omega_4 = halfSEq_1 * w_z + halfSEq_2 * w_y - halfSEq_3 * w_x
#        // Compute then integrate the estimated quaternion derrivative
        self.SEq_1 += (SEqDot_omega_1 - (self.beta * SEqHatDot_1)) * self.deltat
        self.SEq_2 += (SEqDot_omega_2 - (self.beta * SEqHatDot_2)) * self.deltat
        self.SEq_3 += (SEqDot_omega_3 - (self.beta * SEqHatDot_3)) * self.deltat
        self.SEq_4 += (SEqDot_omega_4 - (self.beta * SEqHatDot_4)) * self.deltat
#        // Normalise quaternion
        norm = (self.SEq_1 * self.SEq_1 + self.SEq_2 * self.SEq_2 + self.SEq_3 * self.SEq_3 + self.SEq_4 * self.SEq_4)**0.5
        self.SEq_1 /= norm
        self.SEq_2 /= norm
        self.SEq_3 /= norm
        self.SEq_4 /= norm
    def to_eulerwiki(self):
        roll = np.arctan2(2*(self.SEq_1*self.SEq_2 + self.SEq_3*self.SEq_4), 1-2*(self.SEq_2**2 + self.SEq_3**2))
        pitch = np.arcsin(2*(self.SEq_1*self.SEq_3 - self.SEq_4*self.SEq_2))
        yaw = np.arctan2(2*(self.SEq_1*self.SEq_4 + self.SEq_2*self.SEq_3), 1-2*(self.SEq_3**2 + self.SEq_4**2))
        return roll, pitch, yaw
    def IMUfilterUpdate_csharp(self,gx, gy, gz, ax, ay, az):
        q1 = self.SEq_1
        q2 = self.SEq_2
        q3 = self.SEq_3
        q4 = self.SEq_4 # short name local variable for readability
#            double norm
#            double s1, s2, s3, s4
#            double qDot1, qDot2, qDot3, qDot4

#            // Auxiliary variables to avoid repeated arithmetic
        _2q1 = 2 * q1
        _2q2 = 2 * q2
        _2q3 = 2 * q3
        _2q4 = 2 * q4
        _4q1 = 4 * q1
        _4q2 = 4 * q2
        _4q3 = 4 * q3
        _8q2 = 8 * q2
        _8q3 = 8 * q3
        q1q1 = q1 * q1
        q2q2 = q2 * q2
        q3q3 = q3 * q3
        q4q4 = q4 * q4

#            // Normalise accelerometer measurement
        norm = 1/(ax * ax + ay * ay + az * az)**0.5
#       if (norm == 0) 
#           return#handle NaN
        norm = 1 / norm        #use reciprocal for division
        ax *= norm
        ay *= norm
        az *= norm

#            // Gradient decent algorithm corrective step
        s1 = _4q1 * q3q3 + _2q3 * ax + _4q1 * q2q2 - _2q2 * ay
        s2 = _4q2 * q4q4 - _2q4 * ax + 4 * q1q1 * q2 - _2q1 * ay - _4q2 + _8q2 * q2q2 + _8q2 * q3q3 + _4q2 * az
        s3 = 4 * q1q1 * q3 + _2q1 * ax + _4q3 * q4q4 - _2q4 * ay - _4q3 + _8q3 * q2q2 + _8q3 * q3q3 + _4q3 * az
        s4 = 4 * q2q2 * q4 - _2q2 * ax + 4 * q3q3 * q4 - _2q3 * ay
        norm = 1 / (s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4)**0.5   # // normalise step magnitude
        s1 *= norm
        s2 *= norm
        s3 *= norm
        s4 *= norm

#            // Compute rate of change of quaternion
        qDot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz) - self.beta * s1
        qDot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy) - self.beta * s2
        qDot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx) - self.beta * s3
        qDot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx) - self.beta * s4

#            // Integrate to yield quaternion
        q1 += qDot1 * self.deltat
        q2 += qDot2 * self.deltat
        q3 += qDot3 * self.deltat
        q4 += qDot4 * self.deltat
        norm = 1 / (q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)**0.5#    // normalise quaternion
#        Quaternion[0] = q1 * norm
#        Quaternion[1] = q2 * norm
#        Quaternion[2] = q3 * norm
#        Quaternion[3] = q4 * norm
        self.SEq_1=q1 * norm
        self.SEq_2=q2 * norm
        self.SEq_3=q3 * norm
        self.SEq_4=q4 * norm
    def showbias(self):
        plt.show()
        plt.plot(self.debug0)
        plt.title('debug0')
        plt.show()
        
        

        
            