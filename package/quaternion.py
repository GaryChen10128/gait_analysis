# -*- coding: utf-8 -*-
"""
    Copyright (c) 2015 Jonas Böer, jonas.boeer@student.kit.edu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import numbers
import math



class Quaternion:
    """
    A simple class implementing basic quaternion arithmetic.
    """
    def __init__(self, w_or_q, x=None, y=None, z=None):
        """
        Initializes a Quaternion object
        :param w_or_q: A scalar representing the real part of the quaternion, another Quaternion object or a
                    four-element array containing the quaternion values
        :param x: The first imaginary part if w_or_q is a scalar
        :param y: The second imaginary part if w_or_q is a scalar
        :param z: The third imaginary part if w_or_q is a scalar
        """
        self._q = np.array([1, 0, 0, 0])

        if x is not None and y is not None and z is not None:
            w = w_or_q
            q = np.array([w, x, y, z])
        elif isinstance(w_or_q, Quaternion):
            q = np.array(w_or_q.q)
        else:
            q = np.array(w_or_q)
            if len(q) != 4:
                raise ValueError("Expecting a 4-element array or w x y z as parameters")

        self._set_q(q)

    # Quaternion specific interfaces
    def rotateaxis(self,theta,axis):
        w=np.cos(theta/180*np.pi/2)
        x=-axis[0]*np.sin(theta/180*np.pi/2)
        y=-axis[1]*np.sin(theta/180*np.pi/2)
        z=-axis[2]*np.sin(theta/180*np.pi/2)
        return Quaternion(w,x,y,z)
    def conj(self):
        """
        Returns the conjugate of the quaternion
        :rtype : Quaternion
        :return: the conjugate of the quaternion
        """
        return Quaternion(self._q[0], -self._q[1], -self._q[2], -self._q[3])
    def get_2NED(self):
        """
        Returns the conjugate of the quaternion
        :rtype : Quaternion
        :return: the conjugate of the quaternion
        """
        return Quaternion(self._q[0], -self._q[1], -self._q[2], -self._q[3])

    def to_angle_axis(self):
        """
        Returns the quaternion's rotation represented by an Euler angle and axis.
        If the quaternion is the identity quaternion (1, 0, 0, 0), a rotation along the x axis with angle 0 is returned.
        :return: rad, x, y, z
        """
        if self[0] == 1 and self[1] == 0 and self[2] == 0 and self[3] == 0:
            return 0, 1, 0, 0
        rad = np.arccos(self[0]) * 2
        imaginary_factor = np.sin(rad / 2)
        if abs(imaginary_factor) < 1e-8:
            return 0, 1, 0, 0
        x = self._q[1] / imaginary_factor
        y = self._q[2] / imaginary_factor
        z = self._q[3] / imaginary_factor
        return rad/np.pi*180, x, y, z
    def getTheta(self):
        return np.arccos(self[0])*2
    def normalized(self):
        SEq_1,SEq_2,SEq_3,SEq_4=self[0],self[1],self[2],self[3],
        norm = (SEq_1 * SEq_1 + SEq_2 * SEq_2 + SEq_3 * SEq_3 + SEq_4 * SEq_4)**0.5
        SEq_1 /= norm
        SEq_2 /= norm
        SEq_3 /= norm
        SEq_4 /= norm
        return Quaternion([SEq_1,SEq_2,SEq_3,SEq_4])
    @staticmethod
    def from_angle_axis(rad, x, y, z):
        s = np.sin(rad / 2)
        return Quaternion(np.cos(rad / 2), -x*s, -y*s, -z*s)

    def to_euler_angles(self):
        pitch = np.arcsin(2 * self[1] * self[2] + 2 * self[0] * self[3])
        if np.abs(self[1] * self[2] + self[3] * self[0] - 0.5) < 1e-8:
            roll = 0
            yaw = 2 * np.arctan2(self[1], self[0])
        elif np.abs(self[1] * self[2] + self[3] * self[0] + 0.5) < 1e-8:
            roll = -2 * np.arctan2(self[1], self[0])
            yaw = 0
        else:
            roll = np.arctan2(2 * self[0] * self[1] - 2 * self[2] * self[3], 1 - 2 * self[1] ** 2 - 2 * self[3] ** 2)
            yaw = np.arctan2(2 * self[0] * self[2] - 2 * self[1] * self[3], 1 - 2 * self[2] ** 2 - 2 * self[3] ** 2)
        return roll, pitch, yaw
    def to_euler_angles_by_madgwick(self):
        pitch=-np.arcsin(2 * self[1] * self[3] + 2 * self[0] * self[2])
        roll=np.arctan2(2 * self[2] * self[3] - 2 * self[0] * self[1], 2 * self[0] ** 2 + 2 * self[3] ** 2-1)
        yaw=np.arctan2(2 * self[1] * self[2] - 2 * self[0] * self[3], 2 * self[0] ** 2 + 2 * self[1] ** 2-1)
        return roll, pitch, yaw
    def to_euler123(self):
        roll = np.arctan2(-2*(self[2]*self[3] - self[0]*self[1]), self[0]**2 - self[1]**2 - self[2]**2 + self[3]**2)
        pitch = np.arcsin(2*(self[1]*self[3] + self[0]*self[1]))
        yaw = np.arctan2(-2*(self[1]*self[2] - self[0]*self[3]), self[0]**2 + self[1]**2 - self[2]**2 - self[3]**2)
        return roll, pitch, yaw
    def to_euler123_nosingularity(self):
        sqw = self[0]**2
        sqx = self[1]**2
        sqy = self[2]**2
        sqz = self[3]**2
        heading = np.arctan2(2.0 * (self[1]*self[2] + self[3]*self[0]),(sqx - sqy - sqz + sqw))
        bank = np.arctan2(2.0 * (self[2]*self[3] + self[1]*self[0]),(-sqx - sqy + sqz + sqw))
        attitude = np.arcsin(-2.0 * (self[1]*self[3] - self[2]*self[0])/sqx + sqy + sqz + sqw)
        return bank, attitude, heading
    def to_euler_angles_specialcase(self):
        pitch=np.arcsin(2 * (self[1] * self[2] +  self[0] * self[3]))
        roll=np.arctan2(2 * (self[0] * self[1] -  self[2] * self[3]), 1- 2 * (self[1] ** 2 +  self[3] ** 2))
        yaw=np.arctan2(2 * (self[0] * self[2] - self[1] * self[3]), 1- 2 * (self[2] ** 2 + self[3] ** 2))
        if self[1]*self[2]+self[0]*self[3]==0.5:
            yaw=2*np.arctan2(self[1],self[0])
            roll=0
        if self[1]*self[2]+self[0]*self[3]==-0.5:
            yaw=-2*np.arctan2(self[1],self[0])
            roll=0
        return roll, pitch, yaw
    def to_euler_angles_by_wiki(self):
        roll = np.arctan2(2.0 * (self[0]*self[1] + self[2]*self[3]),1-2*(self[1]**2+self[2]**2))
        yaw = np.arctan2(2.0 * (self[0]*self[3] + self[1]*self[2]),1-2*(self[2]**2+self[3]**2))
        pitch = np.arcsin(2.0 * (self[0]*self[2] - self[3]*self[1]))
        return roll, pitch, yaw
    
#        ysqr = self[2] * self[2]
#        t0 = +2.0 * (self[0] * self[1] + self[2] * self[3])
#        t1 = +1.0 - 2.0 * (self[1] * self[1] + ysqr)
#        X = math.atan2(t0, t1)
#        t2 = +2.0 * (self[0] * self[2] - self[3] * self[1])
#        t2 = +1.0 if t2 > +1.0 else t2
#        t2 = -1.0 if t2 < -1.0 else t2
#        Y = math.asin(t2)
#        t3 = +2.0 * (self[0] * self[3] + self[1] * self[2])
#        t4 = +1.0 - 2.0 * (ysqr + self[3] * self[3])
#        Z = math.atan2(t3, t4)
##        X=X/np.pi*180
##        Y=Y/np.pi*180
##        Z=Z/np.pi*180
#        return X, Y, Z
        
        

    
    
#        pitch = np.arcsin(2 * self[1] * self[2] + 2 * self[0] * self[3])
#        if np.abs(self[1] * self[2] + self[3] * self[0] - 0.5) < 1e-8:
#            roll = 0
#            yaw = 2 * np.arctan2(self[1], self[0])
#        elif np.abs(self[1] * self[2] + self[3] * self[0] + 0.5) < 1e-8:
#            roll = -2 * np.arctan2(self[1], self[0])
#            yaw = 0
#        else:
#            roll = np.arctan2(2 * self[0] * self[1] - 2 * self[2] * self[3], 1 - 2 * self[1] ** 2 - 2 * self[3] ** 2)
#            yaw = np.arctan2(2 * self[0] * self[2] - 2 * self[1] * self[3], 1 - 2 * self[2] ** 2 - 2 * self[3] ** 2)
        return roll, pitch, yaw
    def to_euler_angles_no_Gimbal(self):
#        ysqr = self[2] * self[2]
#        t0 = +2.0 * (self[0] * self[1] + self[2] * self[3])
#        t1 = +1.0 - 2.0 * (self[1] * self[1] + ysqr)
#        X = math.atan2(t0, t1)
#        t2 = +2.0 * (self[0] * self[2] - self[3] * self[1])
#        t2 = +1.0 if t2 > +1.0 else t2
#        t2 = -1.0 if t2 < -1.0 else t2
#        Y = math.asin(t2)
#        t3 = +2.0 * (self[0] * self[3] + self[1] * self[2])
#        t4 = +1.0 - 2.0 * (ysqr + self[3] * self[3])
#        Z = math.atan2(t3, t4)
        
        q=[self[0],self[1],self[2],self[3]]
        X=np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])
        Y=-np.arcsin(2* (q[1] * q[3] - q[0] * q[2]))
        Z = np.arctan2(2 * (q[1] * q[2] + q[0] * q[3]), q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3])
#        X=X/np.pi*180
#        Y=Y/np.pi*180
#        Z=Z/np.pi*180
        return X, Y, Z
    
    def __mul__(self, other):
        """
        multiply the given quaternion with another quaternion or a scalar
        :param other: a Quaternion object or a number
        :return:
        """
        if isinstance(other, Quaternion):
            w = self._q[0]*other._q[0] - self._q[1]*other._q[1] - self._q[2]*other._q[2] - self._q[3]*other._q[3]
            x = self._q[0]*other._q[1] + self._q[1]*other._q[0] + self._q[2]*other._q[3] - self._q[3]*other._q[2]
            y = self._q[0]*other._q[2] - self._q[1]*other._q[3] + self._q[2]*other._q[0] + self._q[3]*other._q[1]
            z = self._q[0]*other._q[3] + self._q[1]*other._q[2] - self._q[2]*other._q[1] + self._q[3]*other._q[0]

            return Quaternion(w, x, y, z)
        elif isinstance(other, numbers.Number):
            q = self._q * other
            return Quaternion(q)

    def __add__(self, other):
        """
        add two quaternions element-wise or add a scalar to each element of the quaternion
        :param other:
        :return:
        """
        if not isinstance(other, Quaternion):
            if len(other) != 4:
                raise TypeError("Quaternions must be added to other quaternions or a 4-element array")
            q = self.q + other
        else:
            q = self.q + other.q

        return Quaternion(q)

    # Implementing other interfaces to ease working with the class

    def _set_q(self, q):
        self._q = q

    def _get_q(self):
        return self._q

    q = property(_get_q, _set_q)

    def __getitem__(self, item):
        return self._q[item]

    def __array__(self):
        return self._q
