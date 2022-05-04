# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:24:46 2021

@author: 11005080
"""

import numpy as np
from matplotlib import pyplot as plt
from math import pi

u=100.     #x-position of the center
v=50    #y-position of the center
a=50.     #radius on the x-axis
b=30    #radius on the y-axis
v2=80
b2=70
t = np.linspace(0, 2*pi, 100)

x=u+a*np.cos(t)
y=v+b*np.sin(t)
z=v2+b2*np.sin(t)
plt.plot( x , y )
plt.plot( x , z )
plt.plot( y , z )
plt.grid(color='lightgray',linestyle='--')
plt.show()