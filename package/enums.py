# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:47:16 2019

@author: 180218
"""
from enum import Enum
class Color(Enum):
    red=1
    green=2
    blue=3
if __name__=='__main__':
    print(Color.red.name)
    print(Color.red.value)
    print(Color.red==Color.red)