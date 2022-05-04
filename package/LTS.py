# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:49:39 2019

@author: 180218
"""
import numpy as np
import pandas as pd
#class Space(object):
#    def __init__(self,func):
#        
#        self.value=func()
        
class LTS(object):
    @classmethod
    def figure(cls,**kwargs):
        print('parameter setting...')
        for keys in kwargs:
            if keys=='func':
                cls.func=kwargs[keys]
                
    