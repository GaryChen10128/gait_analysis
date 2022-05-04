# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:25:49 2019

@author: 180218
"""
#class Init(object):
#    @classmethod
#    def clr_local(cls):
#        for name in dir():
#            if not name.startswith('_'):
#                del locals()[name]
#    @classmethod
#    def clr_global(cls):
#        for name in dir():
#            if not name.startswith('_'):
#                del globals()[name]
#    @classmethod
#    def clr_all(cls):
#        cls.clr_global()
#        cls.clr_local()

def clr_local():
    for name in dir():
        print(name)
        if not name.startswith('_'):
            print('del:',locals()[name])
            del locals()[name]
    print('clr_local')

def clr_global():
    for name in dir():
        if not name.startswith('_'):
            print('del:',globals()[name])
            del globals()[name]
    print('clr_global')

def clr_all():
    clr_global()
    clr_local()
    print('clr_all')
def ipy_clr():
    from IPython import get_ipython
    get_ipython().magic('reset -sf')
if __name__=='__main__':
#    from IPython import get_ipython
#    get_ipython().magic('reset -sf')
#    a=5
    ipy_clr()
    b=1
#    clr_byipython()
#    clr_all()
#    clr_local()
#    clr_global()
#    print('complete')
#    li=dir()
#    for name in dir():
#        print('list:',locals()[name])
#        if not name.startswith('_'):
##            del locals()[name]
#            print('del:',locals()[name])
            
#    for name in dir():
#        if not name.startswith('_'):
#            print(name)
#a=5
#    Init.clr_all()
#    Init.clr_local()a
#    Init.clr_global()
#    import os
#    import sys
##    os.execv(sys.executable,['a=10'])
#    os.execv(sys.executable,['cd .'])
#    os.execv(sys.executable, [sys.executable] + sys.argv)