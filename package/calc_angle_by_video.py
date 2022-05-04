import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def getangle(datapath):
    data = pd.read_csv(datapath)
    #print (data)
    data = data.as_matrix()
    #print (data)
    gvec = data[0:2,:]
    #print ("gvec=\n",gvec)
    g_point1 = np.array([gvec[0,5],gvec[0,6]])
    g_point2 = np.array([gvec[1,5],gvec[1,6]])
    #print (g_point1)
    #print (g_point2)
    gvector = g_point2-g_point1#下往上
    #print (gvector)
    pgvector= np.array([-gvector[1],gvector[0]]) #g_normal vector
    #print (pgvector)
    data = np.delete(data,[0,1],axis=0)
    #print ("nogori\n",data)
    #print (data.shape)
    ear_x=data[::3,5]#間隔3
    ear_y=data[::3,6]
    c9_x =data[1::3,5]#shift1間隔3
    c9_y=data[1::3,6]
    shoulder_x=data[2::3,5]
    shoulder_y=data[2::3,6]
    #print (ear_x)
    #print (c9)
    #print (shoulder)
    neckvector = np.array([ear_x-c9_x,ear_y-c9_y]).transpose()#耳朵到C9
    shouldervector = np.array([c9_x-shoulder_x,c9_y-shoulder_y]).transpose()#C9_shoulder
    #print (neckvector.shape)
    from numpy import (array, dot, arccos, clip)
    from numpy.linalg import norm
    import math
    #print (math.pi)
    for i in range(neckvector.shape[0]):
        u = pgvector#水平向量
        s = shouldervector[i]
        v = neckvector[i]
        c = dot(u,v)/norm(u)/norm(v) # -> cosine of the angle
        cs = dot(u,s)/norm(u)/norm(s)
        angle = arccos(clip(c, -1, 1)) # if you really want the angle
        angles = arccos(clip(cs, -1, 1))
        #print (c)
        trueangle = 180-(angle/2/math.pi*360)
        trueangles = 180-(angles/2/math.pi*360)
        if i ==0 :
            trueangle1 = np.array(trueangle)
            trueangle2 = np.array(trueangles)
        if i > 0:
            trueangle1 = np.append(trueangle1,trueangle)
            trueangle2 = np.append(trueangle2,trueangles)
        #print (trueangle1)

        #print (180-(angle/2/math.pi*360))
        #print ("\n")
    #print (trueangle1)
    #print (trueangle1.shape)
#    plt.title('ear_c7')
#    plt.plot(trueangle1)
#    plt.show()
#    plt.title('c7_shoulder')
#    plt.plot(trueangle2)
#    plt.show()
    return trueangle1,trueangle2
#dataname = input("請輸入檔名\n")
#print ("dataname=",dataname)
#getangle(dataname)

#plt.plot()