# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:20:14 2022

@author: 11005080
"""

from package.temp_package import *
from calibraxis import Calibraxis
# %matplotlib inline
# pathway=Reader.get_UpperPath('.')+'/190709github_package_data/accCalibration/RoyAFD72_9axis_220608.csv'
pathway=Reader.get_UpperPath('.')+'/190709github_package_data/accCalibration/Roy6BA93_9axis_220610.csv'
pathway=Reader.get_UpperPath('.')+'/190709github_package_data/accCalibration/Roy62E24_9axis_220614acc_cali.csv'

# C:/Users/11005080/Downloads/[Release] MocapSoftware2.0_v002.05_0613/[Release] MocapSoftware2.0_v002.04_0613/Record/calib2/Roy62E24_9axis_220614acc_cali.csv
# Roy6BA93_9axis_220610
Reader.figure(path=pathway)
raw=Reader.export()
raw=raw[0:-10]
# plt.title('old acc')
# plt.plot(raw[:,1:4])
# plt.show()
acc=np.zeros([len(raw),3], dtype=np.float64)
acc_mag=np.zeros(len(raw))
for i in range(len(raw)):
    acc[i]=np.array(raw[i][1:4], dtype=np.float64)
    acc_mag[i]=(acc[i][0]**2+acc[i][1]**2+acc[i][2]**2)**0.5
    #print(np.array(list[1:4], dtype=np.float64))
mk=np.argwhere(acc_mag<2)[:,0]
acc=acc[mk]

plt.title('old acc')
plt.plot(acc)
plt.show()
#acc=np.array(acc, dtype=np.float64)
c=Calibraxis()
c.add_points(acc)
#c.add_points(np.zeros([9,3]))

c.calibrate_accelerometer()
print('--------------------------------')
print(c.bias_vector)
print(c.scale_factor_matrix)

newAcc=c.batch_apply(acc)
plt.title('new acc')
plt.plot(newAcc)
plt.show()


finalacc=[]
for tturple in newAcc:
    finalacc.append([tturple[0],tturple[1],tturple[2]])
newAcc=np.array(finalacc)
plt.title('acc_x comparison')
plt.plot(acc[:,0])
plt.plot(newAcc[:,0])

plt.show()