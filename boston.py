# -*- coding: utf-8 -*-
"""
Created on Fri May 29 19:11:06 2020

@author: Lenovo
"""


import numpy as np

import os
import pandas as pd

os.chdir(r"C:\Users\Lenovo\Desktop\python\100题玩转numpy")

dataall=pd.read_excel(r'house.xlsx',dtype='float')
np_array= np.array(dataall)
np_array1=np_array.reshape(-1)
np_array2=np.delete(np_array1,np.where(np.isnan(np_array1))[0],axis=0)
np_array3=np_array2.reshape(506,14)
df = pd.DataFrame(np_array3,columns=['CRIM','ZN','INDUS','CHAS',\
                                     'NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LAST','MEDV'])
y_data=df['MEDV'].values
x_data=df.drop('MEDV', axis=1).values

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=25,random_state=0)
lr= LinearRegression()
lr.fit(x_train, y_train)
Y_test=lr.predict(x_test)
Y_train=lr.predict(x_train)
rpingfang=r2_score(Y_train,y_train)

import matplotlib.pyplot as plt 
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
mpl.rcParams['axes.unicode_minus'] = False

plt.scatter(y_train,Y_train,label='xunlian')
plt.scatter(y_test,Y_test,label='ceshi')

plt.grid()
plt.title('boshidun')
plt.legend()
plt.text(41,0,r'$R^2=%.4f$'%rpingfang)
print('线性回归的系数为:\n w = %s \n b = %s' % (lr.coef_, lr.intercept_))

