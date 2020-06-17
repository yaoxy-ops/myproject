# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 23:45:34 2020

@author: Lenovo
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pylab import mpl 
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense
import keras
from tensorflow.keras.optimizers import Adam, SGD

inputa = pd.read_excel(r'C:\Users\Lenovo\Desktop\MATLAB\用BP神经网络预测生物质灰分含量\新建 XLSX 工作表.xlsx',sheet_name="input")
outputa =pd.read_excel(r'C:\Users\Lenovo\Desktop\MATLAB\用BP神经网络预测生物质灰分含量\新建 XLSX 工作表.xlsx',sheet_name="output")
##训练集划分
index = np.array([46,12,55,34,6,14,16,17,8,4])
index3=np.arange(67)
index2=np.delete(index3,index)
inputal=inputa.values
outputall=outputa.values

min_max_scaler = preprocessing.MinMaxScaler()
inputall = min_max_scaler.fit_transform(inputal)

train_x_data = inputall[index2,:]
train_y_data= outputall[index2,:]

test_x_data =inputall[index,:]
test_y_data=outputall[index,:]
          #归一化

  ##

model =tf.keras.Sequential([
    Dense(40,activation='relu',input_dim=12),
    Dense(500,activation='relu'),
    Dense(1,activation='relu'),
    ])

model.compile(optimizer='Adam', loss='mse',metrics=['accuracy'])
model.fit(train_x_data,train_y_data,batch_size=20,epochs=1000)
plt.scatter(train_y_data,model.predict(train_x_data))
plt.scatter(test_y_data,model.predict(test_x_data))
a = np.linspace(0,10000, 10000)
b =a
plt.plot(a,b,c='red')
plt.show()
