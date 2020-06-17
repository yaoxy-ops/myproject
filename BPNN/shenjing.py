# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:40:42 2020

@author: Lenovo
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

inputall = pd.read_excel(r'C:\Users\Lenovo\Desktop\MATLAB\用BP神经网络预测生物质灰分含量\新建 XLSX 工作表.xlsx',sheet_name="input")
outputall =pd.read_excel(r'C:\Users\Lenovo\Desktop\MATLAB\用BP神经网络预测生物质灰分含量\新建 XLSX 工作表.xlsx',sheet_name="output")
print (inputall,outputall)
index = np.array([46,12,55,34,6,14,16,17,8,4])
index3=np.arange(67)
index2=np.delete(index3,index)


x_train =np.array(inputall.iloc[index2,:])  ##预测集划分
x_test =np.array(inputall.iloc[index,:])
y_train =np.array(outputall.iloc[index2,:]).reshape(-1)
y_test =np.array(outputall.iloc[index,:]).reshape(-1)
scaler = StandardScaler()           #归一化
scaler.fit(x_train)
X_train = scaler.transform(x_train)
scaler2 = StandardScaler()
scaler2.fit(x_test)
X_test =scaler2.transform(x_test)   ##


bp = MLPClassifier(hidden_layer_sizes=(40,500,),activation='relu',solver='lbfgs'\
                   ,alpha=0.001, batch_size='auto', learning_rate='constant')
bp.fit(X_train,y_train.astype('int'))
y_validation = bp.predict(X_test)
y_predict = bp.predict(X_train)


plt.figure(1)                           ##画图
plt.scatter(y_train, y_predict)

plt.scatter(y_test, y_validation)

legend=plt.legend(["预测集","验证集"])
x = np.linspace(0,10000, 10000)
y =x
plt.plot(x,y,c='red')

plt.show()
#print (x_train)


