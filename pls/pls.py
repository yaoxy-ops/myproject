# -*- coding: utf-8 -*-
"""
Created on Wed May 20 21:48:47 2020

@author: Lenovo
"""


import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\Lenovo\\Desktop\\python\\python做PLS")
input1 = pd.read_excel('ASH_66.xlsx',sheet_name = 0)
input1.to_csv('ASH_pls_input.csv')
##print(input1)
reference = pd.read_excel('ASH_66.xlsx',sheet_name = 1)
reference.to_csv('ASH_pls_output.csv')
##print(reference)
x_train,x_test,y_train,y_test = train_test_split(input1,reference,test_size=10,random_state=0)
##print(y_test)
pls = PLSRegression(n_components=5)
pls.fit(x_train,y_train)
y_predict=pls.predict(x_train)
out_predict =pls.predict(x_test)
##print(y_train,y_predict,y_test,out_predict)
plt.figure(1)
plt.scatter(y_train, y_predict)

plt.scatter(y_test, out_predict)

legend=plt.legend(["预测集","验证集"])
x = np.linspace(0,10, 1000)
y =x
plt.plot(x,y,c='red')

plt.show()