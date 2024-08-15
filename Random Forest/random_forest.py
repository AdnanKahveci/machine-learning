# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:55:09 2024

@author: kahve
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('maaslar.csv')
#test
print(veriler)

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]

#numpy array donusumu
X = x.values
Y = y.values

from sklearn.ensemble import RandomForestRegressor

rf_reg  = RandomForestRegressor(n_estimators= 10,random_state=0)
rf_reg.fit(X, Y.ravel())

print(rf_reg.predict([[6.6]]))

plt.scatter(X, Y)
plt.plot(X, rf_reg.predict(X))