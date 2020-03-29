# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:31:24 2020

@author: Muhammad Junaid Iqbal
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



data = pd.read_csv('salary1.csv')
#print(data.head(10))
x=data.iloc[:,0].values
y=data.iloc[:,1].values
x=x.reshape(-1,1)
y=y.reshape(-1,1)



trainingX, testingX, trainingY , testingY=train_test_split(x,y, test_size=0.3, random_state=0)


lin=LinearRegression()
lin.fit(trainingX, trainingY)
#print(lin.fit(trainingX, trainingY))

pred1=lin.predict(testingX)



#print(testingY,"\n...",pred1)
#print(testingY[2])
#print(pred1[2])

b1=lin.coef_
b0=lin.intercept_
#d1=int(input("Enter data: "))
print(b1*4.0+b0)








plt.plot(trainingX,lin.predict(trainingX),color='blue')
plt.scatter(trainingX,trainingY,color='green')
plt.xlabel("Years")
plt.ylabel("pay")
plt.show()

#y=mx+c









