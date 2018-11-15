
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veri=pd.read_csv('x.csv')

aylar=veri[['Aylar']]
Satislar=veri[['Satislar']]

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(aylar,Satislar,test_size =0.33,random_state=0)

from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)