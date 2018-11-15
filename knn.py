import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

veriler = pd.read_csv('veriler.csv')

x=veriler.iloc[:,1:4].values
xx=pd.DataFrame(veriler,columns=['boy','kilo','yas'])
y=veriler.iloc[:,4:].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(xx,y,test_size=0.33,random_state=0)

from sklearn.neighbors import KNeigborsClassifier
knn=KNeigborsClassifier(n_neighbors=5,metric='minkowski')
knn.fit(x_train,y_train)
y_pred2=knn.predict(x_test)
