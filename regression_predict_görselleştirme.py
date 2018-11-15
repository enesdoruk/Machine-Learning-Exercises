
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veri=pd.read_csv('x.csv')

aylar=veri[['Aylar']]
Satislar=veri[['Satislar']]

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(aylar,Satislar,test_size =0.33,random_state=0)



from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
tahmin=lr.predict(x_test)
x_train=x_train.sort_index()
y_train=y_train.sort_index()


plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
plt.title('aylara göre satıs')
plt.xlabel('aylar')
plt.ylabel('satislar')