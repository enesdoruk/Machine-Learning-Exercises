import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler= pd.read_csv('eksikveriler.csv')

ulke = veriler.iloc[:,0:1].values

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
le = LabelEncoder()
ulke[:,0]=le.fit_transform(ulke[:,0])
ohe= OneHotEncoder(categorical_features='all')
ulke= ohe.fit_transform(ulke).toarray()


sonuc = pd.DataFrame(data=ulke,index=range(22),columns=['fr','tr','us'])
sonuc2= pd.DataFrame(data=yas,index=range(22),columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet=veriler.iloc[:,-1:].values
print(cinsiyet)

sonuc3= pd.DataFrame(data =cinsiyet,index=range(22),columns=['cinsiyet'])
print(sonuc3)
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)
s2=pd.concat([s,sonuc3],axis=1)
print(s2)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(s,sonuc3,test_size =0.33,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train= sc.fit_transform(x_train)
X_test= sc.fit_transform(x_test)