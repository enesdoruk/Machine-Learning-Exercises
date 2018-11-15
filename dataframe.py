import pandas as pd
import numpy as np


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

