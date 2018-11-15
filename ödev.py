import pandas as pd
import numpy as np

Veriler = pd.read_csv('odev_tenis.csv')

hava=Veriler.iloc[:,0:1].values
oyun=Veriler.iloc[:,4:].values
ruzgar=Veriler.iloc[:,3:4].values

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
hava[:,0]=le.fit_transform(hava[:,0])
oyun[:,0]=le.fit_transform(oyun[:,0])
ruzgar[:,0]=le.fit_transform(ruzgar[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features='all')
hava=ohe.fit_transform(hava).toarray()
oyun=ohe.fit_transform(oyun).toarray()
ruzgar=ohe.fit_transform(ruzgar).toarray()

ruzgar_en=pd.DataFrame(data=ruzgar,index=range(14),columns=['ruzgar','ruzgar'])
hava_en=pd.DataFrame(data=hava,index=range(14),columns=['gu','bu','ya'])
oyun_en=pd.DataFrame(data=oyun,index=range(14),columns=['ev','oyun'])
son_oyun=oyun_en.iloc[:,1:]
son_ruzgar=ruzgar_en.iloc[:,1:]

x= Veriler.iloc[:,:4]
y=x.iloc[:,1:]
z=y.iloc[:,:-1]

genel1=pd.concat([son_ruzgar,z],axis=1)
genel2=pd.concat([son_oyun,genel1],axis=1)
en_genel=pd.concat([hava_en,genel2],axis=1)

giris=en_genel.iloc[:,:-1]
sonuc=en_genel.iloc[:,6:]

from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test=train_test_split(giris,sonuc,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

import statsmodels.formula.api as sm

w=np.append(arr=np.ones((14,1)).astype(int),values=en_genel,axis=1)
w_l=en_genel.iloc[:,[0,1,2,3,4,5]].values
r_ols=sm.OLS(endog =en_genel.iloc[:,-1:], exog =w_l)
r=r_ols.fit()
print(r.summary())

import statsmodels.formula.api as sm

w=np.append(arr=np.ones((14,1)).astype(int),values=en_genel,axis=1)
w_l=en_genel.iloc[:,[0,1,2,3,5]].values
r_ols=sm.OLS(endog =en_genel.iloc[:,-1:], exog =w_l)
r=r_ols.fit()
print(r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)


'''from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)

Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)'''