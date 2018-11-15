#kütüphaneler
#tek parametreli yani p value yüksek olanları attık
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm

#veri okuma
veriler=pd.read_csv('maaslar_yeni.csv')

#gereksizleri atma
numeric2=veriler.iloc[:,2:6]

#train,test bolümü
y1=numeric2.iloc[:,3:4]
x1=numeric2.iloc[:,:1]
x = numeric2.iloc[:,:1].values
y=numeric2.iloc[:,3:4].values


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#r-squared for linear
print("linear r2")
print(r2_score(y,lin_reg.predict(x)))

#p-value linreg
model=sm.OLS(lin_reg.predict(x),x)
print(model.fit().summary())

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
#poly icindeki lin reg.
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
#p values polyreg
model2=sm.OLS(lin_reg2.predict(poly_reg.fit_transform(x)),x)
print(model2.fit().summary())

#support vector reg
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(x)
sc2=StandardScaler()
y_olcekli=sc2.fit_transform(y)

#svr kernel
from sklearn.svm import SVR
svr_reg=SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)
#svr ols
print("svr ols")
model3=sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model3.fit().summary())
#svr r2 değeri
print("svr r2 değeri")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))



#decision tree reg
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(x,y)

#decision ols
model4=sm.OLS(r_dt.predict(x),x)
print(model4.fit().summary())
#decision r2 değeri
print(r2_score(y,r_dt.predict(x)))

#randomforest regression
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.predict(x,y)

#rf ols
print("rf ols")
model5=sm.OLS(rf_reg.predict(x),x)
print(model5.fit().summary())




