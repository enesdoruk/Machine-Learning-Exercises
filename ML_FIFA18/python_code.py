
"""
Created on Sun Nov 25 17:03:10 2018

@author: enes doruk
"""
#%%
"""
İMPORT LİBRARY
"""
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
#%%
"""
İGNORE WARNİNGS ON CONSOLE"""
import warnings
warnings.filterwarnings("ignore")
#%%
"""
İMPORT DATASET AND İNFO
"""
dataset = pd.read_csv('FIFA 2018 Statistics.csv')

info = dataset.info()
#%%
"""
EDİTTİNG FEATURES 
"""
dataset.columns = [x.replace(' ','_') for x in dataset.columns]
dataset.columns = [x.lower() for x in dataset.columns]

dataset.drop(['date', 'team','opponent','round','pso','goals_in_pso']
              , axis=1,inplace = True)



x1 = dataset.iloc[:,0:16]
x2 = dataset.iloc[:,18:21]    

x= pd.concat([x1, x2],axis=1)
x = x.values

y= dataset['man_of_the_match']
#%%
"""
LABEL ENCODER SRT--->NUMBER
"""
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y =lb.fit_transform(y)

#%%
"""
USİNG İMPUTER FOR MİSSİNG VALUES('NaN') 

"""

from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(x)
x = imputer.transform(x)
#%%
"""
STANDARD SCALER
"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#%%

"""
USİNG RSQUARE AND ADJSQUARE, TO DELETE UNİMPORTANT FEATURES  
"""
import statsmodels.formula.api as sm


x_opt = x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
regressor_ols= sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()


def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((128,18)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
x_opt = x[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
x_Modeled = backwardElimination(x_opt, SL)     


x_yeni = x_Modeled

#%%
"""
SİMPLE PLOT AND HEATMAP
"""

plt.plot(x_yeni,y)
plt.title('plot map')
plt.xlabel('independent values')
plt.ylabel('dependent values')
plt.grid()
plt.show()


sns.set()
ax=sns.heatmap(x_yeni)

#%%

"""
SİMPLE LİNEAR REGRESSİON

"""
from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x_yeni,y,random_state=0,
                                                 test_size=0.3)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit( x_train, y_train)

y_pred = lr.predict(x_test)

for i in range(0,len(y_pred)):
    if y_pred[i] < 0.5:
        y_pred[i] = 0
    else:
        y_pred[i] = 1        


true = []
        
for j in range(0,38):
    if y_test[j] == y_pred[j]:
        true.append(j)


print('DOĞRU SAYILAR',true)         
accuracy = (len(true)/len(y_test))*100
print('ACCURACY: {} '.format(accuracy))
      

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = lr, X = x_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()


#%%
"""
polynomial features
"""
from sklearn.cross_validation import train_test_split

x_train2,x_test2,y_train2,y_test2 = train_test_split(x_yeni,y,random_state=0,
                                                 test_size=0.3)

from sklearn.preprocessing import PolynomialFeatures

pr = PolynomialFeatures(degree= 4)
x_poly = pr.fit_transform(x_yeni)
pr.fit(x_train2,y_train2)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

y_pred2 = lin_reg_2.predict(pr.fit_transform(x_test2))

for i in range(0,len(y_pred2)):
    if y_pred2[i] < 0.5:
        y_pred2[i] = 0
    else:
        y_pred2[i] = 1  

true2 = []
        
for j in range(0,38):
    if y_test[j] == y_pred[j]:
        true2.append(j)


print('DOĞRU SAYILAR',true2)         
accuracy2 = (len(true2)/len(y_test))*100
print('ACCURACY: {} '.format(accuracy2))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies2 = cross_val_score(estimator = lin_reg_2, X = x_train2, y = y_train2, cv = 10)
accuracies2.mean()
accuracies2.std()


#%%

"""
DECİSİON TREE REGRESSOR
"""
from sklearn.cross_validation import train_test_split

x_train3,x_test3,y_train3,y_test3 = train_test_split(x_yeni,y,random_state=0,
                                                 test_size=0.3)

from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(random_state = 0)
dtr.fit(x_train3,y_train3)

y_pred3 = dtr.predict(x_test3)

for i in range(0,len(y_pred3)):
    if y_pred3[i] < 0.5:
        y_pred3[i] = 0
    else:
        y_pred3[i] = 1  

true3 = []
        
for j in range(0,38):
    if y_test[j] == y_pred3[j]:
        true3.append(j)


print('DOĞRU SAYILAR',true3)         
accuracy3 = (len(true3)/len(y_test))*100
print('ACCURACY: {} '.format(accuracy3))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test3, y_pred3)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies3 = cross_val_score(estimator = dtr, X = x_train3, y = y_train3, cv = 10)
accuracies3.mean()
accuracies3.std()

## Visualising the Decision Tree Regression results (higher resolution)
#X_grid = np.arange(min(x), max(x), 0.01)
#X_grid = X_grid.reshape((len(X_grid), 1))
#plt.scatter(x, y, color = 'red')
#plt.plot(X_grid, dtr.predict(X_grid), color = 'blue')
#plt.title('Truth or Bluff (Decision Tree Regression)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()

#%%
"""
RANDOM FOREST REGRESSOR
"""

from sklearn.cross_validation import train_test_split

x_train4,x_test4,y_train4,y_test4 = train_test_split(x_yeni,y,random_state=0,
                                                 test_size=0.3)

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=10,random_state = 0)
rfr.fit(x_train4,y_train4)

y_pred4 = rfr.predict(x_test4)



for i in range(0,len(y_pred4)):
    if y_pred4[i] < 0.5:
        y_pred4[i] = 0
    else:
        y_pred4[i] = 1  

true4 = []
        
for j in range(0,38):
    if y_test4[j] == y_pred4[j]:
        true4.append(j)


print('DOĞRU SAYILAR',true4)         
accuracy4 = (len(true4)/len(y_test4))*100
print('ACCURACY: {} '.format(accuracy4))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test4, y_pred4)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies4 = cross_val_score(estimator = rfr, X = x_train4, y = y_train4, cv = 10)
accuracies4.mean()
accuracies4.std()


#X_grid = np.arange(min(x_train), max(x_train), 0.01)
#X_grid = X_grid.reshape((len(X_grid), 1))
#plt.scatter(x_test, y, color = 'red')
#plt.plot(X_grid, rfr.predict(X_grid), color = 'blue')
#plt.title('Truth or Bluff (Random Forest Regression)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()

#%%
"""
CROSS VALİDATİON

"""
from sklearn.cross_validation import train_test_split

x_train5,x_test5,y_train5,y_test5 = train_test_split(x_yeni,y,random_state=0,
                                                 test_size=0.3)


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train5, y_train5)

y_pred5 = regressor.predict(x_test5)


for i in range(0,len(y_pred5)):
    if y_pred5[i] < 0.5:
        y_pred5[i] = 0
    else:
        y_pred5[i] = 1  

true5 = []
        
for j in range(0,38):
    if y_test5[j] == y_pred5[j]:
        true5.append(j)


print('DOĞRU SAYILAR',true5)         
accuracy5 = (len(true5)/len(y_test5))*100
print('ACCURACY: {} '.format(accuracy5))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm5 = confusion_matrix(y_test5, y_pred5)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies5 = cross_val_score(estimator = regressor, X = x_train5, y = y_train5, cv = 10)
accuracies5.mean()
accuracies5.std()

#grid search
#from sklearn.model_selection import GridSearchCV
#parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 
#               'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
#grid_search = GridSearchCV(estimator = regressor,
#                           param_grid = parameters,
#                           scoring = 'accuracy',
#                           cv = 10,
#                           n_jobs = -1)
#grid_search = grid_search.fit(x_train5, y_train5)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_

## Visualising the SVR results
#plt.scatter(X, y, color = 'red')
#plt.plot(X, regressor.predict(X), color = 'blue')
#plt.title('Truth or Bluff (SVR)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()
#
## Visualising the SVR results (for higher resolution and smoother curve)
#X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
#X_grid = X_grid.reshape((len(X_grid), 1))
#plt.scatter(X, y, color = 'red')
#plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
#plt.title('Truth or Bluff (SVR)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()

#%%

"""
KERNEL PCA
"""

from sklearn.model_selection import train_test_split
x_train6, x_test6, y_train6, y_test6 = train_test_split(x_yeni, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train6 = sc.fit_transform(x_train6)
x_test6 = sc.transform(x_test6)

# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
x_train6 = kpca.fit_transform(x_train6)
x_test6 = kpca.transform(x_test6)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train6, y_train6)

# Predicting the Test set results
y_pred6 = classifier.predict(x_test6)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm6 = confusion_matrix(y_test6, y_pred6)


#%%
"""
LDA

"""

from sklearn.cross_validation import train_test_split

x_train7,x_test7,y_train7,y_test7 = train_test_split(x_yeni,y,random_state=0,
                                                 test_size=0.3)

from sklearn.tree import DecisionTreeRegressor

dtr2 = DecisionTreeRegressor(random_state = 0)
dtr2.fit(x_train7,y_train7)

y_pred7 = dtr2.predict(x_test7)

for i in range(0,len(y_pred7)):
    if y_pred7[i] < 0.5:
        y_pred7[i] = 0
    else:
        y_pred7[i] = 1  

true7 = []
        
for j in range(0,38):
    if y_test7[j] == y_pred7[j]:
        true7.append(j)


print('DOĞRU SAYILAR',true7)         
accuracy7 = (len(true7)/len(y_test7))*100
print('ACCURACY: {} '.format(accuracy7))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm7 = confusion_matrix(y_test7, y_pred7)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies7 = cross_val_score(estimator = dtr2, X = x_train7, y = y_train7, cv = 10)
accuracies7.mean()
accuracies7.std()



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
x_train7 = lda.fit_transform(x_train7, y_train7)
x_test7 = lda.transform(x_test7)

#%%

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train8, X_test8, y_train8, y_test8 = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier2 = XGBClassifier()
classifier2.fit(X_train8, y_train8)

# Predicting the Test set results
y_pred8 = classifier2.predict(X_test8)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm8 = confusion_matrix(y_test8, y_pred8)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies8 = cross_val_score(estimator = classifier2, X = X_train8, y = y_train8, cv = 10)
accuracies8.mean()
accuracies8.std()



#%%






