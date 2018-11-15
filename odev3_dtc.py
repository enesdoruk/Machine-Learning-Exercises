import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix

veriler = pd.read_excel('Iris.xls')

x=veriler.iloc[:,:4].values
y=veriler.iloc[:,4:].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion='entropy')

dtc.fit(x_train,y_train)
#y_proba=rfc.predict_proba(x_test)
y_pred5=dtc.predict(x_test)


cm=confusion_matrix(y_test,y_pred5)
print('______________y_pred____________\n\n')
print(y_pred5)
print('\n\n')
print('___________ytest__________\n\n')
print(y_test)
print('\n\n')
print('_____________confmatrix_____\n\n')
print(cm)
print('\n\n')
#print('_____________proba_____\n\n')
#print(y_proba)