import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

veriler=pd.read_csv('musteriler.csv')

X=veriler.iloc[:,3:]

from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')

y_tahmin=ac.fit_predict(X)
print(y_tahmin)

plt.scatter(X[y_tahmin==0,0],X[y_tahmin==0,1],s=100,c='red')
plt.scatter(X[y_tahmin==1,0],X[y_tahmin==1,1],s=100,c='yellow')
plt.scatter(X[y_tahmin==2,0],X[y_tahmin==2,1],s=100,c='green')
plt.scatter(X[y_tahmin==3,0],X[y_tahmin==3,1],s=100,c='black')
plt.show()

import scipy.cluster.hierarchy as sch

dendogram=sch.dendrogram(sch.linkage(X,method='ward'))