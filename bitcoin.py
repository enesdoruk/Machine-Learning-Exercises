import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('bitcoin.csv')

#data.plot(kind='scatter',x='Open',y='Close',color='red',alpha=0.3,grid=True)
#plt.show()

#plt.scatter(data.Open,data.Close,hold=True,alpha=0.5,linewidths=0.5)
#plt.show()

#plt.hist(data.Open,bins=200,log=False,stacked=False,orientation='horizontal',histtype='bar',color='red')
#import seaborn as sns
#f,ax=plt.subplots(figsize=(15,15))
#sns.heatmap(data.corr(),annot=True,linewidth=.5,fmt='.1f',ax=ax)
#
#
#num1 = [5,10,20]
#num2 = [i**2 if i==10 else i-5 if i<7 else i+5 for i in num1]
#print(num2)

plt.boxplot(data)
plt.show()
