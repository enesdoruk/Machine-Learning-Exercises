import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler= pd.read_csv('veriler.csv')

ulke = veriler.iloc[:,0:1].values

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
ulke[:,0]=le.fit_transform(ulke[:,0])
print(ulke)
