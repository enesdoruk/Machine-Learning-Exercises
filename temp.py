import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler= pd.read_csv('eksikveriler.csv')

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN',strategy= 'mean',axis=0)
yas = veriler.iloc[:,1:4].values

imputer =imputer.fit(yas[:,1:4])
yas[:,1:4]= imputer.transform(yas[:,1:4])

print(yas)