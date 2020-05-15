import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

trained_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')


x_train = training[:,1:]/255
y_train = training[:,0]

x_test = testing[:,1:]/255
y_test = testing[:,0]

from sklearn.model_selection import train_test_split

x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,
                                                         test_size= 0.2,
                                                         random_state=1234)


x_train = x_train.reshape(x_train.shape[0],*(28,28,1))
x_test = x_test.reshape(x_test.shape[0],*(28,28,1))
x_validate = x_validate.reshape(x_validate.shape[0],*(28,28,1))


