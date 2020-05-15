
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

trained_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')

print(trained_df.head())
print(test_df.head())
print('--------------------------')
print(trained_df.tail())
print(test_df.tail())
print('---------------------------')
print(trained_df.shape)
print(test_df.shape)

