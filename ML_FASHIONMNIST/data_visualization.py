import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

trained_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')

training = np.array(trained_df,dtype='float32')
testing = np.array(test_df,dtype='float32')

plt.imshow(training[10,1:].reshape(28,28))

import random 
i = random.randint(1,60000)
plt.imshow(training[i,1:].reshape(28,28))
label = training[i,0]

w_grid = 15
i_grid = 15

fig,axes = plt.subplots(i_grid,w_grid,figsize= (17,17))
axes = axes.ravel()

n_training = len(training)

for i in range(0,w_grid*i_grid):
    index = np.random.randint(0,n_training)
    axes[i].imshow(training[index,1:].reshape(28,28))
    axes[i].set_title(training[index,0])
    axes[i].axis('off')
    
plt.subplots_adjust(hspace=0.4)    
    