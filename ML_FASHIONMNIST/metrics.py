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


from keras.models import Sequential 
from keras.layers import Conv2D,Flatten,Dropout,Dense,MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


model = Sequential()
model.add(Conv2D(32,3,3,input_shape=(28,28,1),activation ='relu'))
model.add(Flatten())
model.add(Dense(output_dim=32,activation='relu'))
model.add(Dense(output_dim=10,activation = 'sigmoid'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.001),metrics=['accuracy'])
epochs = 50
model.fit(x_train,y_train,batch_size=512,nb_epoch=epochs,verbose=1,
          validation_data=(x_validate,y_validate))



'''

EVALUATİNG_MODEL

'''


evaluation = model.evaluate(x_test,y_test)

print('best_accuracy: {.3f}'.format(evaluation[1]))

predicted_classes = model.predict_classes(x_test)


'''

METRİCS


'''

from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test,predicted_classes)

plt.figure(figsize =(14,10))

sns.heatmap(cm,annot=True)

from sklearn.metrics import classification_report

num_classes = 10
target_names = ['class {}'.format(i) for i in range(num_classes)]

print(classification_report(y_test,predicted_classes,target_names=target_names))


















