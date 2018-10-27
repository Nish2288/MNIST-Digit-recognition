# Import libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, MaxPool2D, Flatten



# Import train and test set
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Divide Dependent and independent variables
x_train = train.iloc[ : , 1:]
y_train = train.iloc[ : , 0 ]

# Check for null values
x_train.isnull().any().describe()
test.isnull().any().describe()

# Normalization
x_train=x_train/255.0
test=test/255.0

# Reshape image in 3D gray scale (h=28px, w=28px, c=1 )
x_train=x_train.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)

# One hot encoding of y variable
y_train = to_categorical(y_train)


# Display
#plt.imshow(x_train[0][:,:,0])

# Convolution architecture [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
# 97.56 % Accuracy on training set.

classifier=Sequential()

classifier.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(28,28,1)))
classifier.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Dropout(0.10))

classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
classifier.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2), strides=(2, 2)))
classifier.add(Dropout(0.20))

classifier.add(Flatten())
classifier.add(Dense(units=256, activation='relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(units=10, activation='softmax'))

# Compile
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit
classifier.fit(x_train, y_train, batch_size = 32, epochs = 3)

results=classifier.predict(test)
classifier.save('Digit_rec_model2.h5')

'''
classifier.save('Digit_rec_model.h5')

from keras.models import load_model
newmodel=load_model('Digit_rec_model.h5')
newmodel.summary() '''
''' To store result in csv file.
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False) '''




   
