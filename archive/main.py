
# Part 0 - File structure into df

import os
import pandas as pd

#https://stackoverflow.com/questions/42654961/creating-pandas-dataframe-from-os
res = []
path = 'E:\\011_Fotos\\'
#path = 'E:\\Dados\\FLH HOLIDAY RENTALS\\011_Fotos\\'
for root, dirs, files in os.walk(path, topdown=True):
    if len(files) > 0:
        res.extend(list(zip([root]*len(files), files)))

df = pd.DataFrame(res, columns=['Path', 'File_Name'])


df = df[df['File_Name'] != 'Thumbs.db']
df['ClientId'] = df.Path.apply(lambda x: int(x.split("\\")[-1]))
df = df[df['ClientId'] < 10000]

df['Full_Path'] = df["Path"] + '\\' + df["File_Name"]
df['Cat'] = df.File_Name.apply(lambda x: x.split(".")[0].split("_")[-1])

classes = ['1','3','4']
df = df[df.Cat.isin(classes)]
df_total = df
df = pd.DataFrame(columns=df_total.columns)

numOfSamplesCat = 300
from sklearn.utils import shuffle
# Get only 100 pics of each class
for cl in classes:
    df_class = shuffle(df_total[df_total['Cat'] == cl]).iloc[:numOfSamplesCat, :]
    df = df.append(df_class)


from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.2)


# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_dataframe(dataframe=df_train, directory = None, x_col='Full_Path', y_col='Cat',
                                                 target_size = (64, 64),
                                                 class_mode = 'categorical')

test_set = train_datagen.flow_from_dataframe(dataframe=df_test, directory = None, x_col='Full_Path', y_col='Cat',
                                                 target_size = (64, 64),
                                                 class_mode = 'categorical')

# training_set = train_datagen.flow_from_directory('Lab3/training_set',
#                                                  target_size = (64, 64),
#                                                  batch_size = 32,
#                                                  class_mode = 'binary')
#
# test_set = test_datagen.flow_from_directory('Lab3/test_set',
#                                             target_size = (64, 64),
#                                             batch_size = 32,
#                                             class_mode = 'binary')
#
training_set.class_indices

from datetime import datetime

now = datetime.now()
print("now =", now)

numEpochs = 100
hist=classifier.fit_generator(training_set,
                         #steps_per_epoch = 4000,
                         epochs = numEpochs,
                         validation_data = test_set,
                         #validation_steps = 1000)
                              )

now = datetime.now()
print("now =", now)

filenamebase= '_'.join([str(numOfSamplesCat), str(numEpochs)])
filename = filenamebase + '_train.txt'
with open(filename, 'w') as filehandle:
    for listitem in hist.history['accuracy']:
        filehandle.write('%s\n' % listitem)

filename = filenamebase + '_val.txt'
with open(filename, 'w') as filehandle:
    for listitem in hist.history['val_accuracy']:
        filehandle.write('%s\n' % listitem)

#from sklearn.metrics import classification_report, confusion_matrix
# Y_pred = classifier.predict_generator(test_set, 60) # num_of_test_samples // batch_size+1
# #Y_pred = (Y_pred>0.5)S
# print('Confusion Matrix')
# print(confusion_matrix(test_set.classes, Y_pred))
# print('Classification Report')
# target_names = ['Kitchen', 'Bedroom', 'Bathroom']
# print(classification_report(test_set.classes, Y_pred, target_names=target_names))
