from transferlearning2 import ConvBaseSearchIFE

##################################################################################################
# First we create the dataframe (as previously)
##################################################################################################

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
numOfSamplesCat = 400

df = pd.DataFrame(columns=df_total.columns)
from sklearn.utils import shuffle
# Get only n pics of each class
for cl in classes:
    df_class = shuffle(df_total[df_total['Cat'] == cl]).iloc[:numOfSamplesCat, :]
    df = df.append(df_class)


##################################################################################################
# Creating the ImageDataGenerator objects
##################################################################################################
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.2)

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

##################################################################################################
# Create an instance of the searcher
##################################################################################################

# give as arguments a list of the pretrained models names (in the future it could a list of model objects) and num of classes
# as it is initialized, gets the pre trained models and then adapts (or plugs) them to a custom model (that could also be an argument of this function)
classifier = ConvBaseSearchIFE(['vgg16','vgg19'], len(classes))

##################################################################################################
# Compiling the searcher
##################################################################################################
# internally it's compiling all the models with these params
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

##################################################################################################
# fit the models within the searcher!
##################################################################################################

hist=classifier.fit_generator(training_set,
                         #steps_per_epoch = 4000,
                         epochs = 100,
                         validation_data = test_set,
                         #validation_steps = 1000)
                              )

##################################################################################################
# predict.. evaluate.. (not implemented)
##################################################################################################
#classifier.predict(...)
#classifier.evaluate(...)