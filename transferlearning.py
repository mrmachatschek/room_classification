import os
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.applications import VGG19
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import layers
from keras import optimizers
import numpy as np


class ConvBase():
    
    def __init__(self, model_name):
        self.name = model_name
        self.model = self.build_conv_base(model_name)
        self.feature_map = None
        self.labels = None
        self.score = None
            
            
    def build_conv_base(self, model_name):
        """
        Function that intializes the pretrained model from keras.
        
        Args:
        ---
        model_name: String identifier for pretrained model from Keras. A value from ["vgg16", "vgg19"]
        
        Returns:
        ---
        conv_base: An Keras model object of the corresponding pretrained model.
        """
        if model_name == "vgg19":
            conv_base = VGG19(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
        if model_name == "vgg16":
            conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
            
        return conv_base 
    
    
    def extract_features_from_dataframe(self, df, path_col, target_col, sample_size, flatten=True):
         """
        Function that extracts the feature map from the convolutional base stored in 'model'. 
        It runs each image through the convolutional  base and flattens the output of the last layer.
        The feature map and the labels will be stored in the corresponding attributes of the ConvBase object.
        
        Args:
        ---
        df: Pandas DataFrame object with a column that identifies the path of on image and a column that identifies the class of that image.
        path_col: String of column name that identifies the full path to the image.
        target_col: String of column name that identifies the target class of the image. 
        sample_size: Float between 0 and 1, that represents the proportion of data from with the feature should be extracted.
        flatten: True is array output of last layer should be flattended (default=True).
        """
        if sample_size != 1:
            df, rest = train_test_split(df, test_size=1-sample_size, random_state=1, stratify=df[target_col].values)
        batch_size = 50
        datagen = ImageDataGenerator(rescale = 1./255)
        
        output_shape = self.model.layers[-1].output_shape
        features = np.zeros(shape=(df.shape[0], output_shape[1], output_shape[2], output_shape[3]))
        labels = np.zeros(shape=(df.shape[0], len(df[target_col].unique())))
        
        generator = datagen.flow_from_dataframe(dataframe=df, 
                                                directory=None, 
                                                x_col=path_col,
                                                y_col=target_col,
                                                target_size=(150, 150),
                                                batch_size=batch_size,
                                                class_mode='categorical')
        i=0
        for inputs_batch, labels_batch in generator: 
            features_batch = self.model.predict(inputs_batch)
            features[i * batch_size : (i + 1) * batch_size] = features_batch
            labels[i * batch_size : (i + 1) * batch_size] = labels_batch
            i += 1
            if i * batch_size >= df.shape[0]:
                break
        if flatten:
            features = np.reshape(features, (features.shape[0], features.shape[1] * features.shape[2] * features.shape[3]))
        
        print("Successfully extraced features from ", self.name)
        self.feature_map = features 
        self.labels = labels
    
    
    def set_score(self, score):
        self.score = score
      
        
        
class ConvBaseSearch():
    
    def __init__(self, pretrained_models):
        self.conv_bases = [ConvBase(ptm) for ptm in pretrained_models]
        self.best_score = None
        self.best_base = None
        
        
    def fit(self, df, path_col, target_col, sample_size, data_augmentation=False):
        """
        Function that fits the data to all models that were passed to the constructor. 
        Updates the attributes best_score and best_base.
        
        Args:
        ---
        df: Pandas DataFrame object with a column that identifies the path of on image and a column that identifies the class of that image.
        path_col: String of column name that identifies the full path to the image.
        target_col: String of column name that identifies the target class of the image. 
        sample_size: Float between 0 and 1, that represents the proportion of data from with the feature should be extracted.
        data_augmentation: Should be set to True if data augmentation techniques should be applied (default=False).
        """
        if ~data_augmentation:
            for conv_base in self.conv_bases:
                print("Fitting ", conv_base.name)
                conv_base.extract_features_from_dataframe(df, path_col, target_col, sample_size)
                feature_map = conv_base.feature_map
                labels = conv_base.labels

                top_model = self.build_custom_model(feature_map.shape[1], labels.shape[1])
                
                train_input, test_input, train_labels, test_labels = train_test_split(feature_map, labels, test_size=0.2, random_state=1, stratify=labels)
                top_model.fit(train_input, train_labels)
                scores = top_model.evaluate(test_input, test_labels)
                conv_base.set_score(scores[1])
                print("Score: ", scores[1])
            
            best_score = 0
            for conv_base in self.conv_bases:
                score = conv_base.score
                if score > best_score:
                    best_score = score
                    self.best_base = conv_base
            self.best_score = best_score
    
    
    def build_custom_model(self, input_dim, n_classes):
        """
        Function that intializes a densely connected network that will be trained on top of the convolutional base.
        
        Args:
        ---
        input_dim: Integer of input dimension for first layer.
        n_classes: Integer of number of classes that the model should predict.
        
        Returns:
        ---
        model: An Keras Sequential object. 
        """
        if n_classes == 2:
            n_classes = 1
            loss = "binary_crossentropy"
        else:
            loss = 'categorical_crossentropy'
            
        model = Sequential()
        model.add(layers.Dense(256, activation="relu", input_dim= input_dim ))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_classes, activation="softmax"))

        model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss = loss, metrics = ['accuracy'])
        
        return model
                
                
                
                
                
            
        
        
  