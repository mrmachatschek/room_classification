
'''
As taken from https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/
and from "Deep Learning with Pyhton" from Francois Chollet
there are 3 common approaches for transfer learning
    1. Standalone Feature Extractor: The pre-trained model, or some portion of the model, is used to pre-process images and extract relevant features.
    2. Integrated  Feature Extractor: The pre-trained model, or some portion of the model, is integrated into a new model, but layers of the pre-trained model are frozen during training.
    3. Fine-Tuning: The pre-trained model, or some portion of the model, is integrated into a new model, and some layers of the pre-trained model are trained in concert with the new model.
'''

'''
In this package, the idea is to create 3 classes, one for each approach, that are named:
    1. ConvBaseSearchSFE (SFE stands for Standalone Feature Extractor)
    2. ConvBaseSearchIFE (IFE stands for Integrated  Feature Extractor)
    3. ConvBaseSearchFT (FT stands for Fine-Tuning)
'''

'''
First, just to introduce some formalism, we create an abstract class which will serve as parent for the 4 
transfer learning model classes (they will be child). 
Abstract means that this class cannot be instanciated directly.
Some of the methods in this class are also abstract (their code must be defined in the child class)
while some others are not abstract (their code is defined in the parent class so they do not have to be repeated in the child class)
'''


from abc import ABC, abstractmethod
from keras.models import Sequential
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

    def __init__(self, model_name, input_shape):
        self.name = model_name
        self.model = self.build_conv_base(model_name, input_shape)
        self.score = None
        self.history = None
        self.train_feature_map = None
        self.train_labels = None
        self.val_feature_map = None
        self.val_labels = None

    def build_conv_base(self, model_name, input_shape):
        """
        Function that intializes the pretrained model from keras.

        Args:
        ---
        model_name: String identifier for pretrained model from Keras. A value from ["vgg16", "vgg19"]
        input_shape: A tuple with the input shape for the pretrained model
        
        Returns:
        ---
        conv_base: An Keras model object of the corresponding pretrained model.
        """
        if model_name == "vgg19":
            conv_base = VGG19(weights="imagenet", include_top=False, input_shape=input_shape)
        if model_name == "vgg16":
            conv_base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

        conv_base.trainable = False

        return conv_base

    def extract_features(self, generator, validation_data=None):
        for gen, dataset in zip([generator, validation_data], ["train", "val"]):
            output_shape = self.model.layers[-1].output_shape
            features = np.zeros(shape=(gen.samples, output_shape[1], output_shape[2], output_shape[3]))
            labels = np.zeros(shape=(gen.samples, len(gen.class_indices)))
            batch_size = gen.batch_size
            
            i=0
            for inputs_batch, labels_batch in gen:
                features_batch = self.model.predict(inputs_batch)
                features[i * batch_size: (i + 1) * batch_size] = features_batch
                labels[i * batch_size: (i + 1) * batch_size] = labels_batch
                i += 1
                if i * batch_size >= gen.samples:
                    break
                
            #Flatten feature map
            features = np.reshape(features,(features.shape[0], features.shape[1] * features.shape[2] * features.shape[3]))
            
            #For evaluating model
            if validation_data == None:
                return features, labels
            
            if dataset == "train":
                self.train_feature_map = features
                self.train_labels = labels
            else: 
                self.val_feature_map = features
                self.val_labels = labels
        
        print("Successfully extraced features from ", self.name)


    def set_score(self, score):
        self.score = score
    
    def set_history(self, history):
        self.history = history


class ConvBaseSearch(ABC):

    def __init__(self, pretrained_models, input_shape):
        self.conv_bases = self.__input_pretrained_models(pretrained_models, input_shape)
        self.compile_params = None
        self.best_model = None
        self.models = self.__init_models()

    def __init_models(self):
        '''
        We will assume that when the instance of the class is initialized, the models to be trained are exactly the same
        as the pre trained models
        '''
        models = []
        for conv_base in self.conv_bases:
            base_model = conv_base.model
            models.append(base_model)
        return models

    def compile(self, optimizer, loss, metrics):
        '''
        Compiles all models and the models list
        '''
        #just storing as internal variable in case it's neeeded in the future
        self.compile_params = {'optimizer': optimizer, 'loss': loss, 'metrics': metrics}
        for model in self.models:
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self):
        #To do
        return None

    def fit_generator(self, generator, steps_per_epoch, epochs = 1, validation_data = None, validation_steps=None):
         '''
         Similarly to keras.model.fit_generator, receives a generator as argument and fits for every model
         '''
         for model, conv_base in zip(self.models, self.conv_bases):
            print("Fitting ", conv_base.name)
            #Fit model
            hist = model.fit_generator(generator=generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                validation_data = validation_data, validation_steps=validation_steps)
            #Saves history of model
            conv_base.set_history(hist)
            #Saves score of last epoch
            conv_base.set_score(hist.history['val_accuracy'][-1])
            print("Score on val set: ", conv_base.score, "\n")


    def evaluate_generator(self, test_generator):
        results = {}
        for model, conv_base in zip(self.models, self.conv_bases):
            results[conv_base.name] = model.evaluate_generator(test_generator)
            
        return results

    def predict(self):
        #To do
        return None

    def __input_pretrained_models(self, pretrained_models, input_shape):
        '''
        Processes variable pretrained_models which could be
        - one string
        - one pre trained model object
        - a list of strings
        - a list of pre trained model objects
        Returns a list of ConvBase objects
        '''
        if is_list_of_strings(pretrained_models):
            return [ConvBase(ptm, input_shape) for ptm in pretrained_models]
        #To do
        # elif is_list_of_models(pretrained_models):
        # else is_model(pretrained_models):
        # else is_string(pretrained_models):
    
    @abstractmethod
    def _adapt_models(self):
        '''
        Here is the method in which the layers of the pre trained model and top model are adapted
        As this is depending on the child class, it's an abstract method
        '''
        pass


class ConvBaseSearchIFE(ConvBaseSearch):

    def __init__(self, pretrained_models, n_classes, input_shape, top_model = None):
        # some more arguments should be passed here to instruct the number of layers from the pretrained model to be used
        self.top_model = self.__input_top_model(top_model, n_classes)
        super().__init__(pretrained_models, input_shape)
        self._adapt_models()

    def _adapt_models(self):
        '''
        Here is where we adapt the base model (which is the pretrained model) and connect it to the top_model
        '''
        top_model = self.top_model
        for i in range(len(self.models)):
            sequential = Sequential()
            sequential.add(self.models[i])
            sequential.add(layers.Flatten())
            sequential.add(top_model)
            self.models[i] = sequential

    def __input_top_model(self, top_model, n_classes):
        if top_model == None:
            # use custom top model
            return self.__build_custom_model(n_classes)
        else:
            # use the model input by the user
            return top_model

    def __build_custom_model(self, n_classes):
        """
        Function that initializes a densely connected network that will be trained on top of the convolutional base.

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
            activation = "sigmoid"
        else:
            loss = 'categorical_crossentropy'
            activation = "softmax"

        model = Sequential()
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_classes, activation=activation))
        return model


class ConvBaseSearchFT(ConvBaseSearch):

    def __init__(self, pretrained_models, n_classes, input_shape, n_trainable, top_model = None):
        self.top_model = self.__input_top_model(top_model, n_classes)
        super().__init__(pretrained_models, input_shape)
        self._adapt_models()
        self.n_trainable = n_trainable

    def _adapt_models(self):
        '''
        Here is where we adapt the base model (which is the pretrained model) and connect it to the top_model
        '''
        top_model = self.top_model
        for i in range(len(self.models)):
            #Plug pretrained model and top model together
            sequential = Sequential()
            sequential.add(self.models[i])
            sequential.add(layers.Flatten())
            sequential.add(top_model)
            self.models[i] = sequential
            
    def fit_generator(self, generator, steps_per_epoch, epochs = 1, validation_data = None, validation_steps=None):
        print("Initial training")
        super().fit_generator(generator, steps_per_epoch, epochs = 1, validation_data = validation_data, validation_steps=validation_steps)
        print("Fine tuning of last", self.n_trainable, "layers")
        self.__set_trainable_layers(self.n_trainable)
        super().fit_generator(generator, steps_per_epoch, epochs = 1, validation_data = validation_data, validation_steps=validation_steps)
    
    def __set_trainable_layers(self, n_trainable):
            for i in range(len(self.models)):
                #Last n_trainable layers are set to trainable
                for layer in self.models[i].layers[0].layers[-n_trainable:]:
                    layer.trainable = True
                        
    def __input_top_model(self, top_model, n_classes):
        if top_model == None:
            # use custom top model
            return self.__build_custom_model(n_classes)
        else:
            # use the model input by the user
            return top_model

    def __build_custom_model(self, n_classes):
        """
        Function that initializes a densely connected network that will be trained on top of the convolutional base.

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
            activation = "sigmoid"
        else:
            activation = "softmax"

        model = Sequential()
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_classes, activation=activation))
        return model


class ConvBaseSearchSFE(ConvBaseSearch):

    def __init__(self, pretrained_models, n_classes, input_shape, top_model = None):
        # some more arguments should be passed here to instruct the number of layers from the pretrained model to be used
        self.top_model = self.__input_top_model(top_model, n_classes)
        super().__init__(pretrained_models, input_shape)
        self._adapt_models()
        

    def _adapt_models(self):
        '''
        Here is where we adapt the base model (which is the pretrained model) and connect it to the top_model
        '''
        top_model = self.top_model
        for i in range(len(self.models)):
            self.models[i] = top_model

    def fit_generator(self, generator, steps_per_epoch, epochs = 1, batch_size=32, validation_data = None, validation_steps=None):
        print("Extracting Features...")
        self.__extract_features(generator, validation_data)
        print("\nFit top model with feature maps")
        self.__fit_top_model(epochs, batch_size)
    
    def evaluate_generator(self, test_generator):
        results = {}
        for model, conv_base in zip(self.models, self.conv_bases):
            test_set, test_labels = conv_base.extract_features(test_generator)
            results[conv_base.name] = model.evaluate(test_set, test_labels)
            
        return results
        
    
    def __extract_features(self, generator, validation_data):
        for conv_base in self.conv_bases:
            conv_base.extract_features(generator, validation_data)
    
    def __fit_top_model(self, epochs, batch_size):
        for model, conv_base in zip(self.models, self.conv_bases):
            print("Fitting ", conv_base.name)
            #Get input data and labels
            train_set = conv_base.train_feature_map
            train_labels = conv_base.train_labels
            val_set = conv_base.val_feature_map
            val_labels = conv_base.val_labels
            #Fit model
            hist = model.fit(train_set, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(val_set, val_labels))
            #Saves history of model
            conv_base.set_history(hist)
            #Saves score of last epoch
            conv_base.set_score(hist.history['val_accuracy'][-1])
            print("Score on val set: ", conv_base.score)
            
        
    
    def __input_top_model(self, top_model, n_classes):
        if top_model == None:
            # use custom top model
            return self.__build_custom_model(n_classes)
        else:
            # use the model input by the user
            return top_model

    def __build_custom_model(self, n_classes):
        """
        Function that initializes a densely connected network that will be trained on top of the convolutional base.

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
            activation = "sigmoid"
        else:
            loss = 'categorical_crossentropy'
            activation = "softmax"

        model = Sequential()
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(n_classes, activation=activation))
        return model


# Utils
def is_list_of_strings(lst):
    if lst and isinstance(lst, list):
        return all(isinstance(elem, str) for elem in lst)
    else:
        return False
