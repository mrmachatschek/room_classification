
'''
As taken from
https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/
there are 4 approaches for transfer learning
    1. Classifier: The pre-trained model is used directly to classify new images.
    2. Standalone Feature Extractor: The pre-trained model, or some portion of the model, is used to pre-process images and extract relevant features.
    3. Integrated  Feature Extractor: The pre-trained model, or some portion of the model, is integrated into a new model, but layers of the pre-trained model are frozen during training.
    4. Weight Initialization: The pre-trained model, or some portion of the model, is integrated into a new model, and the layers of the pre-trained model are trained in concert with the new model.
'''

'''
The idea is to create 4 classes (maybe for this project just one or two), one for each approach, that could be named
    1. ConvBaseSearchClassifier
    2. ConvBaseSearchSFE (SFE stands for Standalone Feature Extractor)
    3. ConvBaseSearchIFE (IFE stands for Integrated  Feature Extractor)
    4. ConvBaseSearchWI (WI stands for Weight Initialization)
'''

'''
First, just to introduce some formalism, we create an abstract class which will serve as parent for the 4 
transfer learning model classes (they will be child). 
Abstract means that this class cannot be instanciated directly.
Some of the methods in this class are also abstract (their code must be defined in the child class)
while some others are not abstract (their code is defined in the parent class so they do not have to be repeated in the child class)
'''

'''
What is common between the child classes that can be implemented in the parent class?
    - a group of pre trained models must be defined for further study
    - each model is defined by a sequence of layers which have the same structure whithin the same child class
    - the pipeline: build layers + compile + fit 
    - 
'''

'''
What is different between child classes that needs to be implemented in the child classes?
    - how the mode is created
        1. ConvBaseSearchClassifier: the sequence is the same of the pretrained model
        2. ConvBaseSearchSFE: 
        3. ConvBaseSearchIFE: the sequence is the same (or part) of the pretrained model plus a new sequence
        4. ConvBaseSearchWI: the sequence is the same (or part) of the pretrained model plus a new sequence
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

    def __init__(self, model_name):
        self.name = model_name
        self.model = self.build_conv_base(model_name)
        self.feature_map = None
        self.labels = None
        self.score = None

    def build_conv_base(self, model_name, trainable=False):
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
            conv_base = VGG19(weights="imagenet", include_top=True, input_shape=(150, 150, 3))
        if model_name == "vgg16":
            conv_base = VGG16(weights="imagenet", include_top=True, input_shape=(150, 150, 3))

        conv_base.trainable = trainable

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
            df, rest = train_test_split(df, test_size=1 - sample_size, random_state=1, stratify=df[target_col].values)
        batch_size = 50
        datagen = ImageDataGenerator(rescale=1. / 255)

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
        i = 0
        for inputs_batch, labels_batch in generator:
            features_batch = self.model.predict(inputs_batch)
            features[i * batch_size: (i + 1) * batch_size] = features_batch
            labels[i * batch_size: (i + 1) * batch_size] = labels_batch
            i += 1
            if i * batch_size >= df.shape[0]:
                break
        if flatten:
            features = np.reshape(features,
                                  (features.shape[0], features.shape[1] * features.shape[2] * features.shape[3]))

        print("Successfully extraced features from ", self.name)
        self.feature_map = features
        self.labels = labels

    def set_score(self, score):
        self.score = score



class ConvBaseSearch(ABC):

    def __init__(self, pretrained_models):
        self.conv_bases = self.__input_pretrained_models(pretrained_models)
        self.compile_params = None
        self.best_model = None
        self.models = self.__init_models()
        self.__adapt_models()

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

    @abstractmethod
    def __adapt_models(self):
        '''
        Here is the method in which the layers of the pre model and top model are adapted
        As this is depending on the child class, it's an abstract method
        '''
        pass

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
        pass

    def fit_generator(self, generator, steps_per_epoch, epochs = 1, validation_data = None):
         '''
         Similarly to keras.model.fit_generator, receives a generator as argument and fits for every model
         '''
         for model, conv_base in zip(self.models, self.conv_bases):
            print("Fitting ", conv_base.name)
            #Fit model
            hist = model.fit_generator(generator=generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
                                validation_data = validation_data)
            #Saves score of last epoch
            conv_base.set_score(hist.history['val_accuracy'][-1])
            print("Score on test set: ", conv_base.score)


    def evaluate(self):
        #To do
        pass

    def predict(self):
        #To do
        pass

    def __input_pretrained_models(self, pretrained_models):
        '''
        Processes variable pretrained_models which could be
        - one string
        - one pre trained model object
        - a list of strings
        - a list of pre trained model objects
        Returns a list of ConvBase objects
        '''
        if is_list_of_strings(pretrained_models):
            return [ConvBase(ptm) for ptm in pretrained_models]
        #To do
        # elif is_list_of_models(pretrained_models):
        # else is_model(pretrained_models):
        # else is_string(pretrained_models):


class ConvBaseSearchIFE(ConvBaseSearch):

    def __init__(self, pretrained_models, n_classes, top_model = None, n_trainable = 0):
        # some more arguments should be passed here to instruct the number of layers from the pretrained model to be used
        self.top_model = self.__input_top_model(top_model, n_classes)
        self.n_trainable = n_trainable
        super().__init__(pretrained_models)
        self.__adapt_models(n_trainable)

    def __adapt_models(self):
        '''
        Here is where we adapt the base model (which is the pretrained model) and connect it to the top_model
        '''
        top_model = self.top_model
        for model, conv_base in zip(self.models, self.conv_bases):
            # at this point, model is just the base pre trained model
            # check the number of layers of base model
            base_num_layers = len(model.layers)
            # freezes all layers of the pre trained model (except the last n_trainable)
            for i_layer, layer in enumerate(model.layers):
                if base_num_layers - i_layer > self.n_trainable:
                    layer.trainable = False
            # need to remove the last layer of the pre trained model
            model.pop()
            # flatten the last layer
            model.add(layers.Flatten())
            # add the top model
            model.add(top_model)

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
