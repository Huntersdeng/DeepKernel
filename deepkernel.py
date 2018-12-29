import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Reshape, Dense, Embedding, Lambda
import numpy as np
from functools import reduce


def svm_accuracy(y_true, y_pre):
    return tf.reduce_mean(tf.cast(tf.equal(tf.sign(y_pre),y_true), tf.float32))
    
def svm_loss(y_true, y_pre):
    '''
    Hinge loss function for svm
    compute by max(0, 1-y_true*y_pre)
    '''
    return tf.reduce_mean(tf.maximum(0., 1-tf.multiply(y_pre,y_pre)))

    
class Deepkernel():
    def __init__(self, input_shape, solver='softmax'):
        '''
        A class for training a deep kernel function for get the features of the input data;
        Using tensorflow.keras backend;
        Argument:
            input_shape: the input shape of the data; this argument add a reshape layer in the top 
                         of the kernel, but when fitting, your input data should be of (num, flatten_shape)
            sovler: solver to optimize the kernel, 'softmax' or 'svm'
        '''
        print('Using tensorflow backend')
        self.solver = solver
        self.input_shape=input_shape
        shape = reduce(lambda x,y:x*y,input_shape,1)
        self._kernel = keras.models.Sequential()
        self._kernel.add(Reshape(input_shape,input_shape=(shape,)))
    def add_layer(self, 
            layer, 
            parameters, 
            dropout=0,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None):
        '''
        Adding a layer for your kernel, 4 different layers to choose:
            fully connected layer : 'dense',
            convolution2d : 'conv2d',
            maxpooling2d : 'maxpool',
            averagepooling2d: 'averagepool'
        Argument:
            layer : a string of layer's type
            parameters: a dict to define the parameters of layer
                        example：parameters for conv {'filters':32, 'kernel_size':(2,2)}.
        Raises:
            ValueError if the type of layer is not in ['conv', 'dense', 'maxpool', 'averagepool']
        '''
        try:
            units=parameters['units']
        except KeyError:
            units=1
        try:
            filters=parameters['filters']
        except KeyError:
            filters=None
        try:
            kernel_size=parameters['kernel_size']
        except KeyError:
            kernel_size=None
        try:
            strides = parameters['strides']
        except KeyError:
            strides=(1,1)
        try:
            padding = parameters['padding']
        except KeyError:
            padding = 'valid'
        try:
            pool_size = parameters['pool_size']
        except KeyError:
            pool_size = (2,2)
        try:
            data_format = parameters['data_format']
        except KeyError:
            data_format='channels_last'
        if kernel_initializer=='xaiver':
            kernel_initializer=tf.contrib.layers.xavier_initializer()
        if layer=='dense':
            self._kernel.add(Dense(units=units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, 
                                         bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer))
        if layer=='conv':
            self._kernel.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, 
                                         bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer))
        if layer=='maxpool':
            self._kernel.add(MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format))
        if layer=='averagepool':
            self._kernel.add(AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding, data_format=data_format))
        if layer not in ['conv', 'dense', 'maxpool', 'averagepool']:
            raise ValueError('No such layer named '+layer)
        if dropout>0:
            self._kernel.add(Dropout(dropout))


    def add_Flatten_layer(self):
        '''
        Add a flatten layer in the kernel
        '''
        self._kernel.add(Flatten())

    def add_Reshape_layer(self, target_shape):
        '''
        Add a reshape layer in the kernel
        '''
        self._kernel.add(Reshape(target_shape))

    def sample(self):
        '''
        A sample of the kernel's model:
        conv2d -> maxpool2d -> conv2d -> maxpool2d -> dense -> dropout -> dense -> dropout
        You can call function kernel_summary to see details
        '''
        self._kernel.add(Conv2D(filters=32, kernel_size=(2,2), activation='relu', kernel_initializer=tf.contrib.layers.xavier_initializer()))
        self._kernel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        self._kernel.add(Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation='relu', kernel_initializer=tf.contrib.layers.xavier_initializer()))
        self._kernel.add(MaxPooling2D(pool_size=(2,2)))
        self._kernel.add(Flatten())
        self._kernel.add(Dense(1024, activation='relu'))
        self._kernel.add(Dropout(0.4))
        self._kernel.add(Dense(128, activation='relu'))
        self._kernel.add(Dropout(0.4))

    def kernel_outputshape(self):
        '''
        Return the output shape of the kernel
        '''
        return self._kernel.output_shape
    
    def kernel_summary(self):
        '''
        Prints a string summary of the kernel's structure.
        '''
        self._kernel.summary()

    def model_summary(self):
        '''
        Prints a string summary of the whole model(kernel+solver)'s structure.
        '''
        self.model_train.summary()

    def kernel(self, X, y):
        '''
        return the inner product of two kernel function
        '''
        return np.dot(self._kernel.predict(X),self._kernel.predict(y).T)

    
    def build(self, nb_classes, l2_loss_rate=0.2, optimizer='adam'):
        '''
        Build the whole model (kernel+solver) 
        For different solver, different structures are implemented.
        Argument:
            nb_classes: int, the number of types of input data
            optimizer: string, default 'adam'
                choosing optimizer of the training model
                For example: 'adam', 'sgd', 'momentum'
        '''
        features = self._kernel.output
        feature_size = self._kernel.output_shape[1]
        if self.solver=='softmax':
            input_target = Input(shape=(1,))
            centers = Embedding(nb_classes, feature_size)(input_target)
            l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]), 1, keepdims=True), name='l2_loss')([features,centers])
            y_predict = Dense(nb_classes, activation='softmax', name='softmax')(features)
            self.model_train = keras.models.Model(inputs = [self._kernel.input,input_target], outputs = [y_predict,l2_loss])
            self.model_train.compile(optimizer=optimizer, loss=['sparse_categorical_crossentropy',lambda y_true,y_pred: y_pred], loss_weights=[1.,l2_loss_rate], metrics={'softmax':'accuracy'})
        if self.solver=='svm':
            if nb_classes!=2:
                raise ValueError('svm solver only accept binary classes')
            y_predict = Dense(1, kernel_regularizer=keras.regularizers.l2(l=0.5))(features)
            self.model_train = keras.models.Model(inputs=self._kernel.input, outputs=y_predict)
            self.model_train.compile(optimizer=optimizer, loss=svm_loss, metrics=[svm_accuracy])

    def fit(self, train_data, train_labels, batch_size=None, epochs=1, validation_data=None):
        '''
        Trains the model for a fixed number of epochs (iterations on a dataset).

        Arguments:
        train_data: Input data. It could be:
            A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
            A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
            A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
        train_labels: Target data. Like the input data x, it could be either Numpy array(s) or TensorFlow tensor(s). 
            It should be consistent with x (you cannot have Numpy inputs and tensor targets, or inversely)
        batch_size: Integer or None. 
            Number of samples per gradient update. If unspecified, batch_size will default to 32. 
        epochs: Integer. 
            Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
        validation_data: Data on which to evaluate the loss and any model metrics at the end of each epoch. 
            The model will not be trained on this data. validation_data could be: 
            - tuple (x_val, y_val) of Numpy arrays or tensors 
            - tuple (x_val, y_val, val_sample_weights) of Numpy arrays
        '''
        if self.solver=='svm':
            if len(list(set(train_labels)))!=2:
                raise ValueError('svm solver only accept binary classes')
            self.model_train.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=validation_data)
        if self.solver=='softmax':
            random_y = np.random.randn(train_data.shape[0],1)
            if validation_data is not None:
                data_val, labels_val = validation_data
                random_y_val = np.random.randn(data_val.shape[0],1)
                self.model_train.fit([train_data,train_labels], [train_labels,random_y], batch_size=batch_size, epochs=epochs,validation_data=([data_val,labels_val],[labels_val, random_y_val]))
            else:
                self.model_train.fit([train_data,train_labels], [train_labels,random_y], batch_size=batch_size, epochs=epochs)

    def evaluate(self, test_data, test_labels, batch_size=None):
        '''
        Returns the loss value & accuracy values for the model in test mode.
        Computation is done in batches.
        Arguments:
        test_data: Input data. It could be:
            A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
            A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
            A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
        test_labels: Target data. Like the input data x, it could be either Numpy array(s) or TensorFlow tensor(s). 
            It should be consistent with x (you cannot have Numpy inputs and tensor targets, or inversely)
        batch_size: Integer or None. 
            Number of samples per gradient update. If unspecified, batch_size will default to 32. 
        epochs: Integer. 
            Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided.
        '''
        _, acc = self.model_train.evaluate(test_data, test_labels, batch_size=batch_size)
        return acc

    def predict(self, data, batch_size=None):
        '''
        Generates output predictions for the input samples.
        Computation is done in batches.
        Arguments:
        data: Input data. It could be:
            A Numpy array (or array-like), or a list of arrays (in case the model has multiple inputs).
            A TensorFlow tensor, or a list of tensors (in case the model has multiple inputs).
            A dict mapping input names to the corresponding array/tensors, if the model has named inputs.
        batch_size: Integer or None. 
            Number of samples per gradient update. If unspecified, batch_size will default to 32. 
        '''
        label_predict = self.model_train.predict(data, batch_size=batch_size)
        return label_predict

            