import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer,  text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout,concatenate,Masking
from keras.layers.core import Reshape
from keras import backend as K
from keras import optimizers
from keras.models import Model,load_model
import nltk
import re
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import roc_auc_score
from nltk import tokenize
import re
import os
import datetime
my_path = os.path.abspath(os.path.dirname(__file__))
train_dataset_path = "" #These should be word Indexes.
validation_dataset_path = "" #These should be word Indexes.
test_dataset_path = "" #These should be word Indexes. 

#TODO : Check if the dataset is downloaded. 
# TODO : Save Indexed Dataset and Reuse in the Other Traing Sessions. 

# train_path = os.path.join(my_path,'datasets/train.csv')
# test_path = os.

class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatibl|e with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

def get_testset_accuracy(model,test_vectors,label_arr):
    predictions = model.predict(test_vectors[0])
    predicted_classes = np.argmax(predictions,axis=1)
    iterator_index = 0
    correct_preds = 0
    test_set_accuracy = 0
    label_pred = [0 for label in label_arr]
    label_total = [0 for label in label_arr]
    for actual_vals in test_vectors[1].values:
        if actual_vals[predicted_classes[iterator_index]] == 1:
            correct_preds+=1
            label_pred[predicted_classes[iterator_index]]+=1
        label_total[predicted_classes[iterator_index]]+=1
        iterator_index+=1
    test_set_accuracy = float(correct_preds/iterator_index)
    val = []
    for i in range(0,label_arr.__len__()):
        if label_total[i] == 0:
            val.append([label_arr[i],label_pred[i],label_total[i],0])
        else:
            val.append([label_arr[i],label_pred[i],label_total[i],(label_pred[i]/label_total[i])])
    df = pd.DataFrame(val,columns=['Label','Correct_Label_Predictions','Total_Label_Docs','Prediction_Accuracy'])    
    return test_set_accuracy,df

def get_test_data(file_path):
    #Todo : go to the test case path and load its dataframe 
    return ([],[]) # (Target,predictions)

test_vectors = get_test_data("") # (Target,predictions)
loaded_model = load_model("models/test-prod/HAN-test-1.h5",{'AttentionWithContext':AttentionWithContext})

loaded_model.summary()



test_set_accuracy,df = get_testset_accuracy(loaded_model,test_vectors,test_vectors[1].columns.values)
