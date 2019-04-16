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
from keras.models import Model,load_model,model_from_json
import nltk
import re
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import roc_auc_score
from nltk import tokenize
import re
import os
import datetime
import glob
import pickle
my_path = os.path.abspath(os.path.dirname(__file__))

dataset_name = 'split-3'
dataset_path = 'datasets/'+dataset_name+'.csv'
word_index = None
with open('training-data/'+dataset_name+'-tokenizer.pickle', 'rb') as handle: #Load the Word Tokenized Index
    word_index = pickle.load(handle)

final_dataframe = pd.read_csv(dataset_path)
models_for_cases = [
    'Test-cases/Case-11/',
    'Test-cases/Case-2/',
    'Test-cases/Case-3/',
    'Test-cases/Case-4/'
]

case_models = {
    1 : {
        'han': {
            'json':'Test-cases/Case-20/models/HAN.json',
            'h5':'Test-cases/Case-20/models/HAN.h5'
        },
        # 'lstm': {
        #     'json':'Test-cases/Case-11/models/LSTM.json',
        #     'h5':'Test-cases/Case-11/models/LSTM.h5'
        # },
        # 'han3': {
        #     'json':'Test-cases/Case-11/models/HAN3.json',
        #     'h5':'Test-cases/Case-11/models/HAN3.h5'
        # }
    },
    # 2 : {
    #     'han': {
    #         'json':'Test-cases/Case-11/models/HAN.json',
    #         'h5':'Test-cases/Case-11/models/HAN.h5'
    #     },
    #     'lstm': {
    #         'json':'Test-cases/Case-11/models/LSTM.json',
    #         'h5':'Test-cases/Case-11/models/LSTM.h5'
    #     },
    #     'han3': {
    #         'json':'Test-cases/Case-11/models/HAN3.json',
    #         'h5':'Test-cases/Case-11/models/HAN3.h5'
    #     }
    # },
    # 3 : {
    #     'han': {
    #         'json':'Test-cases/Case-11/models/HAN.json',
    #         'h5':'Test-cases/Case-11/models/HAN.h5'
    #     },
    #     'lstm': {
    #         'json':'Test-cases/Case-11/models/LSTM.json',
    #         'h5':'Test-cases/Case-11/models/LSTM.h5'
    #     },
    #     'han3': {
    #         'json':'Test-cases/Case-11/models/HAN3.json',
    #         'h5':'Test-cases/Case-11/models/HAN3.h5'
    #     }
    # }
}

MAX_SEQUENCE_LENGTH = 1000
max_features=200000
max_senten_len=40
max_senten_num=12
embed_size=100
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
sample_dataset = False
NUM_SAMPLES = 20


#TODO : Check if the dataset is downloaded. 
# TODO : Save Indexed Dataset and Reuse in the Other Traing Sessions. 

# train_path = os.path.join(my_path,'datasets/train.csv')
# test_path = os.

def log(statement):
    print("+"*30)
    print(statement)
    print("+"*30)

def print_seperation():
    print("=============================================================================================")

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
        # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
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


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()


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

def preprocess_data(data_frame):
    data_frame['body'] = data_frame['title'] +'. ' +data_frame  ['body']
    data_frame = data_frame[['body', 'label','title']]
    shuffle(data_frame).reset_index()
    data_frame =data_frame[~data_frame['body'].isnull()]
    log(len(data_frame['label'].unique()))
    return data_frame


def generate_test_data():
    global word_index
    global final_dataframe
    data_frame = final_dataframe
    if word_index is None:
        log("Word Index is Not Present. Please Download To Ensure Predictions")
        return None        
    paras = []
    texts = []
    sent_lens = []
    sent_nums = []
    if sample_dataset:
        data_frame = data_frame.sample(n=NUM_SAMPLES)
    labels = data_frame['label']
    headlines = []
    for row in data_frame.itertuples():
        body_text = row.body
        headline_text = clean_str(row.title)
        text = clean_str(body_text)
        texts.append(text)
        headlines.append(headline_text)
        sentences = tokenize.sent_tokenize(text)
        sent_nums.append(len(sentences))
        for sent in sentences:
            sent_lens.append(len(text_to_word_sequence(sent)))
        paras.append(sentences)
    tokenized_rnn_body = np.zeros((len(texts), MAX_SEQUENCE_LENGTH), dtype='int32')

    #Tokenize For the RNN
    for i,text in enumerate(texts):
        wordTokens = text_to_word_sequence(text)
        k=0
        for _,word in enumerate(wordTokens):
            try:
                if k<MAX_SEQUENCE_LENGTH and word_index[word]<max_features:
                    tokenized_rnn_body[i,k] = word_index[word]
                    k=k+1
            except:
                print(word)
                pass

    #Tokenize For the HAN
    tokenized_han_body = np.zeros((len(texts), max_senten_num, max_senten_len), dtype='int32')
    for i, sentences in enumerate(paras):
        #print(sentences)
        #print(i)
        for j, sent in enumerate(sentences):
            if j< max_senten_num:
                wordTokens = text_to_word_sequence(sent)
                k=0
                for _, word in enumerate(wordTokens):
                    try:
                        if k<max_senten_len and word_index[word]<max_features:
                            # print(word,word_index[word])   
                            tokenized_han_body[i,j,k] = word_index[word]
                            k=k+1
                    except:
                        print(word)
                        pass
    #Datastructure from the above [i:Article_Number,j:Sentence_number,k:Word Index]
    
    #Tokenize Headlines For the 3HAN
    tokenized_headlines = np.zeros((len(texts), max_senten_len), dtype='int32')
    for i,headline in enumerate(headlines):
        wordTokens = text_to_word_sequence(headline)
        for j,word in enumerate(wordTokens):
            try:
                if j<max_senten_len and word_index[word]<max_features:
                    tokenized_headlines[i,j] = word_index[word]
            except:
                print(word)
                pass
    seperated_labels = pd.get_dummies(labels)
    indices = np.arange(tokenized_han_body.shape[0])

    # np.random.shuffle(indices)
    tokenized_han_body = tokenized_han_body[indices]
    seperated_labels = seperated_labels.iloc[indices]
    tokenized_headlines =tokenized_headlines[indices]
    tokenized_rnn_body = tokenized_rnn_body[indices]

    nb_validation_samples = int(VALIDATION_SPLIT * tokenized_han_body.shape[0])

    pre_rnn_x_train = tokenized_rnn_body[:-nb_validation_samples]
    pre_rnn_y_train = tokenized_rnn_body[:-nb_validation_samples]
    pre_han_x_train = tokenized_han_body[:-nb_validation_samples]
    pre_han_y_train = seperated_labels[:-nb_validation_samples]
    pre_headline_train = tokenized_headlines[:-nb_validation_samples]

    nb_test_samples = int(TEST_SPLIT * pre_han_x_train.shape[0])

    han_x_test = pre_han_x_train[-nb_test_samples:]  #Create Test Samples From the Train Samples 
    han_y_test = seperated_labels[-nb_test_samples:]
    
    rnn_x_test = pre_rnn_x_train[-nb_test_samples:]
    rnn_y_test = seperated_labels[-nb_test_samples:]
    
    han3_headlines_test = pre_headline_train[-nb_test_samples:]
    
    han3_x_test = [han_x_test,han3_headlines_test]
  
    han_vectors = (han_x_test,han_y_test)
    rnn_vectors = (rnn_x_test,rnn_y_test)
    han3_vectors = (han3_x_test,han_y_test)    

    word_index = word_index

    #Create the Embedding Matrix
    return han_vectors,han3_vectors,rnn_vectors # each vector is (testx,testy)

possible_models = ['han','rnn','han3']

han_test_vectors,han3_test_vectors,rnn_test_vectors = generate_test_data()

def predict_for_case(json_path,h5_path,model_name):
    j_file = open(json_path, 'r')
    loaded_json_model = j_file.read()
    j_file.close()
    loaded_model = model_from_json(loaded_json_model,custom_objects={'AttentionWithContext':AttentionWithContext})
    # predict_from_case(models_for_cases[0])
    loaded_model.load_weights(h5_path)
    loaded_model.summary()
    test_vectors = None
    if model_name  == 'han':
        test_vectors = han_test_vectors
    elif model_name  == 'lstm':
        test_vectors = rnn_test_vectors
    elif model_name  == 'han3':
        test_vectors = han3_test_vectors
    test_set_accuracy,test_df = get_testset_accuracy(loaded_model,test_vectors,test_vectors[1].columns.values)
    log("Accuracy Distribution Over the Labels.")
    log(test_df)
    log("Test Accuracy : "+str(test_set_accuracy))


for case_id in case_models:
    for model_name in case_models[case_id]:
        print_seperation()
        log("Running "+ model_name+" Test Case "+str(case_id))
        predict_for_case(case_models[case_id][model_name]['json'],case_models[case_id][model_name]['h5'],model_name)
        print_seperation()

