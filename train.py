import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer,  text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed, Dropout
from keras import backend as K
from keras import optimizers
from keras.models import Model
import nltk
import re
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import roc_auc_score
from nltk import tokenize
import seaborn as sns
from sklearn.utils import shuffle
import re
import os
import datetime
my_path = os.path.abspath(os.path.dirname(__file__))
print(my_path)


MAX_SEQUENCE_LENGTH = 1000
max_features=200000
max_senten_len=40
max_senten_num=12
embed_size=100
VALIDATION_SPLIT = 0.2
learning_rate = 0.6
REG_PARAM = 1e-13
PLOT_FOLDER = os.path.join(my_path, 'plots/')
MODEL_FOLDER = os.path.join(my_path, 'models/')
sample_dataset = True
NUM_SAMPLES = 5

GLOVE_DIR = "glove.6B.100d.txt"



def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()

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

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

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

def log(statement):
    print("+"*30)
    print(statement)
    print("+"*30)

# TODO : Train on basis of the size of the data.

# TODO : Train on basis of the word vectors. 

# TODO : Train basis different learing rates. 

# data_frame : ['body','label']
def generate_han_embedding_matrix(data_frame):
    paras = []
    #labels = []
    texts = []
    sent_lens = []
    sent_nums = []
    if sample_dataset:
        data_frame = data_frame.sample(n=NUM_SAMPLES)

    labels = data_frame['label']
    text = data_frame['body']

    for body_text in data_frame['body']:
        text = clean_str(body_text)
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        sent_nums.append(len(sentences))
        for sent in sentences:
            sent_lens.append(len(text_to_word_sequence(sent)))
        paras.append(sentences)
    
    tokenizer = Tokenizer(num_words=max_features, oov_token=True)
    tokenizer.fit_on_texts(texts)

    data = np.zeros((len(texts), max_senten_num, max_senten_len), dtype='int32')
    for i, sentences in enumerate(paras):
        for j, sent in enumerate(sentences):
            if j< max_senten_num:
                wordTokens = text_to_word_sequence(sent)
                k=0
                for _, word in enumerate(wordTokens):
                    try:
                        if k<max_senten_len and tokenizer.word_index[word]<max_features:
                            data[i,j,k] = tokenizer.word_index[word]
                            k=k+1
                    except:
                        print(word)
                        pass
    #TODO: Figure the above
    print(data.shape)

    #Converts Labels to Digits. 
    seperated_labels = pd.get_dummies(labels)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    seperated_labels = seperated_labels.iloc[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = seperated_labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = seperated_labels[-nb_validation_samples:]

    train_vectors = (x_train,y_train)
    validation_vectors = (x_val,y_val)
    word_index = tokenizer.word_index

    #Create the Embedding Matrix
    embeddings_index = {}
    f = open(GLOVE_DIR)
    for line in f:
        try:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except:
            print(word)
            pass
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, embed_size))
    
    #TODO: Figure Need for Absent word calc.
    absent_words = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            absent_words += 1
    print('Total absent words are', absent_words, 'which is', "%0.2f" % (absent_words * 100 / len(word_index)), '% of total words')

    return embedding_matrix,data,seperated_labels,word_index,train_vectors,validation_vectors

def generate_rnn_embedding_matrix(data_frame):
    paras = []
    #labels = []
    texts = []
    sent_lens = []
    sent_nums = []
    if sample_dataset:
        data_frame = data_frame.sample(n=NUM_SAMPLES)
    labels = data_frame['label']
    text = data_frame['body']
    print(data_frame.body.shape[0])
    for body_text in data_frame['body']:
        text = clean_str(body_text)
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        sent_nums.append(len(sentences))
        for sent in sentences:
            sent_lens.append(len(text_to_word_sequence(sent)))
        paras.append(sentences)

    tokenizer = Tokenizer(num_words=max_features, oov_token=True)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    #Converts Labels to Digits. 
    log('Shape of Data Tensor:')
    log(data.shape)
    seperated_labels = pd.get_dummies(labels)
    log('Shape of Label Tensor:')
    log(seperated_labels.shape)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    seperated_labels = seperated_labels.iloc[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = seperated_labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = seperated_labels[-nb_validation_samples:]

    train_vectors = (x_train,y_train)
    validation_vectors = (x_val,y_val)
    word_index = tokenizer.word_index

    #Create the Embedding Matrix
    embeddings_index = {}
    f = open(GLOVE_DIR)
    for line in f:
        try:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except:
            print(word)
            pass
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, embed_size))
    
    #TODO: Figure Need for Absent word calc.
    absent_words = 0
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            absent_words += 1
    print('Total absent words are', absent_words, 'which is', "%0.2f" % (absent_words * 100 / len(word_index)), '% of total words')

    return embedding_matrix,data,seperated_labels,word_index,train_vectors,validation_vectors

def preprocess_data(data_frame):
    data_frame['body'] = data_frame['title'] +'. ' +data_frame  ['body']
    data_frame = data_frame[['body', 'label']]
    shuffle(data_frame).reset_index()
    data_frame =data_frame[~data_frame['body'].isnull()]
    log(len(data_frame['label'].unique()))
    return data_frame

def plot_figure(model_op,plot_title,lengend,keys,xlabel,ylabel,network_name,plot_name):
    fig1 = plt.figure()
    for key in keys: # ['acc','val_acc']
        plt.plot(model_op.history[key])
        #plt.plot(model_op.history['val_acc'])
    plt.title(plot_title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(lengend, loc='upper left')
    save_time=datetime.datetime.now().strftime('%b-%d-%Y-%H')
    plot_name = network_name+'-'+plot_name+'-'+save_time 
    plot_path = PLOT_FOLDER+plot_name+'-'+plot_title+'.png'
    log(plot_path)
    plot_path = os.path.join(my_path,plot_path)
    #plt.savefig(plot_path)
    fig1.savefig(plot_path)


def train_lstm(data_frame,plot_name):
    model_name = 'Bidirectional_LSTM'
    data_frame = preprocess_data(data_frame)
    num_labels = len(data_frame['label'].unique())
    log("Running Bidirectional LSTM")
    embedding_matrix,data,seperated_labels,word_index,train_vectors,validation_vectors = generate_rnn_embedding_matrix(data_frame)
    embedding_layer = Embedding(len(word_index) + 1,embed_size,weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
    preds = Dense(num_labels, activation='softmax')(l_lstm)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
    model.summary()
    checkpoint_model_path =os.path.join(MODEL_FOLDER,model_name+'-'+plot_name+'.h5')
    checkpoint = ModelCheckpoint(checkpoint_model_path, verbose=0, monitor='val_loss',save_best_only=True, mode='auto') 
    history = model.fit(train_vectors[0], train_vectors[1], validation_data=(validation_vectors[0], validation_vectors[1]), epochs=5, batch_size=512, callbacks=[checkpoint])
    #Plot for Accurracy
    plot_figure(history,'Model Accuracy',['train','test'],['acc','val_acc'],'accuracy','epoch','Bidirectional_LSTM',plot_name)
    # summarize history for loss
    plot_figure(history,'Model Loss',['train','test'],['loss','val_loss'],'loss','epoch','Bidirectional_LSTM',plot_name)
    model_json_path = os.path.join(MODEL_FOLDER,model_name+'-'+plot_name+".json")
    model_json = model.to_json()
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    h5_path = os.path.join(MODEL_FOLDER,model_name+'-'+plot_name+".h5")
    model.save_weights(h5_path)
    log("Saved model to disk")

# data_frame : ['content_id','url','title','body','label']
def train_han(data_frame,plot_name):
    model_name = 'HAN'
    data_frame = preprocess_data(data_frame)
    log("Running HAN")
    num_labels = len(data_frame['label'].unique())
    embedding_matrix,data,seperated_labels,word_index,train_vectors,validation_vectors = generate_han_embedding_matrix(data_frame)
    embedding_layer = Embedding(len(word_index) + 1,embed_size,weights=[embedding_matrix], input_length=max_senten_len, trainable=False)
    
    #LSTM Regularizers --> Figure More
    l2_reg = regularizers.l2(REG_PARAM)
    
    #Word Encoding Layers
    word_input = Input(shape=(max_senten_len,), dtype='float32')
    word_sequences = embedding_layer(word_input)
    word_lstm = Bidirectional(LSTM(150, return_sequences=True, kernel_regularizer=l2_reg))(word_sequences)
    word_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg))(word_lstm)
    word_att = AttentionWithContext()(word_dense)
    wordEncoder = Model(word_input, word_att)
    
    #Sentence Encoding Layers
    sent_input = Input(shape=(max_senten_num, max_senten_len), dtype='float32')
    sent_encoder = TimeDistributed(wordEncoder)(sent_input)
    sent_lstm = Bidirectional(LSTM(150, return_sequences=True, kernel_regularizer=l2_reg))(sent_encoder)
    sent_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg))(sent_lstm)
    sent_att = Dropout(0.5)(AttentionWithContext()(sent_dense))
    preds = Dense(num_labels, activation='softmax')(sent_att)
    model = Model(sent_input, preds)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    log("Training Model ")
    checkpoint = ModelCheckpoint('best_model.h5', verbose=0, monitor='val_loss',save_best_only=True, mode='auto') 
    history = model.fit(train_vectors[0], train_vectors[1], validation_data=(validation_vectors[0], validation_vectors[1]), epochs=5, batch_size=512, callbacks=[checkpoint])
   
    #Plot for Accurracy
    plot_figure(history,'Model Accuracy',['train','test'],['acc','val_acc'],'accuracy','epoch',model_name,plot_name)
    # summarize history for loss
    plot_figure(history,'Model Loss',['train','test'],['loss','val_loss'],'loss','epoch',model_name,plot_name)
    log("Plots are Written ")
    # serialize model to JSON
    model_json_path = os.path.join(MODEL_FOLDER,model_name+'-'+plot_name+".json")
    model_json = model.to_json()
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    h5_path = os.path.join(MODEL_FOLDER,model_name+'-'+plot_name+".h5")
    model.save_weights(h5_path)
    log("Saved model to disk")

    #TODO : Figure Confusion Matrix. 
    

