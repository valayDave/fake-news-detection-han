#TODO : Create the Dataset Files for the Train and the test from here. 

#TODO : Create the Word Token files Over here. 

#TODO: Change Train.py to load the files from drive to train the different models.  
import pandas as pd
import numpy as np
import re
import os.path
import nltk
from sklearn.utils import shuffle
from nltk import tokenize

my_path = os.path.abspath(os.path.dirname(__file__))
MAX_SEQUENCE_LENGTH = 1000
max_sentence_num = 12
max_sentence_len = 40
DATASET_FOLDER = os.path.join(my_path,'datasets/')
from keras.preprocessing.text import Tokenizer,  text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
dataset_path = [
    'split-1.csv', #Contains Fake real and Opinions. 
    'split-3.csv' #Contains Fake and Real only
]

sample_dataset = False
NUM_SAMPLES = 20
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
max_features = 200000


def log(statement):
    print("+"*30)
    print(statement)
    print("+"*30)

# data_frame : ['body','label','title'] : Generates the Dataset for the HAN,RNN and the 3HAN
def generate_train_test_split(data_frame):
    paras = []
    #labels = []
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
    
    tokenizer = Tokenizer(num_words=max_features, oov_token=True)
    tokenizer.fit_on_texts(texts)

    log("Data is Now Prepared. ")
    sequences = tokenizer.texts_to_sequences(texts)
    tokenized_rnn_body = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    tokenized_han_body = np.zeros((len(texts), max_sentence_num, max_sentence_len), dtype='int32')
    for i, sentences in enumerate(paras):
        #print(sentences)
        #print(i)
        for j, sent in enumerate(sentences):
            if j< max_sentence_num:
                wordTokens = text_to_word_sequence(sent)
                k=0
                for _, word in enumerate(wordTokens):
                    try:
                        if k<max_sentence_len and tokenizer.word_index[word]<max_features:
                            # print(word,tokenizer.word_index[word])   
                            tokenized_body[i,j,k] = tokenizer.word_index[word]
                            k=k+1
                    except:
                        pass
    #Datastructure from the above [i:Article_Number,j:Sentence_number,k:Word Index]
    tokenized_headlines = np.zeros((len(texts), max_sentence_len), dtype='int32')
    for i,headline in enumerate(headlines):
        wordTokens = text_to_word_sequence(headline)
        for j,word in enumerate(wordTokens):
            try:
                if j<max_sentence_len and tokenizer.word_index[word]<max_features:
                    tokenized_headlines[i,j] = tokenizer.word_index[word]
            except:
                #print(word)
                pass
    seperated_labels = pd.get_dummies(labels)
    indices = np.arange(tokenized_han_body.shape[0])
    log("Creating the three Datasets. ")
    np.random.shuffle(indices)
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
    han_y_test = pre_han_x_train[-nb_test_samples:]
    han_x_train = pre_han_x_train[:-nb_test_samples]
    han_y_train = pre_han_y_train[:-nb_test_samples]
    han_x_val = tokenized_han_body[-nb_validation_samples:]
    han_y_val = seperated_labels[-nb_validation_samples:]
    
    rnn_x_test = pre_rnn_x_train[-nb_test_samples:]
    rnn_y_test = pre_rnn_y_train[-nb_test_samples:]
    rnn_x_train = pre_rnn_x_train[:-nb_test_samples]
    rnn_y_train = pre_rnn_y_train[:-nb_test_samples]
    rnn_x_val = tokenized_rnn_body[-nb_validation_samples:]
    rnn_y_val = seperated_labels[-nb_validation_samples:]

    
    han3_headlines_val = tokenized_headlines[-nb_validation_samples:]
    han3_headlines_test = pre_headline_train[-nb_test_samples:]
    han3_headlines_train = pre_headline_train[:-nb_test_samples]
   
    han3_x_val = [han_x_val,han3_headlines_val]
    han3_x_train = [han_x_train,han3_headlines_train]
    han3_x_test = [han_x_test,han3_headlines_test]

  
    han_vectors = ((han_x_train,han_y_train),(han_x_val,han_y_val),(han_x_test,han_y_test))
    rnn_vectors = ((rnn_x_train,rnn_y_train),(rnn_x_val,rnn_y_val),(rnn_x_test,rnn_y_test))
    han3_vectors = ((han3_x_train,han_y_train),(han3_x_val,han_y_val),(han3_x_test,han_y_test))

    
    word_index = tokenizer.word_index
    
    #Create the Embedding Matrix
    return word_index,han_vectors,han3_vectors,rnn_vectors # each vector is ((trainx,trainy),(validatex,validatey),(testx,testy))

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()


def preprocess_data(data_frame):
    data_frame['body'] = data_frame['title'] +'. ' +data_frame  ['body']
    data_frame = data_frame[['body', 'label','title']]
    shuffle(data_frame).reset_index()
    data_frame =data_frame[~data_frame['body'].isnull()]
    log(len(data_frame['label'].unique()))
    return data_frame

def prepare_dataset(file_name):
    df = pd.read_csv(os.path.join(DATASET_FOLDER,file_name))
    df = preprocess_data(df)

    word_index,han_vectors,han3_vectors,rnn_vectors = generate_train_test_split(df)

    han_train,han_val,han_test = han_vectors
    log(han_train[0].shape)
    log(han_train[1].columns.values)

prepare_dataset(dataset_path[0])

