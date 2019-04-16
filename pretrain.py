#Create the Word Token files Over here. 
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
import pickle

dataset_path = [
    'split-1.csv', #Contains Fake real and Opinions. 
    'split-3.csv' #Contains Fake and Real only
]


sample_dataset = False
NUM_SAMPLES = 50
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
max_features = 200000


def log(statement):
    print("+"*30)
    print(statement)
    print("+"*30)

def get_word_index(data_frame):
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
    word_index = tokenizer.word_index
    return word_index

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

def prepare_dataset(file_name,path):
    df = pd.read_csv(os.path.join(DATASET_FOLDER,path))
    df = preprocess_data(df)
    word_index = get_word_index(df)
    with open('training-data/'+file_name+'-tokenizer.pickle', 'wb') as handle:
        pickle.dump(word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

prepare_dataset('split-3',dataset_path[1])

