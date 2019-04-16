import pandas as pd
import pickle
from sklearn.utils import shuffle


dataset_name = 'split-3'
dataset_path = 'datasets/'+dataset_name+'.csv'
word_index = None
with open('training-data/'+dataset_name+'-tokenizer.pickle', 'rb') as handle: #Load the Word Tokenized Index
    word_index = pickle.load(handle)

final_dataframe = pd.read_csv(dataset_path)

test_frac = 0.15


def preprocess_data(data_frame):
    data_frame['body'] = data_frame['title'] +'. ' +data_frame  ['body']
    data_frame = data_frame[['body', 'label','title']]
    shuffle(data_frame).reset_index()
    data_frame =data_frame[~data_frame['body'].isnull()]
    log(len(data_frame['label'].unique()))
    return data_frame

test_dataframe = final_dataframe.sample(frac=test_frac)
train_dataframe = final_dataframe.drop(test_dataframe.index)

test_dataframe.to_csv('datasets/'+dataset_name+'-test.csv')
train_dataframe.to_csv('datasets/'+dataset_name+'-train.csv')