import pandas as pd
import train as model
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import os.path
import datetime
import pickle
my_path = os.path.abspath(os.path.dirname(__file__))
PLOT_FOLDER = os.path.join(my_path, 'plots/')
MODEL_FOLDER = os.path.join(my_path, 'models/')
TRAINING_DATA_FOLDER = os.path.join(my_path,'datasets/')

test_case_name = 'HAN_LOSS_OPT'

LSTM_COUNT = 100
DROPOUT_VALUE = 0.5
REGULARIZER = 1
L1_REG_VALUE = 0.001
L2_REG_VALUE = 1e-13
REG_VALUE = None
if REGULARIZER == 1:
    REG_VALUE = L1_REG_VALUE
else:
    REG_VALUE = L2_REG_VALUE

dataset_name = 'split-3'
dataset_path = 'datasets/'+dataset_name+'.csv'
# loading Word Index.
word_index = None
with open('training-data/'+dataset_name+'-tokenizer.pickle', 'rb') as handle:
    word_index = pickle.load(handle)

def plot_models(model_arr,model_name_arr,model_key,xlabel,ylabel,plot_name):
    fig1 = plt.figure()
    for model in model_arr:
        plt.plot(model.history[model_key])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(model_name_arr, loc='upper left')
    fig1.savefig(os.path.join(PLOT_FOLDER,test_case_name+'_'+plot_name+'.png'))

df = pd.read_csv(dataset_path)

test_case_name +='_'+'_'.join(map(str,[LSTM_COUNT,DROPOUT_VALUE,REGULARIZER,REG_VALUE]))
HAN_MODEL_History,HAN_MODEL,HAN_accuracy = model.train_han(df,word_index,test_case_name,LSTM_COUNT,DROPOUT_VALUE,REGULARIZER,REG_VALUE)
han_dataframe = pd.DataFrame(HAN_MODEL_History.history)
han_dataframe.to_csv(os.path.join(MODEL_FOLDER,test_case_name+'_HAN.csv'))
HAN_MODEL.save(os.path.join(MODEL_FOLDER,test_case_name+'_HAN.h5'))

with open(os.path.join(MODEL_FOLDER,test_case_name+'_HAN.json'), "w") as j_file:
    j_file.write(HAN_MODEL.to_json())

LSTM_Model_History,LSTM_Model,LSTM_accuracy = model.train_lstm(df,word_index,test_case_name)
LSTM_Model.save(os.path.join(MODEL_FOLDER,test_case_name+'_LSTM.h5'))
lstm_dataframe = pd.DataFrame(LSTM_Model_History.history)
lstm_dataframe.to_csv(os.path.join(MODEL_FOLDER,test_case_name+'_LSTM.csv'))
with open(os.path.join(MODEL_FOLDER,test_case_name+'_LSTM.json'), "w") as j_file:
    j_file.write(LSTM_Model.to_json())

HAN3_MODEL_History,HAN3_MODEL,HAN3_accuracy = model.train_han_3(df,word_index,test_case_name)
han3_dataframe = pd.DataFrame(HAN3_MODEL_History.history)
HAN3_MODEL.save(os.path.join(MODEL_FOLDER,test_case_name+'_HAN3.h5'))
han3_dataframe.to_csv(os.path.join(MODEL_FOLDER,test_case_name+'_HAN3.csv'))
with open(os.path.join(MODEL_FOLDER,test_case_name+'_HAN3.json'), "w") as j_file:
    j_file.write(HAN3_MODEL.to_json())

plot_models([LSTM_Model_History,HAN_MODEL_History,HAN3_MODEL_History],['LSTM','HAN','HAN-3'],'val_loss','Epochs','Validation Loss','Validation_Loss')
plot_models([LSTM_Model_History,HAN_MODEL_History,HAN3_MODEL_History],['LSTM','HAN','HAN-3'],'categorical_accuracy','Epochs','Accuracy','Accuracy')
plot_models([LSTM_Model_History,HAN_MODEL_History,HAN3_MODEL_History],['LSTM','HAN','HAN-3'],'loss','Epochs','Loss','Loss')
plot_models([LSTM_Model_History,HAN_MODEL_History,HAN3_MODEL_History],['LSTM','HAN','HAN-3'],'val_categorical_accuracy','Epochs','Validation Accuracy','Validation_Accuracy')

# # # plot_model(LSTM_Model, to_file=os.path.join(MODEL_FOLDER,test_case_name+'_Model_LSTM.png'), show_shapes=True, show_layer_names=True)
# # # plot_model(HAN_MODEL, to_file=os.path.join(MODEL_FOLDER,test_case_name+'_Model_LSTM.png'), show_shapes=True, show_layer_names=True)


print("#"*20+" Completed Execution "+"#"*20)