import pandas as pd
import train as model
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import os.path
import datetime
my_path = os.path.abspath(os.path.dirname(__file__))
PLOT_FOLDER = os.path.join(my_path, 'plots/')
MODEL_FOLDER = os.path.join(my_path, 'models/')
TRAINING_DATA_FOLDER = os.path.join(my_path,'datasets/')

test_case_name = 'HAN_LOSS_OPT'

data_name = ['split-1','split-3']

training_files = {
    'han' : [
        'han_train_x.csv',
        'han_train_y.csv',
        'han_val_x.csv' ,
        'han_val_y.csv'  ,
        'han_test_x.csv',
        'han_test_y.csv'
        ],
    'han3' : [
        'han3_train_x.csv',
        'han3_train_y.csv',
        'han3_val_x.csv' ,
        'han3_val_y.csv'  ,
        'han3_test_x.csv',
        'han3_test_y.csv'
        ],
    'rnn' : [
        'rnn_train_x.csv',
        'rnn_train_y.csv',
        'rnn_val_x.csv' ,
        'rnn_val_y.csv'  ,
        'rnn_test_x.csv',
        'rnn_test_y.csv'
        ]
}

#model.train(df,'first_split_dataset_han')
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

MODEL_CHOICE = ['han','rnn','han3']

dataframes = [pd.read_csv(os.path.join(TRAINING_DATA_FOLDER,file)) for file in training_files['han']]

print([df.shape for df in dataframes])

dataset_path = 'datasets/split-1.csv'

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
HAN_MODEL_History,HAN_MODEL,HAN_accuracy = model.train_han(df,test_case_name,LSTM_COUNT,DROPOUT_VALUE,REGULARIZER,REG_VALUE)
han_dataframe = pd.DataFrame(HAN_MODEL_History.history)
han_dataframe.to_csv(os.path.join(MODEL_FOLDER,test_case_name+'_HAN.csv'))

# LSTM_Model_History,LSTM_Model,LSTM_accuracy = model.train_lstm(df,test_case_name)
# HAN3_MODEL_History,HAN3_MODEL,HAN3_accuracy = model.train_han_3(df,test_case_name)


# plot_models([LSTM_Model_History,HAN_MODEL_History,HAN3_MODEL_History],['LSTM','HAN','HAN-3'],'val_loss','Epochs','Validation Loss','Validation_Loss')
# plot_models([LSTM_Model_History,HAN_MODEL_History,HAN3_MODEL_History],['LSTM','HAN','HAN-3'],'categorical_accuracy','Epochs','Accuracy','Accuracy')
# plot_models([LSTM_Model_History,HAN_MODEL_History,HAN3_MODEL_History],['LSTM','HAN','HAN-3'],'loss','Epochs','Loss','Loss')
# plot_models([LSTM_Model_History,HAN_MODEL_History,HAN3_MODEL_History],['LSTM','HAN','HAN-3'],'val_categorical_accuracy','Epochs','Validation Accuracy','Validation_Accuracy')
# # plot_model(LSTM_Model, to_file=os.path.join(MODEL_FOLDER,test_case_name+'_Model_LSTM.png'), show_shapes=True, show_layer_names=True)
# # plot_model(HAN_MODEL, to_file=os.path.join(MODEL_FOLDER,test_case_name+'_Model_LSTM.png'), show_shapes=True, show_layer_names=True)
# lstm_dataframe = pd.DataFrame(LSTM_Model_History.history)
# han3_dataframe = pd.DataFrame(HAN3_MODEL_History.history)

# LSTM_Model.save(os.path.join(MODEL_FOLDER,test_case_name+'_LSTM.h5'))
# HAN_MODEL.save(os.path.join(MODEL_FOLDER,test_case_name+'_HAN.h5'))
# HAN3_MODEL.save(os.path.join(MODEL_FOLDER,test_case_name+'_HAN3.h5'))

# han3_dataframe.to_csv(os.path.join(MODEL_FOLDER,test_case_name+'_HAN3.csv'))
# lstm_dataframe.to_csv(os.path.join(MODEL_FOLDER,test_case_name+'_LSTM.csv'))
print("#"*20+" Completed Execution "+"#"*20)