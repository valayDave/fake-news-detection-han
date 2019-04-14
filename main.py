import pandas as pd
import train as model
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import os.path
import datetime
my_path = os.path.abspath(os.path.dirname(__file__))
PLOT_FOLDER = os.path.join(my_path, 'plots/')
MODEL_FOLDER = os.path.join(my_path, 'models/')

test_case_name = 'MULTI_LABEL_SPLIT1_HAN_vs_LSTM'

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
#model.train(df,'first_split_dataset_han')

HAN_MODEL_History,HAN_MODEL,HAN_accuracy = model.train_han(df,test_case_name)

LSTM_Model_History,LSTM_Model,LSTM_accuracy = model.train_lstm(df,test_case_name)

plot_models([LSTM_Model_History,HAN_MODEL_History],['LSTM','HAN'],'val_loss','Epochs','Validation Loss','Validation_Loss')
plot_models([LSTM_Model_History,HAN_MODEL_History],['LSTM','HAN'],'categorical_accuracy','Epochs','Accuracy','Accuracy')
plot_models([LSTM_Model_History,HAN_MODEL_History],['LSTM','HAN'],'loss','Epochs','Loss','Loss')
plot_models([LSTM_Model_History,HAN_MODEL_History],['LSTM','HAN'],'val_categorical_accuracy','Epochs','Validation Accuracy','Validation_Accuracy')
# plot_model(LSTM_Model, to_file=os.path.join(MODEL_FOLDER,test_case_name+'_Model_LSTM.png'), show_shapes=True, show_layer_names=True)
# plot_model(HAN_MODEL, to_file=os.path.join(MODEL_FOLDER,test_case_name+'_Model_LSTM.png'), show_shapes=True, show_layer_names=True)

LSTM_Model.save(os.path.join(MODEL_FOLDER,test_case_name+'_LSTM.h5'))
HAN_MODEL.save(os.path.join(MODEL_FOLDER,test_case_name+'_HAN.h5'))
print("#"*20+" Completed Execution "+"#"*20)