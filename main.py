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

HAN3_MODEL_History,HAN3_MODEL,HAN3_accuracy = model.train_han_3(df,test_case_name)

LSTM_Model_History,LSTM_Model,LSTM_accuracy = model.train_lstm(df,test_case_name)

HAN_MODEL_History,HAN_MODEL,HAN_accuracy = model.train_han(df,test_case_name)

plot_models([LSTM_Model_History,HAN_MODEL_History],['LSTM','HAN','HAN-3'],'val_loss','Epochs','Validation Loss','Validation_Loss')
plot_models([LSTM_Model_History,HAN_MODEL_History],['LSTM','HAN','HAN-3'],'categorical_accuracy','Epochs','Accuracy','Accuracy')
plot_models([LSTM_Model_History,HAN_MODEL_History],['LSTM','HAN','HAN-3'],'loss','Epochs','Loss','Loss')
plot_models([LSTM_Model_History,HAN_MODEL_History],['LSTM','HAN','HAN-3'],'val_categorical_accuracy','Epochs','Validation Accuracy','Validation_Accuracy')
# plot_model(LSTM_Model, to_file=os.path.join(MODEL_FOLDER,test_case_name+'_Model_LSTM.png'), show_shapes=True, show_layer_names=True)
# plot_model(HAN_MODEL, to_file=os.path.join(MODEL_FOLDER,test_case_name+'_Model_LSTM.png'), show_shapes=True, show_layer_names=True)
han_dataframe = pd.DataFrame(HAN_MODEL_History.history)
lstm_dataframe = pd.DataFrame(LSTM_Model_History.history)
han3_dataframe = pd.DataFrame(HAN3_MODEL_History.history)

LSTM_Model.save(os.path.join(MODEL_FOLDER,test_case_name+'_LSTM.h5'))
HAN_MODEL.save(os.path.join(MODEL_FOLDER,test_case_name+'_HAN.h5'))
HAN3_MODEL.save(os.path.join(MODEL_FOLDER,test_case_name+'_HAN3.h5'))

han3_dataframe.to_csv(os.path.join(MODEL_FOLDER,test_case_name+'_HAN3.csv'))
lstm_dataframe.to_csv(os.path.join(MODEL_FOLDER,test_case_name+'_LSTM.csv'))
han_dataframe.to_csv(os.path.join(MODEL_FOLDER,test_case_name+'_HAN.csv'))
print("#"*20+" Completed Execution "+"#"*20)