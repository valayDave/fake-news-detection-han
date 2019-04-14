import pandas as pd
import train as model
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import os.path
import datetime
my_path = os.path.abspath(os.path.dirname(__file__))
PLOT_FOLDER = os.path.join(my_path, 'plots/')
MODEL_FOLDER = os.path.join(my_path, 'models/')

test_case_name = 'HAN_vs_LSTM_Test_case_1'

dataset_path = 'datasets/first_split.csv'

def plot_models(model_arr,model_name_arr,model_key,xlabel,ylabel,plot_name):
    fig1 = plt.figure()
    for model in model_arr:
        plt.plot(model.history['model_key'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(model_name_arr, loc='upper left')
    fig1.savefig(os.path.join(PLOT_FOLDER,test_case_name+'_'+plot_name+'.png'))

df = pd.read_csv(dataset_path)
#model.train(df,'first_split_dataset_han')
LSTM_Model = model.train_lstm(df,test_case_name)

HAN_MODEL = model.train_han(df,test_case_name)

plot_models([LSTM_Model,HAN_MODEL],['LSTM','HAN'],'val_loss','Epochs','Validation Loss','Validation_Loss')
plot_models([LSTM_Model,HAN_MODEL],['LSTM','HAN'],'acc','Epochs','Accuracy','Accuracy')
plot_models([LSTM_Model,HAN_MODEL],['LSTM','HAN'],'loss','Epochs','Loss','Loss')
plot_models([LSTM_Model,HAN_MODEL],['LSTM','HAN'],'val_acc','Epochs','Validation Accuracy','Validation_Accuracy')
# plot_model(LSTM_Model, to_file=os.path.join(MODEL_FOLDER,test_case_name+'_Model_LSTM.png'), show_shapes=True, show_layer_names=True)
# plot_model(HAN_MODEL, to_file=os.path.join(MODEL_FOLDER,test_case_name+'_Model_LSTM.png'), show_shapes=True, show_layer_names=True)

LSTM_Model.save(os.path.join(MODEL_FOLDER,test_case_name+'_LSTM.h5'))
HAN_MODEL.save(os.path.join(MODEL_FOLDER,test_case_name+'_HAN.h5'))