import streamlit as st
import numpy as np
import pandas as pd

from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import *
import xgboost as xgb
from catboost import Pool, CatBoostRegressor
from utils import lottie_ion
# from streamlit_lottie import st_lottie
from sklearn.model_selection import KFold
import joblib

import functools

from preprocessing import X_test, load_data


# st.set_page_config(layout='wide')

################################################################################

# st_lottie(lottie_ion, height=200)


st.title('Liverpool Ion Channel')

st.markdown('Disclaimer: This is a project by Low Kheng Oon. This application is not production ready. Use at your own discretion')

st.header('Background')

st.subheader('Electrophysiology')

st.write('Electrophysiology is a branch of physiology where the electrical properties of biological cells and tissues are studied. It involves the measurement of voltage or current changes on a wide variety of scales from single ion channel proteins to whole organs like the heart. In neuroscience, it includes the measurement of electrical activity in neurons, and, in particular, action potential activity. These measured signals are then used to perform medical diagnoses and analyses on humans. Below is an example of an elecrophysiological signal:')

st.image('intro.gif')

st.write('From the above graph, it can be seen that the voltage is being plotted against time. These voltages are measured from human organs using special methods and apparatus. One such method is called the clamp method.')

st.subheader('Clamp Method')

st.write('The clamp method is one of many methods to measure voltage and current changes in human organs. Below is a video that explains how the clamp method works. The relevant section ends at 2:55.')

st.video('https://youtu.be/CvfXjGVNbmw')

st.write('The clamp method uses a pair of electrodes, an amplifier, and a signal generator to measure voltage and current changes in organs. A simple diagram of the setup is given below:')

st.image('intro2.png')

st.write('The electrodes in the aobve diagram measure the electrical impulse from the axon (a part of the neuron). These electric signals are then amplified by the potential amplifier because these electric signals are very small in amplitude. Without amplification, these signals would go undetected. These signals are fianlly displayed on a screen using a signal generator and a montior (display screen).')

st.subheader('Ion channels')

st.write('Now since we understand how voltage and current signals are measured and recorded, let us understand the meaning of "ion channels".')

st.write('Ion channels are pore-forming membrane proteins that allow ions to pass through the channel pore. Ion channels are "closed" when they do not allow ions to pass through and "open" when they allow ions to pass through. Ion channels are especially prominent components of the nervous system. In addition, they are key components in a wide variety of biological processes that involve rapid changes in cells, such as cardiac, skeletal, and smooth muscle contraction, epithelial transport of nutrients and ions, T-cell activation and pancreatic beta-cell insulin release.')

st.write('The pivotal role of ion channels in several biological processes makes it an excellent way to discover new drugs and medicines for various diseases.')

st.write('Therefore, finding a relationship between current signals and open ion channels can unlock new possibilities in the fields of medicine and environmental studies. And hence this competition.')

st.header('Signal data')

st.subheader('Signal data vs Time')

load_data('1IAFQ8OjZCnrIrOJHicslYHmd_gR1rsp0', 'train.csv')
load_data('1kfSPuQM40P8o1m7jrPuvcT74IRYof7lY', 'test.csv')

train_data = pd.read_csv('data/train.csv')
st.dataframe(train_data.head())

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
ax1.plot(train_data['time'], train_data['signal'], color='b')
ax1.set_title('Signal data')
ax1.set_xlabel('Time')
ax1.set_ylabel('Signal')
ax2.plot(train_data['time'], train_data['open_channels'], color='r')
ax2.set_title('Channel data')
ax2.set_xlabel('Time')
ax2.set_ylabel('Opened Channels')

st.pyplot(fig)

def get_prediction(test, model, model_type):
    if model_type == 'xgb':
        y_pred = model.predict(xgb.DMatrix(test))
    else:
        y_pred = model.predict(test, num_iteration=model.best_iteration)
    y_pred = np.round(np.clip(y_pred, 0, 10)).astype(int)
    return y_pred

testcsv = pd.read_csv('data/test.csv')
st.write('This is the test file: ')
st.dataframe(testcsv)

input_test = X_test

if st.button('Run Prediction'):

    preds = []
    for fold in range(0,5):
        model_lgb = joblib.load(f'models/lgb_v{fold}.sav')
        # model_xgb = joblib.load(f'models/xgb_v{fold}.sav')
        preds.append(get_prediction(input_test, model_lgb, 'lgb'))
        # preds.append(get_prediction(input, model_xgb, 'xgb'))
    preds = np.average(preds, axis=0)

    sub = pd.read_csv('sample_submission.csv', dtype={'time': np.float32})
    sub['open_channels'] = np.array(np.round(preds,0), np.int)
    st.write(f'Predicted Channels opened: ')
    st.dataframe(sub) 

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.plot(testcsv['time'], testcsv['signal'], color='b')
    ax1.set_title('Signal data')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Signal')
    ax2.plot(sub['time'], sub['open_channels'], color='r')
    ax2.set_title('Channel data')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Opened Channels')

    st.pyplot(fig)


    
            
