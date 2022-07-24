# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 22:39:45 2022

@author: LENOVO
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline,make_pipeline
import pickle
import warnings
warnings.filterwarnings("ignore")

pipe = pickle.load(open('pipe.pkl','rb'))


st.title('FRAUD_DETECTION_APP')
step= st.number_input("STEP")
type= st.selectbox('TYPE_OF_TRANSACTION',("CASH_OUT","PAYMENT","CASH_IN","TRANSFER","DEBIT"))
amount=st.number_input("Amount of transaction")
newbalanceOrig=st.number_input("Enter new Balance of origin")
newbalanceDest=st.number_input("Enter new Balance of destination")


def prediction(step,type,amount,newbalanceOrig,newbalanceDest):
    test_input2 = np.array([step,type,amount,newbalanceOrig,newbalanceDest],dtype=object).reshape(1,5)
    result=pipe.predict(test_input2)
    if result==1:
        pred='Fraud'
    else:
        pred='Not Fraud'
    return pred
        
if st.button('predict'):
    result=prediction(step,type,amount,newbalanceOrig,newbalanceDest)
    st.success('Transaction is {}'.format(result))


