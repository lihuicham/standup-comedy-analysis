import streamlit as st
import pandas as pd 

'''
# Performance Evaluation 

We trained `MultinomialNB`, `Logistic Regression`, `Random Forest`, `Adaboost`, 
`Gradient Boost` and `XGBoost` models and predict them. Below is the performance comparison 
between models. 
'''

model_perf_df = pd.read_pickle('/Users/lihuicham/Desktop/Y2S2/BT4222/project/standup-comedy-analysis/main/model_performance_df')

st.dataframe(model_perf_df)