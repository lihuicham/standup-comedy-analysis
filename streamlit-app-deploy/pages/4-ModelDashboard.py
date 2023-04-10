import streamlit as st
import pandas as pd 

'''
# Performance Evaluation 

We trained `MultinomialNB`, `Logistic Regression`, `Random Forest`, `Adaboost`, 
`Gradient Boost` and `XGBoost` models and predict them. Below is the performance comparison 
between models. 
'''
@st.cache
def load_model() :
    return pd.read_pickle('main/model_performance_df')

model_perf_df = load_model()
st.dataframe(model_perf_df)

'''
## ROC Curves
'''
@st.cache
def load_nb() :
    return pd.read_pickle('main/nb_roc_fig')

@st.cache
def load_logreg() :
    return pd.read_pickle('main/logreg_roc_fig')

@st.cache
def load_rf() :
    return pd.read_pickle('main/rf_roc_fig')

@st.cache
def load_ada() :
    return pd.read_pickle('main/ada_roc_fig')

@st.cache
def load_grad() :
    return pd.read_pickle('main/grad_roc_fig')

@st.cache
def load_xgb() :
    return pd.read_pickle('main/xgb_roc_fig')

@st.cache
def load_vote() :
    return pd.read_pickle('main/vote_roc_fig')

nb_roc_fig = load_nb()
logreg_roc_fig = load_logreg()
rf_roc_fig = load_rf()
ada_roc_fig = load_ada()
grad_roc_fig = load_grad()
xgb_roc_fig = load_xgb()
vote_roc_fig = load_vote()


option = st.selectbox(
    'Area Under Curve (AUC) of Different Models\' ROC',
    ('Multinomial Naive Bayes Model', 
     'Logistic Regression Model',
     'Random Forest Model',
     'Adaboost Model',
     'Gradient Boost Model',
     'XG Boost Model',
     'Voting Model (Ensemble Method)'))

option_dict = {
    'Multinomial Naive Bayes Model': nb_roc_fig,
    'Logistic Regression Model' : logreg_roc_fig,
    'Random Forest Model' : rf_roc_fig,
    'Adaboost Model' : ada_roc_fig,
    'Gradient Boost Model' : grad_roc_fig,
    'XG Boost Model' : xgb_roc_fig,
    'Voting Model (Ensemble Method)' : vote_roc_fig
}

st.plotly_chart(option_dict.get(option))