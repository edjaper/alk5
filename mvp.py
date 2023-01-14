from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pandas as pd
import lime
import lime.lime_tabular
import numpy as np
import torch
import sklearn_json as skljson

df = st.cache(pd.read_csv)('df_x_SKBfregression_545noADME_withYandYpredandId.csv', sep=',', decimal='.')

#feedback = pd.read_csv("feedback.csv", sep=";")
#if "feedback" not in st.session_state:
#    st.session_state['feedback'] = pd.DataFrame(columns=['id','feedback'])
    


#Loading up the Regression model we created
#model = Ridge()
#model = skljson.from_json('rr_model.json')

model_save_name = 'rf_model_mvp230113.pt'
path = F'./{model_save_name}'
model=(torch.load(path))


estimator = model
df_x = df.iloc[:, 0:40]
x_featurenames = df_x.columns


explainer1 = lime.lime_tabular.LimeTabularExplainer(np.array(df_x),feature_names=x_featurenames, verbose=False, mode='regression')


#Caching the model for faster loading
#@st.cache

st.image('handLens.png')
st.subheader('Explanation setup:')

i = st.number_input('ID Instancia:', min_value=0, max_value=544, value=0)


if st.button('Mostrar Explicacao'):
    explanation = explainer1.explain_instance(df_x.iloc[i,:], estimator.predict, num_features=10)
    st.pyplot(explanation.as_pyplot_figure())

   

page_names_to_funcs = {
    "—": intro,
    "Plotting Demo": plotting_demo,
    "Mapping Demo": mapping_demo,
    "DataFrame Demo": data_frame_demo
}

demo_name = st.sidebar.selectbox("Visualizar", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()    
    
