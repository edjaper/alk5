from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pandas as pd
import lime
import lime.lime_tabular
import numpy as np
import torch
import sklearn_json as skljson

df = pd.read_csv("df_x_SKBfregression_545noADME_withYandYpredandId.csv", sep=";", decimal=".")

#feedback = pd.read_csv("feedback.csv", sep=";")
#if "feedback" not in st.session_state:
#    st.session_state['feedback'] = pd.DataFrame(columns=['id','feedback'])
    


#Loading up the Regression model we created
#model = Ridge()
#model = skljson.from_json('rr_model.json')

model_save_name = 'rf_model_mvp230113.pt'
#path = F"/content/drive/MyDrive/BACKUPs/USP/data/noDuplicates/{model_save_name}"
#model=(torch.load(path))
model=(torch.load(model_save_name))



estimator = model
df_x = df.iloc[:, 0:40]
x_featurenames = df_x.columns

explainer1 = lime.lime_tabular.LimeTabularExplainer(np.array(df_x),
                    feature_names=x_featurenames, 
                    # class_names=['cnt'], 
                    # categorical_features=, 
                    # There is no categorical features in this example, otherwise specify them.                               
                    verbose=False, mode='regression')



#Caching the model for faster loading
@st.cache

st.image("handLens.PNG")
st.subheader('Explanation setup:')

i = st.number_input('ID Instancia:', min_value=0, max_value=544, value=0)


if st.button('Mostrar Explicação'):
    explanation = explainer1.explain_instance(df_x.iloc[i,:], estimator.predict, num_features=10)
    st.pyplot(explanation.as_pyplot_figure())
