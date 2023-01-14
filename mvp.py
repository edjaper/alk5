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

def intro():
    import streamlit as st

    st.write("# LIME QSAR 🔍")
    st.sidebar.success("Navegue aqui.")

    st.markdown(
        """
        Vamos inspecionar o modelo opaco através de uma aproximação linear.

        **👈 Navegue pelo menu à esquerda**

        ### Ver explicação

        - Escolha o identificador da molécula
        - Analise a explicação gerada pelo LIME
          - Verde = contribui positivamente para o resultado
          - Vermelho = contribui negativamente para o resultado

        ### Mapa

        - Visão geral da previsão do modelo versus valor real
    """
    )
    
def plotting():
    import streamlit as st
    import time
    import numpy as np

    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """
        O gráfico mostra o pIC50 real versus o predito pelo modelo. Você pode anotar os identificadores de moléculas que deseja inspecionar no menu **Ver explicação**
"""
    )

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    chart = st.line_chart(last_rows)

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % i)
        chart.add_rows(new_rows)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")

    
    
def lime():
    import streamlit as st

    st.image('handLens.png')
    st.subheader("Explicação via LIME")
    i = st.number_input('ID da molécula:', min_value=0, max_value=544, value=0)

    if st.button('Mostrar Explicacao'):
        explanation = explainer1.explain_instance(df_x.iloc[i,:], estimator.predict, num_features=10)
        st.pyplot(explanation.as_pyplot_figure())    
    
    
    
page_names_to_funcs = {
    "Sobre": intro,
    #"Visão Geral": plotting,
    #'Mapping Demo': mapping_demo,
    "Ver explicação": lime
}

demo_name = st.sidebar.selectbox('Choose a demo', page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()




   


