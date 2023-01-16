from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pandas as pd
import lime
import lime.lime_tabular
import numpy as np
import torch
import sklearn_json as skljson
import plotly.express as px
import plotly.figure_factory as ff

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



    
def dataset():    
    import streamlit as st

    st.subheader("Moléculas")
    
    # write dataframe to screen
    st.write(df)
    

    
    
    
    
def plotting():
    import streamlit as st
    import time
    import numpy as np

    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """
        O gráfico mostra o pIC50 real versus o pIC50 predito pelo modelo. Você pode anotar os identificadores de moléculas que deseja inspecionar no menu **Ver explicação**
"""
    )

    import plotly.express as px
    import pandas as pd

    y1=list(df.loc[:, 'pIC50'])
    y2=list(df.loc[:, 'pIC50_predito'])
    y1.extend(y2)
    
    x1=df.loc[:, 'id'].to_list()
    n_x = len(x1)
    x2=[*x1, *x1]

    a= ['real'] * 545
    b= ['predito'] * 545
    c=[*a, *b]    

    dfp = pd.DataFrame(dict(id=x2, pIC50=y1, valor=c))

    # Use column names of df for the different parameters x, y, color, ...
    fig = px.line(dfp, x="id", y="pIC50", color="valor", title="pIC50 Real x Predição do modelo",
                 labels={"id":"IDentificador da molécula"} # customize axis label
                )

    st.plotly_chart(fig, theme='streamlit', use_conatiner_width=True)

    
    
def mapping_demo():
    import streamlit as st
    import time
    import numpy as np

    st.markdown(f'# {list(page_names_to_funcs.keys())[2]}')
    st.write(
        """
        O gráfico mostra o pIC50 real versus o pIC50 predito pelo modelo. Você pode anotar os identificadores de moléculas que deseja inspecionar no menu **Ver explicação**
"""
    )

    import plotly.express as px
    import pandas as pd

    schools = ["Brown", "NYU", "Notre Dame", "Cornell", "Tufts", "Yale",
           "Dartmouth", "Chicago", "Columbia", "Duke", "Georgetown",
           "Princeton", "U.Penn", "Stanford", "MIT", "Harvard"]
    n_schools = len(schools)

    women_salary = [72, 67, 73, 80, 76, 79, 84, 78, 86, 93, 94, 90, 92, 96, 94, 112]
    men_salary = [92, 94, 100, 107, 112, 114, 114, 118, 119, 124, 131, 137, 141, 151, 152, 165]

    dfw = pd.DataFrame(dict(school=schools*2, salary=men_salary + women_salary,
                       gender=["Men"]*n_schools + ["Women"]*n_schools))

    # Use column names of df for the different parameters x, y, color, ...
    fig = px.scatter(dfw, x="salary", y="school", color="gender", title="Gender Earnings Disparity",labels={"salary":"Annual Salary (in thousands)"}) # customize axis label
    st.plotly_chart(fig, theme='streamlit', use_conatiner_width=True)

    
    
def mapping_demo_1():
    import streamlit as st
    import time
    import numpy as np

    st.markdown(f'# {list(page_names_to_funcs.keys())[2]}')
    st.write(
        """
        O gráfico mostra o pIC50 real versus o pIC50 predito pelo modelo. Você pode anotar os identificadores de moléculas que deseja inspecionar no menu **Ver explicação**
"""
    )

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)
    last_rows1 = np.array(df.loc[0, 'pIC50'])
    last_rows2 = np.array(df.loc[0, 'pIC50_predito'])
    chart = st.line_chart(last_rows1)
    #chart.add_rows(last_rows1)
    chart.add_rows(last_rows2)
    t = 545

    last_rows1 = np.array(df.loc[1, 'pIC50'])
    last_rows2 = np.array(df.loc[1, 'pIC50_predito'])
    chart.add_rows(last_rows1)
    chart.add_rows(last_rows2)
    
    for i in range(1, t):
        new_rows1 = last_rows1 + np.array(df.loc[i, 'pIC50'])
        new_rows2 = last_rows2 + np.array(df.loc[i, 'pIC50_predito'])
        p = int(np.round(((i+1)/t)*100,2))
        status_text.text("%i Moléculas" % i)
        chart.add_rows(new_rows1)
        chart.add_rows(new_rows2)
        progress_bar.progress(p)
        last_rows1 = new_rows1
        last_rows2 = new_rows2
        time.sleep(0.005)

    progress_bar.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Ver de novo")


    
    
def intro():
    import streamlit as st

    st.write("# LIME QSAR 🔍")
    st.sidebar.success("Navegue aqui.")

    st.markdown(
        """
        Vamos inspecionar o modelo opaco através de uma aproximação linear.

        **👈 Navegue pelo menu à esquerda**

        ### Visão geral

        - Visão geral da previsão do modelo versus valor real
        
        ### Ver explicação

        - Escolha o identificador da molécula
        - Analise a explicação gerada pelo LIME
          - Verde = contribui positivamente para o resultado
          - Vermelho = contribui negativamente para o resultado


    """
    )
    
def mapping_demo2():
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
    last_rows1 = np.random.randn(1, 1)
    chart = st.line_chart(last_rows)

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(3, 1).cumsum(axis=0)
        new_rows1 = last_rows1[-1, :] + np.random.randn(3, 2).cumsum(axis=0)
        status_text.text("%i%% Completo" % i)
        chart.add_rows(new_rows)
        chart.add_rows(new_rows1)
        progress_bar.progress(i)
        last_rows = new_rows
        last_rows1 = new_rows1
        time.sleep(0.05)

    progress_bar.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Ver de novo")
    
    
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
    "Visão geral": plotting,
    "Ver explicação": lime,
    "Mapping Demo": mapping_demo,    
    "Temp": mapping_demo2,
    "Conjunto de dados": dataset
}

demo_name = st.sidebar.selectbox('Menu', page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()




   


