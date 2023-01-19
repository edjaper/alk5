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
from gsheetsdb import connect


# Create a connection object.
conn = connect()
sheet_url = "https://docs.google.com/spreadsheets/d/11QZjVGnbT3y7enxDc4IWCLcxy2gGpVQAfFoN8r3ytRM/edit#gid=0"
# Perform SQL query on the Google Sheet.
# Uses st.cache to only rerun when the query changes or after 1 sec.
@st.cache(ttl=1)
def run_query(query):
    rows = conn.execute(query, headers=1)
    rows = rows.fetchall()
    return rows



from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


df = st.cache(pd.read_csv)('df_x_SKBfregression_545noADME_withYandYpredandId.csv', sep=',', decimal='.')

    

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Adicionar filtros")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Descritores utilizados para filtrar moléculas", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Valores para {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df



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



def jointplot():    
    import streamlit as st

    st.subheader("Jointplot")  
    variavel_a = st.selectbox( 'Descritor 1', df.columns.to_list()) 
    variavel_b = st.selectbox( 'Descritor 2', df.columns.to_list()) 
    
    fig = px.scatter(df, x = variavel_a, y = variavel_b, marginal_x="histogram", marginal_y="histogram", color = "pIC50", hover_name="id", size="pIC50_predito", size_max=5)
    st.plotly_chart(fig, use_conatiner_width=True, theme=None)
 
def violinplot():    
    import streamlit as st
    st.subheader("Violinplot")  

    y1=list(df.loc[:, 'pIC50'])
    y2=list(df.loc[:, 'pIC50_predito'])
    y1.extend(y2)
    
    x1=df.loc[:, 'id'].to_list()
    x2=[*x1, *x1]

    a= ['real'] * 545
    b= ['predito'] * 545
    c=[*a, *b]    

    dfp = pd.DataFrame(dict(id=x2, pIC50=y1, valor=c))
  
    fig = px.violin(dfp, x="pIC50", y="valor", color="valor", box=True, points="all", title="pIC50 Real x Predição do modelo", labels={"id":"IDentificador da molécula"}, violinmode='overlay', hover_data=dfp.columns)
    #fig = px.scatter(df, x = variavel_a, y = variavel_b, marginal_x="histogram", marginal_y="histogram", color = "pIC50", hover_name="id", size="pIC50_predito", size_max=5)
    st.plotly_chart(fig, use_conatiner_width=True, theme=None)
    
    
    
def dataset():    
    import streamlit as st

    st.subheader("Compare moléculas")
    ids = st.multiselect('Selecionar moléculas', df['id'].unique())
    st.write('Lista:')
    st.write(df[df['id'].isin(ids)])
    
    st.subheader("Todas")
    #st.write(df)
    st.dataframe(filter_dataframe(df))
    

    
    
    
    
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
    fig = px.line(dfp, x="id", y="pIC50", color="valor", markers=True, title="pIC50 Real x Predição do modelo",
                 labels={"id":"IDentificador da molécula"} # customize axis label
                )

    st.plotly_chart(fig, theme='streamlit', use_conatiner_width=True)

    
    
def mapping_demo5():
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
          - **:green[Verde]** = contribui positivamente para o resultado
          - **:red[Vermelho]** = contribui negativamente para o resultado
        
        ### Conjunto de dados

        - Veja o conjunto de dados completo e
        - Selecione uma lista de identificadores de moléculas para comparar        

        ### Jointplot

        - Selecione duas variáveis e compare as suas distribuições no gráfico
        
         ### Violintplot

        - Compare as distribuições de pIC50 real versus predito no gráfico
        
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
        
        rows = run_query(f'SELECT * FROM "{sheet_url}"')
        for row in rows:
            if( int(row.id)==int(i)):
                texto = str(row.feedback)
                if (texto=="nan" or texto=="None"):
                    texto=""
        st.text_area('Clique no botão **Mostrar Explicação** para buscar dados da [planilha de comentários](https://docs.google.com/spreadsheets/d/11QZjVGnbT3y7enxDc4IWCLcxy2gGpVQAfFoN8r3ytRM/edit#gid=0) novamente', value=texto)     
            
 
    
    


page_names_to_funcs = {
    "Sobre": intro,
    "Visão geral": plotting,
    "Ver explicação": lime,
    "Conjunto de dados": dataset,    
    "Jointplot": jointplot,
    "Violinplot": violinplot 
    #"Temp": mapping_demo2
}

demo_name = st.sidebar.selectbox('Menu', page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()




   


