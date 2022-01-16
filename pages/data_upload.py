import streamlit as st
import numpy as np
import pandas as pd
from pages import utils

# @st.cache
def app():
    st.markdown("## Cargar de datos")

    # Cargue el conjunto de datos y guárdelo como csv
    st.markdown("### Sube un archivo csv para analizarlo") 
    st.write("\n")

    # Código para leer un solo archivo
    uploaded_file = st.file_uploader("Elija un archivo", type = ['csv', 'xlsx'])
    global data
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            data = pd.read_excel(uploaded_file)


    '''Cargue los datos y guarde las columnas con categorías como un marco de datos. 
    Esta sección también permite cambios en las columnas numéricas y categóricas. '''
    if st.button("Cargar datos"):
        
        # Datos brutos
        st.dataframe(data)
        data.to_csv('data/main_data.csv', index=False)
        
        # Recoge las columnas categóricas y numéricas 
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = list(set(list(data.columns)) - set(numeric_cols))
        
        # Guarde las columnas como un marco de datos o diccionario
        columns = []

        # Itere a través de las columnas numéricas y categóricas y guarde en columnas 
        columns = utils.genMetaData(data) 
        
        # Guarde las columnas como un marco de datos con categorías
        # Aquí column_name es el nombre del campo y el tipo es si es numérico o categórico
        columns_df = pd.DataFrame(columns, columns = ['column_name', 'type'])
        columns_df.to_csv('data/metadata/column_type_desc.csv', index = False)

         # Mostrar columnas 
        st.markdown("**Column Name**-**Type**")
        for i in range(columns_df.shape[0]):
            st.write(f"{i+1}. **{columns_df.iloc[i]['column_name']}** - {columns_df.iloc[i]['type']}")
        
        st.markdown( """Los anteriores son los tipos de columna automatizados detectados por la aplicación en los datos.
        En caso de que desee cambiar los tipos de columna, diríjase a la sección ** Cambio de columna **. """)