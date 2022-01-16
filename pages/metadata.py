# Cargar bibliotecas importantes 
import pandas as pd
import streamlit as st 
import os

def app():
    """Esta aplicación se creó para ayudar al usuario a cambiar los metadatos del archivo cargado. 
    Pueden realizar fusiones. Cambie los nombres de las columnas, etc.  
    """

    # Cargar los datos cargados  
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        data = pd.read_csv('data/main_data.csv')
        st.dataframe(data)

        # Leer los metadatos de la columna para este conjunto de datos
        col_metadata = pd.read_csv('data/metadata/column_type_desc.csv')

        '''Cambiar la información sobre los tipos de columna
            Aquí, la información de los tipos de columna se puede cambiar usando menús desplegables.
            La página está dividida en dos columnas usando columnas beta. 
        '''
        st.markdown("#### Cambiar la información sobre los tipos de columna")
        
        # Utilice la técnica de dos columnas 
        col1, col2 = st.columns(2)

        global name, type
        # Diseño de la columna 1 
        name = col1.selectbox("Seleccionar columna", data.columns)
        
         # Diseño de la columna 2
        current_type = col_metadata[col_metadata['column_name'] == name]['type'].values[0]
        print(current_type)
        column_options = ['numeric', 'categorical','object']
        current_index = column_options.index(current_type)
        
        type = col2.selectbox("Seleccionar tipo de columna", options=column_options, index = current_index)
        
        st.write( """Seleccione el nombre de su columna y el nuevo tipo de los datos.
                    Para enviar todos los cambios, haga clic en * Enviar cambios * """)

        
        if st.button("Cambiar tipo de columna"): 

            # Set the value in the metadata and resave the file 
            # col_metadata = pd.read_csv('data/metadata/column_type_desc.csv')
            st.dataframe(col_metadata[col_metadata['column_name'] == name])
            
            col_metadata.loc[col_metadata['column_name'] == name, 'type'] = type
            col_metadata.to_csv('data/metadata/column_type_desc.csv', index = False)

            st.write("Your changes have been made!")
            st.dataframe(col_metadata[col_metadata['column_name'] == name])