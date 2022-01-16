import numpy as np
import pandas as pd
import streamlit as st
import os
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Web App Title


def app(): 

# Web App Title
    st.markdown('''
# **APLICACION EDA**

Construiremos el futuro y lo haremos mejor.

---
''')


    # Pandas Profiling Report
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("¡Cargue datos a través de la página `Upload Data` !")
    else:
            #st.info('Awaiting for CSV file to be uploaded.')
            #@st.cache
            def load_csv():
                csv = pd.read_csv('data/main_data.csv')
                return csv
            df = load_csv()
            pr = ProfileReport(df, explorative=True)
            st.header('**Input DataFrame**')
            st.write(df)
            st.write('---')
            st.header('**Pandas Profiling Report**')
            st_profile_report(pr)
    
