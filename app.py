import os
import streamlit as st
import numpy as np
from PIL import  Image

# Importaciones personalizadas 
from multipage import MultiPage
from pages import data_upload, machine_learning, metadata, data_visualize, redundant, eda# import your pages here

# Crea una instancia de la aplicación
app = MultiPage()

# Título de la página principal
display = Image.open('Logo.png')
display = np.array(display)
# st.image(display, width = 400)
# st.title("Data Storyteller Application")
col1, col2 = st.columns(2)
col1.image(display, width = 300)
col2.title("CAFFII DATA")

# Agrega toda tu aplicación aquí
app.add_page("Cargar datos", data_upload.app)
app.add_page("Cambio de metadatos", metadata.app)
app.add_page("Aprendizaje automático", machine_learning.app)
app.add_page("Análisis de datos",data_visualize.app)
app.add_page("Optimización Y-Parameter",redundant.app)
app.add_page("EDA",eda.app)

# La aplicación principal
app.run()
