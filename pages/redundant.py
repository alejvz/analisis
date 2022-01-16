import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
import os

def app():
	
	if 'main_data.csv' not in os.listdir('data'):
		st.markdown("¡Cargue datos a través de la página `Cargar datos`! ")
	else:
		df = pd.read_csv('data/main_data.csv')
		st.markdown("### Una pequeña demostración para mostrar columnas redundantes de un csv")

		redCols = utils.getRedundentColumns
		corr = df.corr(method='pearson')
		y_var = st.radio("Seleccione la variable a predecir (y)", options=corr.columns)
		th = st.slider("Umbral", min_value=0.05, max_value=0.95, value=0.25, step=0.01, format='%f')#, key=None, help=None)
		# st.write(df.col)
		redundantCols = utils.getRedundentColumns(corr, y_var, th)
		newDF = utils.newDF(df, redundantCols)
		# st.write("Redundant Columns:", redundantCols)
		st.write("Número de columnas eliminadas:",len(redundantCols))
		st.write("Datos nuevos: \n", newDF.head())

	