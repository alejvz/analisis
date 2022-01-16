# Importar bibliotecas necesarias
import json
import joblib

import pandas as pd
import streamlit as st
import numpy as np

# Machine Learning 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Clases personalizadas 
from .utils import isNumerical
import os

def app():
    """Esta aplicación ayuda a ejecutar modelos de aprendizaje automático sin tener que escribir código explícito 
    por el usuario. Ejecuta algunos modelos básicos y permite al usuario seleccionar las variables X e y.
    """
    
    # Cargar los datos 
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        data = pd.read_csv('data/main_data.csv')

        # Crea el diccionario de parámetros del modelo
        params = {}

        # Utilice la técnica de dos columnas 
        col1, col2 = st.columns(2)

         # Diseño de la columna 1 
        y_var = col1.radio("seleccionar varible a predecir (y)", options=data.columns)

        # Columna de diseño 2 
        X_var = col2.multiselect("Select the variables to be used for prediction (X)", options=data.columns)

        # Comprueba si la longitud de x no es cero 
        if len(X_var) == 0:
            st.error("Tienes que introducir alguna variable X y no se puede dejar vacía. ")

        # Check if y not in X 
        if y_var in X_var:
            st.error("¡Advertencia! La variable Y no puede estar presente en su variable X. ")

        # Opción para seleccionar el tipo de condición previa
        pred_type = st.radio("Seleccione el tipo de proceso que desea ejecutar. ", 
                            options=["Regression", "Classification"], 
                            help="Escribir sobre registro y clasificación")

        # Agregar a los parámetros del modelo
        params = {
                'X': X_var,
                'y': y_var, 
                'pred_type': pred_type,
        }

        # if botón st. ("Ejecutar modelos")

        st.write(f"**Variable a predecir:** {y_var}")
        st.write(f"**Variable que se utilizará para la predicción:** {X_var}")
        
        # Divida los datos en pruebas y conjuntos de entrenamiento
        X = data[X_var]
        y = data[y_var]

        # Perform data imputation 
        # st.write("THIS IS WHERE DATA IMPUTATION WILL HAPPEN")
        
         # Realizar codificación
        X = pd.get_dummies(X)

         # Compruebe si y necesita codificarse
        if not isNumerical(y):
            le = LabelEncoder()
            y = le.fit_transform(y)
            
            # Imprime todas las clases 
            st.write("Las clases y la clase que se les asigna es la siguiente: -")
            classes = list(le.classes_)
            for i in range(len(classes)):
                st.write(f"{classes[i]} --> {i}")
        

        # Realizar divisiones de prueba de entrenaminto
        st.markdown("#### División de prueba de entrenaminto")
        size = st.slider("Porcentaje de división de valor",
                            min_value=0.1, 
                            max_value=0.9, 
                            step = 0.1, 
                            value=0.8, 
                            help="Este es el valor que se usará para dividir los datos para entrenamiento y prueba. Predeterminado = 80%")

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)
        st.write("Número de muestras de entrenamiento:", X_train.shape[0])
        st.write("Número de muestras de prueba:", X_test.shape[0])

        # Guarde los parámetros del modelo como un archivo json
        with open('data/metadata/model_params.json', 'w') as json_file:
            json.dump(params, json_file)

        '''FUNCIONAMIENTO DE LOS MODELOS DE APRENDIZAJE MÁQUINA'''
        if pred_type == "Regression":
            st.write("Ejecución de modelos de regresión en la muestra")

            # Tabla para almacenar modelo y precisión 
            model_r2 = []

            # Modelo de regresión lineal  
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            lr_r2 = lr_model.score(X_test, y_test)
            model_r2.append(['Linear Regression', lr_r2])

            x_in = np.array([1,7.587])
            MedV = lr_model.predict(x_in.reshape((1, X_test.shape[1])))
            st.write("Número prediccion", MedV)

            # Modelo de árbol de decisión
            dt_model = DecisionTreeRegressor()
            dt_model.fit(X_train, y_train)
            dt_r2 = dt_model.score(X_test, y_test)
            model_r2.append(['Decision Tree Regression', dt_r2])

            # Guarda uno de los modelos 
            if dt_r2 > lr_r2:
                 # guardar árbol de decisiones 
                joblib.dump(dt_model, 'data/metadata/model_reg.sav')
            else: 
                joblib.dump(lr_model, 'data/metadata/model_reg.sav')

            # Hacer un marco de datos de resultados
            results = pd.DataFrame(model_r2, columns=['Models', 'R2 Score']).sort_values(by='R2 Score', ascending=False)
            st.dataframe(results)
        
        if pred_type == "Classification":
            st.write("Ejecución de modelos de clasificación en muestra")

            # Tabla para almacenar modelo y precisión 
            model_acc = []

            # Linear regression model 
            lc_model = LogisticRegression()
            lc_model.fit(X_train, y_train)
            lc_acc = lc_model.score(X_test, y_test)
            model_acc.append(['Linear Regression', lc_acc])

            # Decision Tree model 
            dtc_model = DecisionTreeClassifier()
            dtc_model.fit(X_train, y_train)
            dtc_acc = dtc_model.score(X_test, y_test)
            model_acc.append(['Decision Tree Regression', dtc_acc])

            # Guarda uno de los modelos  
            if dtc_acc > lc_acc:
                # guardar árbol de decisiones
                joblib.dump(dtc_model, 'data/metadata/model_classification.sav')
            else: 
                joblib.dump(lc_model, 'data/metadata/model_classificaton.sav')

            # Hacer un marco de datos de resultados 
            results = pd.DataFrame(model_acc, columns=['Models', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
            st.dataframe(results)