import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title('Aplicación de Análisis de Datos')

# Lectura de datos
st.header('Lectura de Datos')
uploaded_file = st.file_uploader('Sube tu archivo CSV', type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    # Resumen de Datos
    st.header('Resumen de Datos')
    st.subheader('Primeras filas del dataframe')
    st.write(df.head())

    st.subheader('Estadísticas descriptivas')
    st.write(df.describe())

    # Visualización de Datos
    st.header('Visualización de Datos')

    # Histograma
    st.subheader('Histograma')
    column = st.selectbox('Selecciona la columna para el histograma', df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, ax=ax)
    st.pyplot(fig)

    # Gráfico de dispersión
    st.subheader('Gráfico de Dispersión')
    x_col = st.selectbox('Selecciona la columna para el eje X', df.columns)
    y_col = st.selectbox('Selecciona la columna para el eje Y', df.columns)
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
    st.pyplot(fig)

    # Técnica Estadística: Regresión Lineal Simple
    st.header('Técnica Estadística: Regresión Lineal Simple')
    st.subheader('Selecciona las columnas para la regresión')
    x_reg_col = st.selectbox('Selecciona la columna para el predictor (X)', df.columns, key='x_reg')
    y_reg_col = st.selectbox('Selecciona la columna para la variable respuesta (Y)', df.columns, key='y_reg')

    if st.button('Calcular Regresión'):
        X = df[[x_reg_col]].values
        Y = df[y_reg_col].values

        model = LinearRegression()
        model.fit(X, Y)

        Y_pred = model.predict(X)

        st.subheader('Resultados de la Regresión')
        st.write('Intercepto:', model.intercept_)
        st.write('Coeficiente:', model.coef_[0])

        fig, ax = plt.subplots()
        ax.scatter(X, Y, color='blue')
        ax.plot(X, Y_pred, color='red')
        st.pyplot(fig)
else:
    st.write("Sube un archivo CSV para comenzar.")