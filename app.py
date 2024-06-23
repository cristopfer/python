import os
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
from datetime import datetime
from funciones.datosProcesados import cargarDatos

app = Flask(__name__)
app.static_folder = 'static'

file_path = 'dataframe/finanzas.csv'  # Reemplaza con la ruta a tu archivo CSV

@app.route('/')
def home():
   df2 = pd.read_csv(file_path)
   # BVN
   corr_matrix = df2.tail()
   corr_html = corr_matrix.to_html(classes='data', header="true")
   soup = BeautifulSoup(corr_html, 'html.parser')  
   data = soup.prettify()

   return render_template('RBF.html', table1=data)

@app.route('/BA.html')
def home1():
   df2 = pd.read_csv(file_path)
   # BVN
   corr_matrix = df2.tail()
   corr_html = corr_matrix.to_html(classes='data', header="true")
   soup = BeautifulSoup(corr_html, 'html.parser')  
   data = soup.prettify()

   return render_template('BA.html', table1=data)

@app.route('/SVM.html')
def home2():
   df2 = pd.read_csv(file_path)
   # BVN
   corr_matrix = df2.tail()
   corr_html = corr_matrix.to_html(classes='data', header="true")
   soup = BeautifulSoup(corr_html, 'html.parser')  
   data = soup.prettify()

   return render_template('SVM.html', table1=data)

@app.route('/KNN.html')
def home3():
   df2 = pd.read_csv(file_path)
   # BVN
   corr_matrix = df2.tail()
   corr_html = corr_matrix.to_html(classes='data', header="true")
   soup = BeautifulSoup(corr_html, 'html.parser')  
   data = soup.prettify()

   return render_template('KNN.html', table1=data)

@app.route('/predict', methods=['POST'])
def predict():
   fecha_seleccionada = request.form['fecha']
   fecha_futura = datetime.strptime(fecha_seleccionada, '%Y-%m-%d')
   fecha_seleccionada = pd.to_datetime(fecha_seleccionada)
   df2 = pd.read_csv(file_path)
   df2.index = pd.to_datetime(df2.index)
   df2 = df2.loc[df2.index < fecha_seleccionada]
   featuresBVN1 = ['High_BVN', 'Low_BVN', 'Adj Close_BVN','Open_GLD',
       'High_GLD', 'Low_GLD', 'Adj Close_GLD', 'Open_GCF', 'High_GCF',
       'Low_GCF', 'Adj Close_GCF', 'Open_GSPC', 'High_GSPC',
       'Low_GSPC', 'Close_GSPC', 'Open_PEN_X', 'High_PEN_X', 'Low_PEN_X', 'Close_PEN_X','Year_df']
   targetBVN1 = 'Open_BVN'
   X1 = df2[featuresBVN1].iloc[1:]
   Series_Temporal = df2[targetBVN1].shift(-1)
   y1 = Series_Temporal.iloc[1:]
   mask = ~y1.isna()
   X1 = X1[mask]
   y1 = y1[mask]
   XBVN_train, XBVN_test, yBVN_train, yBVN_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
   rbf_feature = RBFSampler(gamma=1, random_state=42, n_components=100)
   modelRBF = make_pipeline(MinMaxScaler(feature_range=(0, 1)), rbf_feature, Ridge(alpha=1.0))
   modelRBF.fit(XBVN_train, yBVN_train)
   yBVN_pred = modelRBF.predict(XBVN_test)
   ultima_fila = df2[featuresBVN1].iloc[-1]
   nueva_entrada = ultima_fila.copy()
   print(df2[featuresBVN1])
   #nueva_entrada['Year_df'] = fecha_futura.year

   nueva_entrada_df = pd.DataFrame([nueva_entrada])
   nueva_entrada_scaled = modelRBF.named_steps['minmaxscaler'].transform(nueva_entrada_df)
   nueva_entrada_rbf = modelRBF.named_steps['rbfsampler'].transform(nueva_entrada_scaled)
   prediccion = modelRBF.named_steps['ridge'].predict(nueva_entrada_rbf)
   return jsonify({'prediccion': prediccion[0]})

@app.route('/predict1', methods=['POST'])
def predict1():
   fecha_seleccionada = request.form['fecha']
   fecha_futura = datetime.strptime(fecha_seleccionada, '%Y-%m-%d')
   df2 = pd.read_csv(file_path)
   featuresBHP1 = ['High_T03_BHP', 'Low_T03_BHP', 'Adj Close_T03_BHP','Open_SIF','High_SIF', 'Low_SIF',
       'Adj Close_SIF', 'Open_HGF', 'High_HGF', 'Low_HGF', 'Adj Close_HGF','Open_GSPC', 'High_GSPC',
       'Low_GSPC', 'Close_GSPC', 'Open_DJI', 'High_DJI', 'Low_DJI',
       'Close_DJI','Year_df']
   targetBHP1 = 'Open_T03_BHP'
   X1 = df2[featuresBHP1].iloc[1:]
   y1 = df2[targetBHP1].iloc[1:]
   XBHP_train, XBHP_test, yBHP_train, yBHP_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
   rbf_feature = RBFSampler(gamma=1, random_state=42, n_components=100)
   modelRBF = make_pipeline(MinMaxScaler(feature_range=(0, 1)), rbf_feature, Ridge(alpha=1.0))
   modelRBF.fit(XBHP_train, yBHP_train)
   yBHP_pred = modelRBF.predict(XBHP_test)
   ultima_fila = df2[featuresBHP1].iloc[-1]
   nueva_entrada = ultima_fila.copy()
   nueva_entrada['Year_df'] = fecha_futura.year

   nueva_entrada_df = pd.DataFrame([nueva_entrada], index=[fecha_futura])
   nueva_entrada_scaled = modelRBF.named_steps['minmaxscaler'].transform(nueva_entrada_df)
   nueva_entrada_rbf = modelRBF.named_steps['rbfsampler'].transform(nueva_entrada_scaled)
   prediccion = modelRBF.named_steps['ridge'].predict(nueva_entrada_rbf)
   return jsonify({'fecha': fecha_seleccionada, 'prediccion': prediccion[0]})

@app.route('/predict2', methods=['POST'])
def predict2():
   fecha_seleccionada = request.form['fecha']
   fecha_futura = datetime.strptime(fecha_seleccionada, '%Y-%m-%d')
   df2 = pd.read_csv(file_path)
   featuresFSM = ['High_T05_SCCO', 'Low_T05_SCCO', 'Adj Close_T05_SCCO','Open_SIF','High_SIF', 'Low_SIF',
   'Adj Close_SIF', 'Open_HGF', 'High_HGF', 'Low_HGF', 'Adj Close_HGF','Open_GSPC', 'High_GSPC',
   'Low_GSPC', 'Close_GSPC', 'Open_DJI', 'High_DJI', 'Low_DJI',
   'Close_DJI','Year_df']
   targetFSM = 'Open_T05_SCCO'
   X1 = df2[featuresFSM].iloc[1:]
   y1 = df2[targetFSM].iloc[1:]
   XFSM_train, XFSM_test, yFSM_train, yFSM_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
   rbf_feature = RBFSampler(gamma=1, random_state=42, n_components=100)
   modelRBF = make_pipeline(MinMaxScaler(feature_range=(0, 1)), rbf_feature, Ridge(alpha=1.0))
   modelRBF.fit(XFSM_train, yFSM_train)
   yFSM_pred = modelRBF.predict(XFSM_test)
   ultima_fila = df2[featuresFSM].iloc[-1]
   nueva_entrada = ultima_fila.copy()
   nueva_entrada['Year_df'] = fecha_futura.year

   nueva_entrada_df = pd.DataFrame([nueva_entrada], index=[fecha_futura])
   nueva_entrada_scaled = modelRBF.named_steps['minmaxscaler'].transform(nueva_entrada_df)
   nueva_entrada_rbf = modelRBF.named_steps['rbfsampler'].transform(nueva_entrada_scaled)
   prediccion = modelRBF.named_steps['ridge'].predict(nueva_entrada_rbf)
   return jsonify({'fecha': fecha_seleccionada, 'prediccion': prediccion[0]})

@app.route('/predict3', methods=['POST'])
def predict3():
   fecha_seleccionada = request.form['fecha']
   fecha_futura = datetime.strptime(fecha_seleccionada, '%Y-%m-%d')
   df2 = pd.read_csv(file_path)

   # Definir las características y el objetivo
   featuresBVN2 = ['High_BVN', 'Low_BVN', 'Adj Close_BVN','Open_GLD','High_GLD', 'Low_GLD', 'Adj Close_GLD', 'Open_GCF', 'High_GCF','Low_GCF', 'Adj Close_GCF', 'Open_GSPC', 'High_GSPC','Low_GSPC', 'Close_GSPC', 'Open_PEN_X', 'High_PEN_X', 'Low_PEN_X', 'Close_PEN_X','Indicadores de riesgo - Perú', 'Year_df']
   targetBVN2 = 'Open_BVN'

   X1 = df2[featuresBVN2].iloc[1:]
   y1 = df2[targetBVN2].iloc[1:]

   # Dividir los datos en conjuntos de entrenamiento y prueba
   XBVN_train, XBVN_test, yBVN_train, yBVN_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

   # Escalar los datos
   scaler = StandardScaler()
   XBVN_train_scaled = scaler.fit_transform(XBVN_train)
   XBVN_test_scaled = scaler.transform(XBVN_test)

   # Definir y entrenar el modelo Gradient Boosting Regression
   linear_model = LinearRegression()
   linear_model.fit(XBVN_train_scaled, yBVN_train)

   # Preparar la nueva entrada para la predicción
   ultima_fila = df2[featuresBVN2].iloc[1]
   nueva_entrada = ultima_fila.copy()
   nueva_entrada['Year_df'] = fecha_futura.year

   X_new = pd.DataFrame([nueva_entrada])
   X_new_scaled = scaler.transform(X_new)
   # Realizar la predicción
   prediccion = linear_model.predict(X_new_scaled)
   return jsonify({'fecha': fecha_seleccionada, 'prediccion': prediccion[0]})

@app.route('/predict4', methods=['POST'])
def predict4():
   fecha_seleccionada = request.form['fecha']
   fecha_futura = datetime.strptime(fecha_seleccionada, '%Y-%m-%d')
   df2 = pd.read_csv(file_path)
   featuresBHP2 = ['High_T03_BHP', 'Low_T03_BHP', 'Adj Close_T03_BHP','Open_SIF','High_SIF', 'Low_SIF',
   'Adj Close_SIF', 'Open_HGF', 'High_HGF', 'Low_HGF', 'Adj Close_HGF','Open_GSPC', 'High_GSPC',
   'Low_GSPC', 'Close_GSPC', 'Open_DJI', 'High_DJI', 'Low_DJI',
   'Close_DJI','Year_df']
   targetBHP2 = 'Open_T03_BHP'
   X1 = df2[featuresBHP2].iloc[1:]
   y1 = df2[targetBHP2].iloc[1:]
   XBHP_train, XBHP_test, yBHP_train, yBHP_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

   scaler = StandardScaler()
   XBHP_train_scaled = scaler.fit_transform(XBHP_train)
   XBHP_test_scaled = scaler.transform(XBHP_test)

   linear_model = LinearRegression()
   linear_model.fit(XBHP_train_scaled, yBHP_train)
   
   ultima_fila = df2[featuresBHP2].iloc[-1]
   nueva_entrada = ultima_fila.copy()
   nueva_entrada['Year_df'] = fecha_futura.year
   X_new = pd.DataFrame([nueva_entrada], index=[fecha_futura])
   X_new_scaled = scaler.transform(X_new)

   prediccion = linear_model.predict(X_new_scaled)
   return jsonify({'fecha': fecha_seleccionada, 'prediccion': prediccion[0]})

@app.route('/predict5', methods=['POST'])
def predict5():
   fecha_seleccionada = request.form['fecha']
   fecha_futura = datetime.strptime(fecha_seleccionada, '%Y-%m-%d')
   df2 = pd.read_csv(file_path)
   featuresFSM2 = ['High_T05_SCCO', 'Low_T05_SCCO', 'Adj Close_T05_SCCO','Open_SIF','High_SIF', 'Low_SIF',
   'Adj Close_SIF', 'Open_HGF', 'High_HGF', 'Low_HGF', 'Adj Close_HGF','Open_GSPC', 'High_GSPC',
   'Low_GSPC', 'Close_GSPC', 'Open_DJI', 'High_DJI', 'Low_DJI',
   'Close_DJI','Year_df']
   targetFSM2 = 'Open_T05_SCCO'
   X1 = df2[featuresFSM2].iloc[1:]
   y1 = df2[targetFSM2].iloc[1:]
   XFSM_train, XFSM_test, yFSM_train, yFSM_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

   scaler = MinMaxScaler(feature_range=(0, 1))
   XFSM_train_scaled = scaler.fit_transform(XFSM_train)
   XFSM_test_scaled = scaler.transform(XFSM_test)

   linear_model = LinearRegression()
   linear_model.fit(XFSM_train_scaled, yFSM_train)
   
   yFSM_pred = rf_model.predict(XFSM_test_scaled)
   ultima_fila = df2[featuresFSM2].iloc[-1]
   nueva_entrada = ultima_fila.copy()
   nueva_entrada['Year_df'] = fecha_futura.year
   X_new = pd.DataFrame([nueva_entrada], index=[fecha_futura])
   X_new_scaled = scaler.transform(X_new)

   prediccion = linear_model.predict(X_new_scaled)
   return jsonify({'fecha': fecha_seleccionada, 'prediccion': prediccion[0]})

@app.route('/predict6', methods=['POST'])
def predict6():
   fecha_seleccionada = request.form['fecha']
   fecha_futura = datetime.strptime(fecha_seleccionada, '%Y-%m-%d')
   df2 = pd.read_csv(file_path)
   featuresBVN2 = ['High_BVN', 'Low_BVN', 'Adj Close_BVN','Open_GLD','High_GLD', 'Low_GLD', 'Adj Close_GLD', 'Open_GCF', 'High_GCF','Low_GCF', 'Adj Close_GCF', 'Open_GSPC', 'High_GSPC','Low_GSPC', 'Close_GSPC', 'Open_PEN_X', 'High_PEN_X', 'Low_PEN_X', 'Close_PEN_X','Indicadores de riesgo - Perú', 'Year_df']
   targetBVN2 = 'Open_BVN'
   X1 = df2[featuresBVN2].iloc[1:]
   y1 = df2[targetBVN2].iloc[1:]
   XBVN_train, XBVN_test, yBVN_train, yBVN_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

   scaler = StandardScaler()
   XBVN_train_scaled = scaler.fit_transform(XBVN_train)
   XBVN_test_scaled = scaler.transform(XBVN_test)

   svm_model = SVR(kernel='rbf')  # Puedes ajustar el kernel según tus necesidades
   svm_model.fit(XBVN_train_scaled, yBVN_train)
   
   
   ultima_fila = df2[featuresBVN2].iloc[-1]
   nueva_entrada = ultima_fila.copy()
   nueva_entrada['Year_df'] = fecha_futura.year
   X_new = pd.DataFrame([nueva_entrada], index=[fecha_futura])
   X_new_scaled = scaler.transform(X_new)

   prediccion = svm_model.predict(X_new_scaled)
   return jsonify({'fecha': fecha_seleccionada, 'prediccion': prediccion[0]})

@app.route('/predict7', methods=['POST'])
def predict7():
   fecha_seleccionada = request.form['fecha']
   fecha_futura = datetime.strptime(fecha_seleccionada, '%Y-%m-%d')
   df2 = pd.read_csv(file_path)
   featuresBHP2 = ['High_T03_BHP', 'Low_T03_BHP', 'Adj Close_T03_BHP','Open_SIF','High_SIF', 'Low_SIF',
   'Adj Close_SIF', 'Open_HGF', 'High_HGF', 'Low_HGF', 'Adj Close_HGF','Open_GSPC', 'High_GSPC',
   'Low_GSPC', 'Close_GSPC', 'Open_DJI', 'High_DJI', 'Low_DJI',
   'Close_DJI','Year_df']
   targetBHP2 = 'Open_T03_BHP'
   X1 = df2[featuresBHP2].iloc[1:]
   y1 = df2[targetBHP2].iloc[1:]
   XBHP_train, XBHP_test, yBHP_train, yBHP_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

   scaler = StandardScaler()
   XBHP_train_scaled = scaler.fit_transform(XBHP_train)
   XBHP_test_scaled = scaler.transform(XBHP_test)

   svm_model = SVR(kernel='rbf')  # Puedes ajustar el kernel según tus necesidades
   svm_model.fit(XBHP_train_scaled, yBHP_train)
   
   
   ultima_fila = df2[featuresBHP2].iloc[-1]
   nueva_entrada = ultima_fila.copy()
   nueva_entrada['Year_df'] = fecha_futura.year
   X_new = pd.DataFrame([nueva_entrada], index=[fecha_futura])
   X_new_scaled = scaler.transform(X_new)

   prediccion = svm_model.predict(X_new_scaled)
   return jsonify({'fecha': fecha_seleccionada, 'prediccion': prediccion[0]})

@app.route('/predict8', methods=['POST'])
def predict8():
   fecha_seleccionada = request.form['fecha']
   fecha_futura = datetime.strptime(fecha_seleccionada, '%Y-%m-%d')
   df2 = pd.read_csv(file_path)
   featuresFSM2 = ['High_T05_SCCO', 'Low_T05_SCCO', 'Adj Close_T05_SCCO','Open_SIF','High_SIF', 'Low_SIF',
   'Adj Close_SIF', 'Open_HGF', 'High_HGF', 'Low_HGF', 'Adj Close_HGF','Open_GSPC', 'High_GSPC',
   'Low_GSPC', 'Close_GSPC', 'Open_DJI', 'High_DJI', 'Low_DJI',
   'Close_DJI','Year_df']
   targetFSM2 = 'Open_T05_SCCO'
   X1 = df2[featuresFSM2].iloc[1:]
   y1 = df2[targetFSM2].iloc[1:]
   XFSM_train, XFSM_test, yFSM_train, yFSM_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

   scaler = StandardScaler()
   XFSM_train_scaled = scaler.fit_transform(XFSM_train)
   XFSM_test_scaled = scaler.transform(XFSM_test)

   svm_model = SVR(kernel='rbf')  # Puedes ajustar el kernel según tus necesidades
   svm_model.fit(XFSM_train_scaled, yFSM_train)
   
   
   ultima_fila = df2[featuresFSM2].iloc[-1]
   nueva_entrada = ultima_fila.copy()
   nueva_entrada['Year_df'] = fecha_futura.year
   X_new = pd.DataFrame([nueva_entrada], index=[fecha_futura])
   X_new_scaled = scaler.transform(X_new)

   prediccion = svm_model.predict(X_new_scaled)
   return jsonify({'fecha': fecha_seleccionada, 'prediccion': prediccion[0]})

@app.route('/predict9', methods=['POST'])
def predict9():
   fecha_seleccionada = request.form['fecha']
   fecha_futura = datetime.strptime(fecha_seleccionada, '%Y-%m-%d')
   df2 = pd.read_csv(file_path)
   featuresBVN2 = ['High_BVN', 'Low_BVN', 'Adj Close_BVN','Open_GLD','High_GLD', 'Low_GLD', 'Adj Close_GLD', 'Open_GCF', 'High_GCF','Low_GCF', 'Adj Close_GCF', 'Open_GSPC', 'High_GSPC','Low_GSPC', 'Close_GSPC', 'Open_PEN_X', 'High_PEN_X', 'Low_PEN_X', 'Close_PEN_X','Indicadores de riesgo - Perú', 'Year_df']
   targetBVN2 = 'Open_BVN'
   X1 = df2[featuresBVN2].iloc[1:]
   y1 = df2[targetBVN2].iloc[1:]
   XBVN_train, XBVN_test, yBVN_train, yBVN_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

   scaler = StandardScaler()
   XBVN_train_scaled = scaler.fit_transform(XBVN_train)
   XBVN_test_scaled = scaler.transform(XBVN_test)

   knn_model = KNeighborsRegressor(n_neighbors=5)
   knn_model.fit(XBVN_train_scaled, yBVN_train) 
   
   ultima_fila = df2[featuresBVN2].iloc[1]
   nueva_entrada = ultima_fila.copy()
   nueva_entrada['Year_df'] = fecha_futura.year
   print("Nueva entrada antes del escalado:", nueva_entrada)
   X_new = pd.DataFrame([nueva_entrada])
   X_new_scaled = scaler.transform(X_new)
   print("Nueva entrada después del escalado:", X_new_scaled)

   prediccion = knn_model.predict(X_new_scaled)
   return jsonify({'fecha': fecha_seleccionada, 'prediccion': prediccion[0]})

@app.route('/predict10', methods=['POST'])
def predict10():
   fecha_seleccionada = request.form['fecha']
   fecha_futura = datetime.strptime(fecha_seleccionada, '%Y-%m-%d')
   df2 = pd.read_csv(file_path)
   featuresBHP2 = ['High_T03_BHP', 'Low_T03_BHP', 'Adj Close_T03_BHP','Open_SIF','High_SIF', 'Low_SIF',
   'Adj Close_SIF', 'Open_HGF', 'High_HGF', 'Low_HGF', 'Adj Close_HGF','Open_GSPC', 'High_GSPC',
   'Low_GSPC', 'Close_GSPC', 'Open_DJI', 'High_DJI', 'Low_DJI',
   'Close_DJI','Year_df']
   targetBHP2 = 'Open_T03_BHP'
   X1 = df2[featuresBHP2].iloc[1:]
   y1 = df2[targetBHP2].iloc[1:]
   XBHP_train, XBHP_test, yBHP_train, yBHP_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

   scaler = StandardScaler()
   XBHP_train_scaled = scaler.fit_transform(XBHP_train)
   XBHP_test_scaled = scaler.transform(XBHP_test)

   knn_model = KNeighborsRegressor(n_neighbors=5)
   knn_model.fit(XBHP_train_scaled, yBHP_train) 
   
   ultima_fila = df2[featuresBHP2].iloc[1]
   nueva_entrada = ultima_fila.copy()
   nueva_entrada['Year_df'] = fecha_futura.year
   print("Nueva entrada antes del escalado:", nueva_entrada)
   X_new = pd.DataFrame([nueva_entrada])
   X_new_scaled = scaler.transform(X_new)
   print("Nueva entrada después del escalado:", X_new_scaled)

   prediccion = knn_model.predict(X_new_scaled)
   return jsonify({'fecha': fecha_seleccionada, 'prediccion': prediccion[0]})

@app.route('/predict11', methods=['POST'])
def predict11():
   fecha_seleccionada = request.form['fecha']
   fecha_futura = datetime.strptime(fecha_seleccionada, '%Y-%m-%d')
   df2 = pd.read_csv(file_path)
   featuresFSM2 = ['High_T05_SCCO', 'Low_T05_SCCO', 'Adj Close_T05_SCCO','Open_SIF','High_SIF', 'Low_SIF',
   'Adj Close_SIF', 'Open_HGF', 'High_HGF', 'Low_HGF', 'Adj Close_HGF','Open_GSPC', 'High_GSPC',
   'Low_GSPC', 'Close_GSPC', 'Open_DJI', 'High_DJI', 'Low_DJI',
   'Close_DJI','Year_df']
   targetFSM2 = 'Open_T05_SCCO'
   X1 = df2[featuresFSM2].iloc[1:]
   y1 = df2[targetFSM2].iloc[1:]
   XFSM_train, XFSM_test, yFSM_train, yFSM_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

   scaler = StandardScaler()
   XFSM_train_scaled = scaler.fit_transform(XFSM_train)
   XFSM_test_scaled = scaler.transform(XFSM_test)

   knn_model = KNeighborsRegressor(n_neighbors=5)
   knn_model.fit(XFSM_train_scaled, yFSM_train) 
   
   ultima_fila = df2[featuresFSM2].iloc[1]
   nueva_entrada = ultima_fila.copy()
   nueva_entrada['Year_df'] = fecha_futura.year
   X_new = pd.DataFrame([nueva_entrada])
   X_new_scaled = scaler.transform(X_new)

   prediccion = knn_model.predict(X_new_scaled)
   return jsonify({'fecha': fecha_seleccionada, 'prediccion': prediccion[0]})

if __name__ == '__main__':
   port = int(os.environ.get('PORT', 5000))
   app.run(host='0.0.0.0', port=port)