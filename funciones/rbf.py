from flask import  send_file, url_for
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
import matplotlib.pyplot as plt
import tempfile

def bnf(fecha_seleccionada, df):
    fecha_futura = datetime.strptime(fecha_seleccionada, '%Y-%m-%d')
    fecha_seleccionada = pd.to_datetime(fecha_seleccionada)
    #df = pd.read_csv(file_path)
    df['Year_df'] = pd.to_datetime(df['Year_df'])
    df2 = df[df['Year_df'] < fecha_seleccionada]
    featuresBVN1 = ['High_BVN', 'Low_BVN', 'Adj Close_BVN','Open_GLD',
       'High_GLD', 'Low_GLD', 'Adj Close_GLD', 'Open_GCF', 'High_GCF',
       'Low_GCF', 'Adj Close_GCF', 'Open_GSPC', 'High_GSPC',
       'Low_GSPC', 'Close_GSPC', 'Open_PEN_X', 'High_PEN_X', 'Low_PEN_X', 'Close_PEN_X']
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

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir='static') as tmpfile:
        plt.figure(figsize=(10, 5))
        plt.plot(yBVN_test.values, label='Actual')
        plt.plot(yBVN_pred, label='Predicted')
        plt.legend()
        plt.title('Actual vs Predicted Open_BVN')
        plt.xlabel('Samples')
        plt.ylabel('Open_BVN')
        plt.savefig(tmpfile.name, format='png')
        tmpfile_path = tmpfile.name
   
    image_url = url_for('static', filename=os.path.basename(tmpfile_path))
    ultima_fila = df2[featuresBVN1].iloc[-1]
    nueva_entrada = ultima_fila.copy()
    nueva_entrada_df = pd.DataFrame([nueva_entrada])
    nueva_entrada_scaled = modelRBF.named_steps['minmaxscaler'].transform(nueva_entrada_df)
    nueva_entrada_rbf = modelRBF.named_steps['rbfsampler'].transform(nueva_entrada_scaled)
    prediccion = modelRBF.named_steps['ridge'].predict(nueva_entrada_rbf)
    return prediccion[0], image_url
