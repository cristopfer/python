import yfinance as yf

def cargarDatos(fechaInicio,fechaFin)
    BVN_df = yf.download('BVN', start = fechaInicio, end = fechaFin)
    BVN_df.drop( ['Volume', 'Close'] , axis=1, inplace=True)
    BVN_df.columns =  (column + "_BVN" for column in BVN_df.columns )

    T02_ABX = yf.download('ABX.TO', start = fechaInicio, end = fechaFin)
    T02_ABX.drop( ['Volume', 'Close'] , axis=1, inplace=True)
    T02_ABX.columns =  (column + "_T02_ABX" for column in T02_ABX.columns )

    T03_BHP = yf.download('BHP', start = fechaInicio, end = fechaFin)
    T03_BHP.drop( ['Volume', 'Close'] , axis=1, inplace=True)
    T03_BHP.columns =  (column + "_T03_BHP" for column in T03_BHP.columns )

    T04_FSM = yf.download('FSM', start = fechaInicio, end = fechaFin)
    T04_FSM.drop( ['Volume', 'Close'] , axis=1, inplace=True)
    T04_FSM.columns =  (column + "_T04_FSM" for column in T04_FSM.columns )

    T05_SCCO = yf.download('SCCO', start=fechaInicio, end=fechaFin)
    T05_SCCO.drop(['Volume', 'Close'], axis=1, inplace=True)
    T05_SCCO.columns = (column + "_T05_SCCO" for column in T05_SCCO.columns )

    GLD_data = yf.download('GLD', start = fechaInicio, end = fechaFin)
    GLD_data.drop(['Volume', 'Close'], axis=1, inplace=True)
    GLD_data.columns = (column + "_GLD" for column in GLD_data.columns )

    GCF_data = yf.download('GC=F', start = fechaInicio, end = fechaFin)
    GCF_data.drop(['Volume', 'Close'], axis=1, inplace=True)
    GCF_data.columns = (column + "_GCF" for column in GCF_data.columns )

    SIF_data = yf.download('SI=F', start = fechaInicio, end = fechaFin)
    SIF_data.drop(['Volume', 'Close'], axis=1, inplace=True)
    SIF_data.columns = (column + "_SIF" for column in SIF_data.columns )

    HGF_data = yf.download('HG=F', start = fechaInicio, end = fechaFin)
    HGF_data.drop(['Volume', 'Close'], axis=1, inplace=True)
    HGF_data.columns = (column + "_HGF" for column in HGF_data.columns )

    T09_ZINC = yf.download('ZINC.L', start = fechaInicio, end = fechaFin)
    T09_ZINC.drop(['Volume', 'Close'], axis=1, inplace=True)
    T09_ZINC.columns = (column + "_ZINC" for column in T09_ZINC.columns )

    BZ_F_data = yf.download('BZ=F', start = fechaInicio, end = fechaFin)
    BZ_F_data.drop(['Volume', 'Close'], axis=1, inplace=True)
    BZ_F_data.columns = (column + "_BZ_F" for column in BZ_F_data.columns )

    GSPC_data = yf.download('^GSPC', start = fechaInicio, end = fechaFin)
    GSPC_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True)
    GSPC_data.columns = (column + "_GSPC" for column in GSPC_data.columns )

    DJI_data = yf.download('^DJI', start = fechaInicio, end = fechaFin)
    DJI_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True)
    DJI_data.columns = (column + "_DJI" for column in DJI_data.columns )

    PEN_X_data = yf.download('PEN=X', start = fechaInicio, end = fechaFin)
    PEN_X_data.drop(['Volume', 'Adj Close'], axis=1, inplace=True)
    PEN_X_data.columns = (column + "_PEN_X" for column in PEN_X_data.columns )

    df = pd.merge(BVN_df, T03_BHP, on='Date')
    df = pd.merge(df, T04_FSM, on='Date')
    df = pd.merge(df, T05_SCCO, on='Date')
    df = pd.merge(df, GLD_data, on='Date')
    df = pd.merge(df, GCF_data, on='Date')
    df = pd.merge(df, SIF_data, on='Date')
    df = pd.merge(df, HGF_data, on='Date')
    df = pd.merge(df, T09_ZINC, on='Date')
    df = pd.merge(df, BZ_F_data, on='Date')
    df = pd.merge(df, GSPC_data, on='Date')
    df = pd.merge(df, DJI_data, on='Date')
    df = pd.merge(df, PEN_X_data, on='Date')

    df['Year_df'] = df.index.year
    df['Month_df'] = df.index.month
    df['BVN_Return'] = df['Adj Close_BVN'].pct_change()
    df['Trend'] = np.where(df['BVN_Return'] > 0.00, 1, 0)
    df = df.dropna(how='any')
    df2 = df.copy()
    return df2

