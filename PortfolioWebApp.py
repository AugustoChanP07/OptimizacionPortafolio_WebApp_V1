#Creación de página
import streamlit as st

# Procesamiento de datos
import numpy as np
import pandas as pd

# Librerías financieras
import yfinance as yf

# Visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Optimización
from scipy.optimize import minimize
import scipy.optimize as sco
from scipy.stats import norm

# - - - Configuración general de la página - - -
st.set_page_config(
    page_title="Optimización de Portafolios",
    layout="centered",
    initial_sidebar_state="expanded"
)

# - - - SIDEBAR - - -
with st.sidebar:
    st.header("Introduce los activos para el portafolio")

# - - Seleccionar fechas de inicio y de cierre - -
    col1, col2 = st.columns(2)
    with col1:
        # Fecha de inicio
        start_date = st.date_input("Fecha de inicio")
    
    with col2:
        # Fecha de cierre
        end_date = st.date_input("Fecha de cierre")

# - - Introducción de valores númericos - - 

    #Tasa libre de riesgo
    Tasa_Libre_Riesgo = st.number_input("Tasa libre de riesgo (%)", min_value=0.0, step=0.1)
    
    #Pesos máximos y minimos
    PesMax, PesMin = st.columns(2)
    
    with PesMax:
        Peso_Maximo = st.number_input("Peso Máximo", min_value=0.0, step=0.1)
    
    with PesMin:
        Peso_Minimo = st.number_input("Peso Mínimo", min_value=0.0, step=0.1)


# - - Introducir los tickets 
    symbols_input = st.text_area(
        "Introduce los tickets de las acciones (uno por línea)",
        placeholder="Ejemplo:\nAAPL\nMSFT\nGOOGL"
    )

    symbols = [s.strip().upper() for s in symbols_input.splitlines() if s.strip()]

    if symbols:
        st.write("### Tickers seleccionados")
        st.dataframe(pd.DataFrame(symbols, columns=["Ticker"]))
    else:
        st.info("Introduce al menos un ticker para continuar.")

# - - - - DESCARGA Y VALIDACIÓN DE DATOS - - - -

#Dataframe con los datos
Assets = pd.DataFrame()

#Si son válidas las fechas entonces descarga de yahoo finance los precios de cierre
if symbols and start_date < end_date:
    raw = yf.download(symbols, start=start_date, end=end_date)["Close"]

    valid_symbols = [] #Lista donde ponemos los simbolos
    
    #Por cada ticket que tengamos en symbols si no esta ponemos que no fue encontrado
    for symbol in symbols: 
        if symbol not in raw.columns or raw[symbol].dropna().empty:
            st.warning(f"El ticket '{symbol}' no fue encontrado o no tiene datos.")
        else: #si se encuentra lo agregamos a la lista
            valid_symbols.append(symbol)

    # Filtrar activos válidos
    Assets = raw[valid_symbols].dropna(axis=1, how='all')

else: 
    st.info("Asegúrate de ingresar tickers válidos y que la fecha de inicio sea anterior a la de cierre.")

# - - - - PAGINA PRINCIPAL - - - -

if not Assets.empty:

    st.title("Optimización de portafolios con frontera eficiente")
    st.write("By Augusto Chan Pacheco")
        
    #Obtenemos los rendimientos logarítmicos de los precios de cierre
    log_returns = np.log(Assets / Assets.shift(1))
    st.subheader("Rendimientos diarios (LN)")
    
    #Texto explicativo sobre los rendimientos logarítmicos
    st.write("Los rendimientos diarios representan el cambio porcentual el precio de un activo de un día al siguiente. " \
    "Permiten analizar la evolución del valor de una inversión en el corto plazo y " \
    "son fundamentales para calcular métricas como la volatilidad, la covarianza entre activos y el riesgo total del portafolio.")
    
    #Proyectamos el dataframe de los rendimientos  
    st.dataframe(log_returns)

# - - Métricas importantes - -

    #Rendimientos = Exp(promedio de los rendimientos * 1 año) - 1
    Returns = np.exp(log_returns.mean() * 252)-1

    #Volatibilidad = Desviación estándar de los rendimientos x raiz de 1 año - -
    Volatility = log_returns.std() * np.sqrt(252)
    
    #Varianza = Varianza de los rendimientos x 1 año
    Variance = log_returns.var() * 252
    Sharp_Ratio = (Returns - (Tasa_Libre_Riesgo / 100)) / (Volatility)
    
    #Valores Z
    z_95 = norm.ppf(0.95)
    z_99 = norm.ppf(0.99)

    #Valor en riesgo con 95% y 99% de confianza
    VaR_95 = -(z_95 * Volatility - Returns)
    VaR_99 = -(z_99 * Volatility - Returns)


# - - Creación del dataframe para métricas importantes - -
    AssetsInfo = pd.DataFrame({
        "Rendimiento": Returns,
        "Volatibilidad": Volatility,
        "Varianza": Variance,
        "Sharp Ratio": Sharp_Ratio,
        "VaR 95%": VaR_95,
        "VaR 99%": VaR_99
    })

    #Darle formato de dos decimales y porcentaje al Dataframe
    AssetsInfo = AssetsInfo.applymap(lambda x: f"{x*100:.2f}%")
    
    st.subheader("Información individual")
    
    #Texto explicativo sobre las métricas usadas.
    st.write("El rendimiento indica la ganancia total obtenida en el periodo analizado, " \
    "mientras que la volatilidad refleja qué tanto varía el precio, mostrando el nivel de riesgo. " \
    "La varianza complementa esto al medir la dispersión estadística del rendimiento. " \
    "El Sharpe Ratio evalúa la rentabilidad ajustada por riesgo, siendo más alto cuando el desempeño es más eficiente. " \
    "El VaR al 95% y al 99% representan la pérdida máxima esperada bajo condiciones normales con esos niveles de confianza, siendo el 99% más conservador.")
    
    #Proyectamos el dataframe de la información general
    st.dataframe(AssetsInfo)


# - - - Gráfico de los precios de cierres - - -

    #Primero convierte las fechas en el índice y luego los Assets en filas
    InteractiveAssetsPlot = Assets.reset_index().melt(id_vars='Date', var_name='Activo', value_name='Precio')

    #Creamos la gráfica los precios de cierres interactivos
    fig_Prices = px.line(InteractiveAssetsPlot, x='Date', y='Precio', color='Activo',
                title='Precios históricos interactivos',
                labels={'Date': 'Fecha', 'Precio': 'Precio', 'Activo': 'Activo'})

    fig_Prices.update_xaxes(rangeslider_visible=True)
    fig_Prices.update_layout(width=950, height=550)

    #Hacemos gráfica lineal interactiva de los precios de cierre
    st.plotly_chart(fig_Prices, use_container_width=True)


#- - - Gráfico de cambio porcentual acumulado base 0 - - - 
    Assets_AbsChange = (Assets / Assets.iloc[0] * 100)

    #Primero convierte las fechas en el índice y luego los Assets en filas
    Assets_AbsChangePlot = Assets_AbsChange.reset_index().melt(id_vars='Date', var_name='Activo', value_name='Precio')

    #Creamos la grafica los cambios porcentuales acumulados en base 0
    Change = px.line(Assets_AbsChangePlot, x='Date', y='Precio', color='Activo',
                title='Cambio porcentual acumulado (Base 100)',
                labels={'Date': 'Fecha', 'Precio': 'Índice Base 100', 'Activo': 'Activo'})

    Change.update_xaxes(rangeslider_visible=True)
    Change.update_layout(width=950, height=550)

    #Hacemos gráfica lineal interactiva de los cambios porcentual base 0 
    st.plotly_chart(Change, use_container_width=True)


#- - - Gráfico de rendimientos logarítmicos - - -
    log_returns_plot = log_returns.reset_index().melt(id_vars='Date', var_name='Activo', value_name='Precio')

    #Creamos la gráfica de los rendimientos logarítmicos
    LogPlot = px.line(log_returns_plot, x='Date', y='Precio', color='Activo',
                title='Rendimientos logaritmicos interactivos',
                labels={'Date': 'Fecha', 'Precio': 'Precio', 'Activo': 'Activo'})

    LogPlot.update_xaxes(rangeslider_visible=True)
    LogPlot.update_layout(width=950, height=550)

    #Hacemos la gráfica los rendimientos logarítmicos
    st.plotly_chart(LogPlot, use_container_width=True)


# - - - Elaboración de la matriz de correlación y covarianza - - -
    
    #Obtener Matrices de covarianza y correlación - -
    ReturnsCOV = log_returns.cov()
    ReturnsCORR = log_returns.corr()

# - - Matriz de correlación - -
    
    #Creación de matriz de correlación
    fig1, ax1 = plt.subplots()
    sns.heatmap(ReturnsCORR, cmap="Reds", annot=True, ax=ax1)
    
    #Titulo de matriz de correlación
    st.subheader("Matriz de correlación")
    
    #Texto explicación sobre la matriz de correlación
    st.write("La matriz de correlación sirve para identificar y resumir las relaciones lineales entre varias variables " \
    "en un conjunto de datos. Permite detectar dependencias, redundancias y patrones de comportamiento, " \
    "lo que facilita la selección de variables en modelos estadísticos o predictivos")
    
    #Proyectar el mapa de correlación
    st.pyplot(fig1)

# - - Matriz de covarianza - - 
    
    #Creación de matriz de covarianza
    fig2, ax2 = plt.subplots()
    sns.heatmap(ReturnsCOV, cmap="Blues", annot=True, ax=ax2)
    
    #Titulo de matriz de Covarianza
    st.subheader("Matriz de Covarianza")
    
    #Texto explicación sobre la matriz de covarianza
    st.write("La covarianza mide cómo dos variables varían conjuntamente. " \
    "Si tienden a aumentar o disminuir al mismo tiempo, la covarianza es positiva;" \
    "si una sube mientras la otra baja, es negativa.")
    
    #Proyectar el mapa de covarianza
    st.pyplot(fig2)

# - - - Elaboración de la frontera eficiente - - -
    
    #Obtenemos el número de tickets
    noa = len(symbols)

    #Asignamos un peso al azar 
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    
    #Función para obtener el rendimiento
    def port_ret(weights):
        return np.sum(log_returns.mean() * weights) *252

    #Función para obtener la volatbilidad 
    def port_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))

# - - Simulaciones en la frontera eficiente

    #Lista de simulaciones de la frontera eficiente
    prets = []
    pvols = []

    #Repetimos este proceso haciendo smilaciones con diferentes pesos
    for p in range(2500):
        weights = np.random.random(noa)
        weights /= np.sum(weights)
        
        #Una vez hace los pesos los pone en las listas de la frontera.
        prets.append(port_ret(weights))
        pvols.append(port_vol(weights))
    
    #Lo convierte a array de numpy
    prets = np.array(prets)
    pvols = np.array(pvols)

# - - Función Objetivo: Calcula el Sharp Ratio                    
    def min_func_sharpe(weights):
        return -(port_ret(weights) - (Tasa_Libre_Riesgo / 100)) / port_vol(weights)
        # Es negativo porque .minimize busca minimizar mientras que nosotros buscamos maximizar

    #Restricción de igualdad donde la suma de los pesos debe de ser igual a 1
    cons =({'type': 'eq', 'fun' : lambda x: np.sum(x) - 1})
    
    # Cada peso debe ser entre 0 y 1
    bnds =tuple((0,1) for x in range(noa))
    
    #Punto inicial: se empieza con una distribución equitativa
    eweights = np.array(noa * [1. /noa,])
    
# - - Portafolio que maximice la relación riesgo rendimiento
    opts = sco.minimize(min_func_sharpe, eweights,
                method = 'SLSQP', bounds=bnds,
                constraints=cons)
    
# - - Portafolio de mínima volatilidad
    optv = sco.minimize(port_vol, eweights,
                method='SLSQP', bounds=bnds,
                constraints=cons)
    
    #los pesos deben de ser entre 0 y 1
    bnds = tuple((0,1) for x in weights)

    #Calcula el máximo retorno para el portafolio de mínima volatilidad
    min_vol_port_ret = port_ret(optv['x'])

    #Calcula el retorno máximo para cada asset
    max_asset_ret = np.max(log_returns.mean() * 252)

    #Rango de retornos objetivos
    trets = np.linspace(min_vol_port_ret, max_asset_ret * 1.1, 50) 
    tvols = []
    
    #Loop para calcular volatibilidades asociadas
    for tret in trets:
        
        #Restricciones el portafolio debe tener el retorno y la suma de sus pesos menor a 1
        cons = ({'type': 'eq','fun': lambda x, tr=tret: port_ret(x) - tr},
                {'type': 'eq','fun': lambda x: np.sum(x) - 1})
        
        res = sco.minimize(port_vol, eweights, method='SLSQP',
                            bounds=bnds, constraints=cons)
        
        #Devuelve la volatibilidad mínima
        tvols.append(res['fun'])
    
    tvols = np.array(tvols)
    

    # Crear figura y ejes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot de portafolios simulados
    scatter = ax.scatter(
        pvols, prets, c=prets/pvols,
        marker='.', alpha=0.8, cmap='coolwarm'
    )

    # Línea de la frontera eficiente
    ax.plot(tvols, trets, 'b', lw=0.4)

    # Portafolio con máxima razón de Sharpe
    ax.plot(
        port_vol(opts['x']), port_ret(opts['x']),
        'y*', markersize=15.0, label='Max Sharpe Ratio Portfolio'
    )

    # Portafolio con mínima volatilidad
    ax.plot(
        port_vol(optv['x']), port_ret(optv['x']),
        'r*', markersize=15.0, label='Min Volatility Portfolio'
    )

# - - Etiquetas y leyenda de la fronera eficiente - - 
    
    #Titulo
    st.write("## Frontera eficiente")

    #Descripción de la frontera eficiente      
    st.write(" Es una curva que muestra los portafolios óptimos que ofrecen el mayor" \
    "rendimiento posible para cada nivel de riesgo." \
    "Basada en la teoría moderna de portafolios, ayuda a identificar" \
    "combinaciones de activos que maximizan el retorno esperado sin asumir riesgo innecesario.")
    
    ax.set_xlabel('Expected volatility')
    ax.set_ylabel('Expected return')
    fig.colorbar(scatter, label='Sharpe ratio')
    ax.legend()
    ax.grid(True)

    # Mostrar en Streamlit la frontera eficiente
    st.pyplot(fig)
                
# - - - Portafolios de inversión obtenidos de mi frotnera eficiente - - - 
    MinVol_Volatility = port_vol(optv['x'])
    MinVol_Returns = port_ret(optv['x'])

    MaxSharpPort_Volatility = port_vol(opts['x'])
    MaxSharpPort_Returns = port_ret(opts['x'])

    MaxSharpPort_Weights = opts['x'].round(3)
    MinVol_Weight = optv['x'].round(3)

# - - Portafolio de mínima volatilidad - - 

    #Dataframe con los pesos para obtener mínima volatilidad
    min_vol_weights_df = pd.DataFrame({
        'Asset': symbols,
        'Weight': optv['x']})

    #Ordenar los pesos
    min_vol_weights_df = min_vol_weights_df.sort_values(by='Weight', ascending=False)
    
    #Darle formato a los pesos con dos decimales
    min_vol_weights_df['Weight'] = min_vol_weights_df['Weight'].apply(lambda x: f"{x:.2%}")

    #Titulo  y explicación del portafolio
    st.subheader("Distribución de pesos para un portafolio con mínima volatibilidad")
    st.write(f"El portafolio tiene un retorno anualizado de {MinVol_Returns: .2%} \
             mientras que tiene una volatilidad anual de {MinVol_Volatility: .2%}")
    
    #Tabla con los pesos de mínima volatilidad
    st.dataframe(min_vol_weights_df, use_container_width=True)

    
# - - Portafolio de que maximiza el Riesgo-Rendimiento - - 

    #Dataframe con los pesos que maximizan el Riesgo-Rendimiento
    MaxSharpPort_weights_DF = pd.DataFrame({
        'Asset': symbols,
        'Weight': opts['x']})

    #Ordenar los pesos
    MaxSharpPort_weights_DF = MaxSharpPort_weights_DF.sort_values(by='Weight', ascending=False)
    
    #Darle formato a los pesos con dos decimales
    MaxSharpPort_weights_DF['Weight'] = MaxSharpPort_weights_DF['Weight'].apply(lambda x: f"{x:.2%}")

    #Titulo y explicación del portafolio
    st.subheader("Distribución de pesos para un portafolio que máximice la relación riesgo rendimiento")
    st.write(f"El portafolio tiene un retorno anualizado de {MaxSharpPort_Returns: .2%} \
             mientras que tiene una volatilidad anual de {MaxSharpPort_Volatility: .2%}")
    
    #Tabla con los pesos que maximizan la relación riesgo rendimiento
    st.dataframe(MaxSharpPort_weights_DF, use_container_width=True)


# - - - Optimización de portafolios en base a máximos y mínimos - - -

    #Restricciones en base a pesos máximos y minimos
    bnds_constrained = tuple((Peso_Minimo, Peso_Maximo) for x in range(noa))

    # Optimización para el portafolio de Máximo Sharpe Ratio con restricciones de pesos
    opts_constrained = sco.minimize(min_func_sharpe, eweights,
                    method = 'SLSQP', bounds=bnds_constrained,
                    constraints=cons)

    # Optimización para el portafolio de Mínima Volatilidad con restricciones de pesos
    optv_constrained = sco.minimize(port_vol, eweights,
                    method='SLSQP', bounds=bnds_constrained,
                    constraints=cons)

    MaxSharpPort_constrained_Return = port_ret(opts_constrained['x'])
    MaxSharpPort_constrained_Volatility = port_vol(opts_constrained['x'])

# - - Recalcular la frontera eficiente con los nuevos límites - - 

    #Calcula el máximo retorno para el portafolio de mínima volatilidad bajo las restricciones
    min_vol_port_ret_constrained = port_ret(optv_constrained['x'])
   
    #Rango de retornos objetivos
    trets_constrained = np.linspace(min_vol_port_ret_constrained, max_asset_ret * 1.1, 50)
    tvols_constrained = []

    #Loop para calcular volatibilidades asociadas
    for tret in trets_constrained:
        cons_constrained = ({'type': 'eq','fun': lambda x, tr=tret: port_ret(x) - tr},
                            {'type': 'eq','fun': lambda x: np.sum(x) - 1})
        res = sco.minimize(port_vol, eweights, method='SLSQP',
                            bounds=bnds_constrained, constraints=cons_constrained)
        tvols_constrained.append(res['fun'])
    tvols_constrained = np.array(tvols_constrained)

    MaxSharpPort_Weights_constrained = opts_constrained['x']
    MaxSharpPort_Weights_constrained = MaxSharpPort_Weights_constrained / np.sum(MaxSharpPort_Weights_constrained)


    #Dataframe del portafolio la relación riesgo-rendimiento con restricciones
    MaxSharpPort_weights_DF_constrained = pd.DataFrame({
        'Asset': symbols,
        'Weight': MaxSharpPort_Weights_constrained.round(4)
    })

    MaxSharpPort_weights_DF_constrained = MaxSharpPort_weights_DF_constrained.sort_values(by='Weight', ascending=False)
    MaxSharpPort_weights_DF_constrained['Weight'] = MaxSharpPort_weights_DF_constrained['Weight'].apply(lambda x: f"{x:.2%}")
    
    st.subheader("Optimización de portafolio en base a pesos máximos y minimos")

    st.write(f"El portafolio tiene un retorno anualizado de {MaxSharpPort_constrained_Return: .2%} \
             mientras que tiene una volatilidad anual de {MaxSharpPort_constrained_Volatility: .2%}")
    
    st.dataframe(MaxSharpPort_weights_DF_constrained, use_container_width=True)

else:

    st.warning("No se han cargado datos válidos para mostrar análisis.")
