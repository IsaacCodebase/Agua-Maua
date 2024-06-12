import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
from scipy.stats import zscore
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Dados fornecidos ajustados para ter o mesmo comprimento
dados = {
    'Timestamp': [1615089600000, 1615176000000, 1615262400000, 1615348800000, 1615435200000, 1615521600000, 1615608000000, 1615694400000, 1615780800000, 1615867200000, 1615953600000, 1616040000000, 1616126400000, 1616212800000, 1616299200000, 1616385600000, 1616472000000, 1616558400000, 1616644800000, 1616731200000, 1616817600000, 1616904000000, 1616990400000, 1617076800000, 1617163200000],
    'WaterTankLevel': [1000, 950, 900, 850, 800, 831, 755, 797, 818, 836, 806, 761, 825, 778, 832, 785, 829, 300, 834, 828, 789, 807, 758, 753, 774],
    'EntradaPressao': [30, 32, 29, 31, 33, 32, 33, 32, 32, 32, 32, 32, 60, 33, 32, 34, 32, 32, 33, 33, 34, 34, 34, 34, 34],
    'SaidaPressao': [25, 26, 24, 27, 28, 29, 10, 27, 29, 29, 29, 29, 28, 27, 27, 27, 29, 27, 28, 28, 29, 28, 29, 29, 28],
    'Consumo': [100, 150, 130, 170, 160, 138, 120, 202, 162, 176, 600, 112, 110, 193, 174, 142, 174, 131, 179, 125, 206, 157, 120, 191, 149],
    'WaterMeterReading': [1000, 1100, 1200, 1250, 1350, 1488, 1470, 1552, 1512, 1526, 1478, 1462, 1460, 1543, 1524, 1492, 1524, 1481, 1529, 1475, 1556, 1507, 1470, 1541, 1479]
}


# Converter Timestamp para datetime
dados['Timestamp'] = pd.to_datetime(dados['Timestamp'], unit='ms')

# Criar DataFrame
df = pd.DataFrame(dados)

# Seção de Menu Lateral
st.sidebar.title('Menu Lateral')

def parte_bemvindo():
    st.header('Ola')

def parte_graficos():
        # Título do aplicativo
    st.title('Analise de dados')

        # Estatísticas descritivas
    st.subheader("Estatísticas Descritivas")
    st.write(df.describe())

    
    # Seção 3: Gráfico Matplotlib
    st.header('Gráficos')


    def gerar_relatorio(df):
        # Estatísticas descritivas

        # Visualizações
        st.write(f'Consumo de Água ao Longo do Tempo')
        plt.figure(figsize=(10, 5))
        plt.plot(df['Timestamp'], df['Consumo'], marker='o')
        plt.title('Consumo de Água ao Longo do Tempo')
        plt.xlabel('Data')
        plt.ylabel('Consumo (litros)')
        plt.grid(True)
        plt.show()
        st.pyplot(plt)


        st.write(f'Nível do Tanque de Água ao Longo do Tempo')
        plt.figure(figsize=(10, 5))
        plt.plot(df['Timestamp'], df['WaterTankLevel'], marker='o', color='b')
        plt.title('Nível do Tanque de Água ao Longo do Tempo')
        plt.xlabel('Data')
        plt.ylabel('Nível do Tanque (litros)')
        plt.grid(True)
        plt.show()
        st.pyplot(plt)


        st.write(f'Pressão de Entrada e Saída ao Longo do Tempo')
        plt.figure(figsize=(10, 5))
        plt.plot(df['Timestamp'], df['EntradaPressao'], marker='o', label='Pressão de Entrada', color='r')
        plt.plot(df['Timestamp'], df['SaidaPressao'], marker='o', label='Pressão de Saída', color='g')
        plt.title('Pressão de Entrada e Saída ao Longo do Tempo')
        plt.xlabel('Data')
        plt.ylabel('Pressão (psi)')
        plt.legend()
        plt.grid(True)
        plt.show()
        st.pyplot(plt)


        st.write(f'Matriz de Correlação')
        correlation_matrix = df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Matriz de Correlação')
        plt.show()
        st.pyplot(plt)

        # Histogram
        st.write(f'Histograma do Consumo de Água')
        plot_histogram(df)

        # Boxplot
        st.write(f'Boxplot do Consumo de Água')
        plot_boxplot(df)

        # Scatter plot
        st.write(f'Relação entre Pressão de Entrada e Saída')
        plot_scatter(df)

        # Trend line
        st.write(f'Consumo de Água ao Longo do Tempo com Linha de Tendência')
        plot_trend_line(df)

        # Trend line
        st.write(f'Consumo de Água ao Longo do Tempo em comparacao com o Nivel de Agua no Tank')
        plot_water_chart(df)


        st.write(f'Agora para fazer um grafico com uma selecao mais especifica dos dados')
            # Adicionar sliders para filtrar dados
        start_date = st.date_input('Data inicial', df['Timestamp'].min())
        end_date = st.date_input('Data final', df['Timestamp'].max())
        consumo_range = st.slider('Consumo (litros)', int(df['Consumo'].min()), int(df['Consumo'].max()), (int(df['Consumo'].min()), int(df['Consumo'].max())))
        entrada_pressao_range = st.slider('Pressão de Entrada (psi)', int(df['EntradaPressao'].min()), int(df['EntradaPressao'].max()), (int(df['EntradaPressao'].min()), int(df['EntradaPressao'].max())))
        saida_pressao_range = st.slider('Pressão de Saída (psi)', int(df['SaidaPressao'].min()), int(df['SaidaPressao'].max()), (int(df['SaidaPressao'].min()), int(df['SaidaPressao'].max())))

        # Filtrar dados com base nas entradas
        df_filtered = df[(df['Timestamp'] >= pd.to_datetime(start_date)) & (df['Timestamp'] <= pd.to_datetime(end_date))]
        df_filtered = df_filtered[(df_filtered['Consumo'] >= consumo_range[0]) & (df_filtered['Consumo'] <= consumo_range[1])]
        df_filtered = df_filtered[(df_filtered['EntradaPressao'] >= entrada_pressao_range[0]) & (df_filtered['EntradaPressao'] <= entrada_pressao_range[1])]
        df_filtered = df_filtered[(df_filtered['SaidaPressao'] >= saida_pressao_range[0]) & (df_filtered['SaidaPressao'] <= saida_pressao_range[1])]

        if st.button('Atualizar visualizações'):
            # Visualizações
            st.subheader("Consumo de Água ao Longo do Tempo")
            fig, ax = plt.subplots()
            ax.plot(df_filtered['Timestamp'], df_filtered['Consumo'], marker='o')
            ax.set_title('Consumo de Água ao Longo do Tempo')
            ax.set_xlabel('Data')
            ax.set_ylabel('Consumo (litros)')
            ax.grid(True)
            st.pyplot(fig)


        st.subheader("Pressão de Entrada e Saída ao Longo do Tempo")
        fig, ax = plt.subplots()
        ax.plot(df_filtered['Timestamp'], df_filtered['EntradaPressao'], marker='o', label='Pressão de Entrada', color='r')
        ax.plot(df_filtered['Timestamp'], df_filtered['SaidaPressao'], marker='o', label='Pressão de Saída', color='g')
        ax.set_title('Pressão de Entrada e Saída ao Longo do Tempo')
        ax.set_xlabel('Data')
        ax.set_ylabel('Pressão (psi)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


    # Funções adicionais
    def plot_histogram(df):
        plt.figure(figsize=(10, 5))
        plt.hist(df['Consumo'], bins=10, color='c', edgecolor='black')
        plt.title('Histograma do Consumo de Água')
        plt.xlabel('Consumo (litros)')
        plt.ylabel('Frequência')
        plt.grid(True)
        plt.show()
        st.pyplot(plt)

    def plot_boxplot(df):
        plt.figure(figsize=(10, 5))
        plt.boxplot(df['Consumo'], vert=False, patch_artist=True, boxprops=dict(facecolor='c'))
        plt.title('Boxplot do Consumo de Água')
        plt.xlabel('Consumo (litros)')
        plt.grid(True)
        plt.show()
        st.pyplot(plt)

    def plot_scatter(df):
        plt.figure(figsize=(10, 5))
        plt.scatter(df['EntradaPressao'], df['SaidaPressao'], color='m', marker='x')
        plt.title('Relação entre Pressão de Entrada e Saída')
        plt.xlabel('Pressão de Entrada (psi)')
        plt.ylabel('Pressão de Saída (psi)')
        plt.grid(True)
        plt.show()
        st.pyplot(plt)

    def plot_trend_line(df):
        plt.figure(figsize=(10, 5))
        plt.plot(df['Timestamp'], df['Consumo'], marker='o', label='Consumo')
        z = np.polyfit(df['Timestamp'].map(pd.Timestamp.toordinal), df['Consumo'], 1)
        p = np.poly1d(z)
        plt.plot(df['Timestamp'], p(df['Timestamp'].map(pd.Timestamp.toordinal)), "r--", label='Tendência')
        plt.title('Consumo de Água ao Longo do Tempo com Linha de Tendência')
        plt.xlabel('Data')
        plt.ylabel('Consumo (litros)')
        plt.legend()
        plt.grid(True)
        plt.show()
        st.pyplot(plt)

    def plot_water_chart(df):
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df['Timestamp'], df['WaterTankLevel'], linestyle='-', color='b')
        ax1.set_xlabel('Tempo')
        ax1.set_ylabel('Nível de Água (mm)')
        ax1.set_title('Nível de Água do Tanque')
        ax1.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        plt.plot(df['Timestamp'], df['Consumo'], marker='o', label='Consumo')
        plt.plot(df['Timestamp'], df['WaterTankLevel'], marker='o', label='WaterTankLevel')
        ax2.set_xlabel('Tempo')
        ax2.set_ylabel('Volume Total (L)')
        ax2.set_title('Consumo de Água ao Longo do Tempo em comparacao com o Nivel de Agua no Tank')
        ax2.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        ax2.legend()

        plt.show()
        st.pyplot(plt)

    # Gerar relatório
    gerar_relatorio(df)

def parte_previsao(df):
    st.header('Previsoes')

    st.write(f'Previsao do nivel da agua')
    # Substituir 'NaN' por valores ausentes reconhecíveis pelo pandas
    df['WaterTankLevel'] = df['WaterTankLevel'].replace('NaN', pd.NA)

    # Remover valores ausentes
    df = df.dropna()

    # Converter 'WaterTankLevel' para tipo numérico
    df['WaterTankLevel'] = pd.to_numeric(df['WaterTankLevel'])

    # Ajustar o modelo SARIMA
    modelo = SARIMAX(df['WaterTankLevel'], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    resultado = modelo.fit(disp=False)

    # Fazer previsões para o futuro
    horizonte_futuro = 10  # Horizonte de previsão de 5 unidades de tempo (por exemplo, dias)
    previsao = resultado.get_forecast(steps=horizonte_futuro)

    # Obter os valores previstos e seus intervalos de confiança
    previsao_media = previsao.predicted_mean
    intervalo_confianca = previsao.conf_int()

    # Imprimir as previsões

    st.write(f'Previsao do Nível do Tanque de Água da ao Longo do Tempo')
    plt.figure(figsize=(10, 5))
    plt.plot(df['Timestamp'], df['WaterTankLevel'], marker='o')
    plt.title('Nível do Tanque de Água ao Longo do Tempo')
    plt.xlabel('Data')
    plt.ylabel('Consumo (litros)')
    plt.grid(True)
    plt.show()
    st.pyplot(plt)

    df2 = pd.concat([previsao_media, intervalo_confianca], axis=1)
    df2.columns = ['Forecasted', 'Lower CI', 'Upper CI']

    st.write(df2)

    st.write(f'Previsao de consumo')
    # Substituir 'NaN' por valores ausentes reconhecíveis pelo pandas
    df['Consumo'] = df['Consumo'].replace('NaN', pd.NA)

    # Remover valores ausentes
    df = df.dropna()

    # Converter 'Consumo' para tipo numérico
    df['Consumo'] = pd.to_numeric(df['Consumo'])

    # Ajustar o modelo SARIMA
    modelo = SARIMAX(df['Consumo'], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    resultado = modelo.fit(disp=False)

    # Fazer previsões para o futuro
    horizonte_futuro = 10  # Horizonte de previsão de 5 unidades de tempo (por exemplo, dias)
    previsao = resultado.get_forecast(steps=horizonte_futuro)

    # Obter os valores previstos e seus intervalos de confiança
    previsao_media2 = previsao.predicted_mean
    intervalo_confianca2 = previsao.conf_int()

    # Imprimir as previsões

    st.write(f'Previsao do Consumo de Água ao Longo do Tempo')
    plt.figure(figsize=(10, 5))
    plt.plot(df['Timestamp'], df['Consumo'], marker='o')
    plt.title('Consumo de Água ao Longo do Tempo')
    plt.xlabel('Data')
    plt.ylabel('Consumo (litros)')
    plt.grid(True)
    plt.show()
    st.pyplot(plt)

    df2 = pd.concat([previsao_media2, intervalo_confianca2], axis=1)
    df2.columns = ['Forecasted', 'Lower CI', 'Upper CI']

    st.write(df2)


    # Anomalia no consumo
    df['Consumo_zscore'] = zscore(df['Consumo'])
    st.subheader("Anomalias encontradas no Consumo")
    anomalias = df[df['Consumo_zscore'].abs() > 2]
    ano = pd.concat([anomalias['Timestamp'], anomalias['Consumo']], axis=1)
    st.write(ano)

    # Anomalia no nivel da agua
    df['NiveldaAgua_zscore'] = zscore(df['WaterTankLevel'])
    st.subheader("Anomalias encontradas no Nivel da Agua do Tank")
    anomalias2 = df[df['NiveldaAgua_zscore'].abs() > 2]
    ano = pd.concat([anomalias['Timestamp'], anomalias['WaterTankLevel']], axis=1)
    st.write(ano)

    # Anomalia no pressao entrada
    df['pressaoentrada_zscore'] = zscore(df['EntradaPressao'])
    st.subheader("Anomalias encontradas na Pressao de Entrada")
    anomalias3 = df[df['pressaoentrada_zscore'].abs() > 2]
    ano = pd.concat([anomalias['Timestamp'], anomalias['EntradaPressao']], axis=1)
    st.write(ano)

    # Anomalia no pressao saida
    df['pressaosaida_zscore'] = zscore(df['SaidaPressao'])
    st.subheader("Anomalias encontradas na Pressao de Saida")
    anomalias4 = df[df['pressaosaida_zscore'].abs() > 2]
    ano = pd.concat([anomalias['Timestamp'], anomalias['SaidaPressao']], axis=1)
    st.write(ano)

def parte_aviso(df):

    st.title('Teste de gráfico e manipulação')

        # Exibir os dados
    st.write(df)

    # Variável para controlar se o aviso já foi emitido
    aviso_emitido1 = False
    aviso_emitido2 = False

        # Função para destacar pontos específicos
    def plot_with_highlight(ax, x, y, threshold, color='red', label=None):
        ax.plot(x, y, linestyle='-', color='b', label='Volume')
        highlight = y >= threshold
        ax.scatter(x[highlight], y[highlight], color=color, label=label if label else f'<= {threshold} bar')

    def plot_with_highlight2(ax, x, y, threshold, color='red', label=None):
        ax.plot(x, y, linestyle='-', color='b', label='Volume')
        highlight = y <= threshold
        ax.scatter(x[highlight], y[highlight], color=color, label=label if label else f'<= {threshold} bar')
    
    # Função para plotar gráfico
    def plot_pressure_chart(data, pressure_col, title):
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_with_highlight(ax, data['Timestamp'], data[pressure_col], 34, color='red', label=f'>= 34 {pressure_col}')
        ax.set_xlabel('Tempo')
        ax.set_ylabel(f'Pressão de {pressure_col} (bar)')
        ax.set_title(title)
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        ax.legend()
        st.pyplot(fig)

    def plot_pressure_chart2(data, pressure_col, title):
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_with_highlight2(ax, data['Timestamp'], data[pressure_col], 24, color='red', label=f' <=24 {pressure_col}')
        ax.set_xlabel('Tempo')
        ax.set_ylabel(f'Pressão de {pressure_col} (bar)')
        ax.set_title(title)
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        ax.legend()
        st.pyplot(fig)

    # Exibir gráficos de pressão de entrada e saída
    st.title('Pressão de Entrada e Saída por dia')
    st.write("Pressao de entrada")
    plot_pressure_chart(df, 'EntradaPressao', 'Pressão de Entrada por dia')
    for index, row in df.iterrows():
        if (row['EntradaPressao'] >= 34) and aviso_emitido1 == False:
            st.warning(f'Atenção: A pressão de entrada atingiu mais que 34 bars no tempo {row["Timestamp"]}!', icon="⚠️")
            aviso_emitido1 = True
    
    st.write("pressao de saida")
    plot_pressure_chart2(df, 'SaidaPressao', 'Pressão de Saída por dia')
    for index, row in df.iterrows():
        if  (row['SaidaPressao'] <= 24) and not aviso_emitido2:
            st.warning(f'Atenção: A pressão de saida atingiu menos que 24 bars no tempo {row["Timestamp"]}!', icon="⚠️")
            aviso_emitido2 = True

    # Função para plotar gráfico de consumo e nível de água
   

#dicionario
paginas = {
    "index": parte_bemvindo,
    "grafico": parte_graficos,
    "previsao": parte_previsao,
    "aviso": parte_aviso,
}

if 'pagina' not in st.session_state:
    st.session_state.pagina = "index"

st.sidebar.title('Menu')
if st.sidebar.button("Bem Vindo"):
    st.session_state.pagina = "index"
if st.sidebar.button("Graficos"):
    st.session_state.pagina = "grafico"
if st.sidebar.button("Previsao/Anomalias"):
    st.session_state.pagina = "previsao"
if st.sidebar.button("Avisos"):
    st.session_state.pagina = "aviso"

pagina = st.session_state.pagina
if pagina == "previsao":
    parte_previsao(df)
elif pagina == "aviso":
    parte_aviso(df)
else:
    paginas[pagina]()