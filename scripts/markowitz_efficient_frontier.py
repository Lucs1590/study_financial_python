
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from datetime import datetime


start = datetime(2019, 1, 1)
acoes = ['PETR4.SA', 'VALE3.SA', 'MGLU3.SA', 'BIDI4.SA', 'ALPA4.SA']
dados = web.get_data_yahoo(acoes, start)['Adj Close']

dados.describe()

dados.tail()


retorno_diario = dados.pct_change()
retorno_anual = retorno_diario.mean() * 250

retorno_diario.tail()


cov_diaria = retorno_diario.cov()
cov_anual = cov_diaria * 250


retorno_carteira = []
peso_acoes = []
volatilidade_carteira = []
sharpe_ratio = []


numero_acoes = len(acoes)
numero_carteiras = 100_000

np.random.seed(101)

for cada_carteira in range(numero_carteiras):

    peso = np.random.random(numero_acoes)
    peso /= np.sum(peso)

    retorno = np.dot(peso, retorno_anual)

    volatilidade = np.sqrt(np.dot(peso.T, np.dot(cov_anual, peso)))

    sharpe = retorno / volatilidade

    sharpe_ratio.append(sharpe)
    retorno_carteira.append(retorno)
    volatilidade_carteira.append(volatilidade)
    peso_acoes.append(peso)


carteira = {'Retorno': retorno_carteira,
            'Volatilidade': volatilidade_carteira,
            'Sharpe Ratio': sharpe_ratio}

for contar, acao in enumerate(acoes):
    carteira[acao+' Peso'] = [Peso[contar] for Peso in peso_acoes]

df = pd.DataFrame(carteira)

colunas = ['Retorno', 'Volatilidade', 'Sharpe Ratio'] + \
    [acao+' Peso' for acao in acoes]
df = df[colunas]


plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatilidade', y='Retorno', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.xlabel('Volatilidade')
plt.ylabel('Retorno Esperado')
plt.title('Fronteira Eficiente de Markowitz')
plt.show()


menor_volatilidade = df['Volatilidade'].min()
maior_sharpe = df['Sharpe Ratio'].max()

carteira_sharpe = df.loc[df['Sharpe Ratio'] == maior_sharpe]
carteira_min_variancia = df.loc[df['Volatilidade'] == menor_volatilidade]


plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatilidade', y='Retorno', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.scatter(x=carteira_sharpe['Volatilidade'],
            y=carteira_sharpe['Retorno'], c='red', marker='o', s=200)
plt.scatter(x=carteira_min_variancia['Volatilidade'],
            y=carteira_min_variancia['Retorno'], c='blue', marker='o', s=200)
plt.xlabel('Volatilidade')
plt.ylabel('Retorno Esperado')
plt.title('Fronteira Eficiente de Markowitz')
plt.show()


print("Essa é a carteira de Mínima Variância:", '\n', carteira_min_variancia.T)
print('\n')
print("Essa é a carteira com maior Sharpe Ratio:", '\n', carteira_sharpe.T)
