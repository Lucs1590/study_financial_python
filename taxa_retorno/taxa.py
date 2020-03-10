import argparse
import numpy as np
from pandas_datareader import data as wb
import matplotlib.pyplot as plt


def trs(acao):
    acao['simple_return'] = (
        acao['Adj Close'] / acao['Adj Close'].shift(1)) - 1
    print('De {} à {}'.format(
        data_BR(acao['simple_return'].index[0]), data_BR(acao['simple_return'].index[-1])))
    acao['simple_return'].plot(figsize=(8, 5))
    plt.show()
    avg_return_d = acao['simple_return'].mean()
    avg_return_a = acao['simple_return'].mean() * 250
    return round(avg_return_a * 100, 3), round(avg_return_d * 100, 3)


def trl(acao):
    acao['log_return'] = np.log(
        acao['Adj Close'] / acao['Adj Close'].shift(1))
    print('De {} à {}'.format(
        data_BR(acao['log_return'].index[0]), data_BR(acao['log_return'].index[-1])))
    acao['log_return'].plot(figsize=(8, 5))
    plt.show()
    log_return_d = acao['log_return'].mean()
    log_return_a = acao['log_return'].mean() * 250
    return round(log_return_a * 100, 3), round(log_return_d * 100, 3)


def data_BR(data):
    # 2018-05-11 00:00:00
    data = str(data)
    return data[8:10] + '/' + data[5:7] + '/' + data[:4]


ap = argparse.ArgumentParser()
ap.add_argument("-a", "--ativo", required=True, help="Sequence")
ap.add_argument("-o", "--opcao", type=int, required=True,
                help="Opções:\n 1-Taxa de retorno simples\n 2-Taxa de retorno logaritimaca")
args = vars(ap.parse_args())

acao = wb.DataReader(args['ativo'], data_source='yahoo', start='2019-2-11')

if (args["opcao"] == 1):
    tr = trs(acao)
    print(' - {}% ao ano\n - {}% ao dia'.format(tr[0], tr[1]))
elif (args["opcao"] == 2):
    tr = trl(acao)
    print(' - {}% ao ano\n - {}% ao dia'.format(tr[0], tr[1]))
else:
    print("Digite uma opcao valida!")
