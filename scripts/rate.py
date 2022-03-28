import argparse
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as wb


def trs(_active):
    _active['simple_return'] = (
        _active['Adj Close'] / _active['Adj Close'].shift(1)) - 1
    print(
        f'From {to_br_date(_active["simple_return"].index[0])} to {to_br_date(_active["simple_return"].index[-1])}')
    _active['simple_return'].plot(figsize=(8, 5))
    plt.show()
    avg_return_d = _active['simple_return'].mean()
    avg_return_a = _active['simple_return'].mean() * 250
    return round(avg_return_a * 100, 3), round(avg_return_d * 100, 3)


def trl(_active):
    _active['log_return'] = np.log(
        _active['Adj Close'] / _active['Adj Close'].shift(1))
    print(
        f'From {to_br_date(_active["log_return"].index[0])} to {to_br_date(_active["log_return"].index[-1])}')
    _active['log_return'].plot(figsize=(8, 5))
    plt.show()
    log_return_d = _active['log_return'].mean()
    log_return_a = _active['log_return'].mean() * 250
    return round(log_return_a * 100, 3), round(log_return_d * 100, 3)


def to_br_date(data) -> str:
    data = str(data)
    return data[8:10] + '/' + data[5:7] + '/' + data[:4]


ap = argparse.ArgumentParser()
ap.add_argument("-a", "--active", required=True, help="Sequence")
ap.add_argument("-o", "--option", type=int, required=True,
                help="Options:\n 1-Simple Return Rate\n 2-Logarithmic Return Rate")
args = vars(ap.parse_args())

active = wb.DataReader(args['ativo'], data_source='yahoo', start='2020-3-11')

if args["option"] == 1:
    tr = trs(active)
    print(f' > {tr[0]}% by year\n > {tr[1]}% by day')
elif args["option"] == 2:
    tr = trl(active)
    print(f' > {tr[0]}% by year\n > {tr[1]}% by day')
else:
    print("Try to digite a valid option, in this case, 1 or 2")
