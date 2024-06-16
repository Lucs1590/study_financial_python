from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def remove_sold_assets(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Removes assets that have been sold from the portfolio."""
    sold_df = dataframe[dataframe["buy_sell"] == "Venda"][["ticker", "amount"]] \
        .groupby("ticker").sum().reset_index()
    bought_df = dataframe[dataframe["buy_sell"] == "Compra"][["ticker", "amount"]] \
        .groupby("ticker").sum().reset_index()

    dataframe = dataframe.merge(
        sold_df, on="ticker", how="left", suffixes=("", "_sold"))
    dataframe = dataframe.merge(
        bought_df, on="ticker", how="left", suffixes=("", "_bought"))
    dataframe["amount_sold"] = dataframe["amount_sold"].fillna(0)
    dataframe["amount_bought"] = dataframe["amount_bought"].fillna(0)
    dataframe["amount"] = dataframe["amount_bought"] - dataframe["amount_sold"]
    dataframe = dataframe.drop(columns=["amount_sold", "amount_bought"])
    dataframe = dataframe[dataframe["amount"] > 0]
    dataframe["amount"] = dataframe["amount"].astype(int)
    dataframe = dataframe.drop(columns=["buy_sell"])

    return dataframe


def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Downloads adjusted close prices for a given ticker and date range."""
    # sleep(1)
    data = yf.download(ticker, start=start_date, end=end_date)["Adj Close"]
    return data


def calculate_log_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates log returns from adjusted close prices."""
    return np.log(data / data.shift(1))


def main():
    """Main function to calculate and print portfolio risk metrics."""
    # --- Defining parameters and downloading data ---
    df_contributions = pd.read_csv("data/contributions.csv")
    df_contributions["date"] = pd.to_datetime(
        df_contributions["date"],
        format="%d/%m/%Y"
    )
    df_contributions["ticker"] = df_contributions["ticker"].str.upper().apply(
        lambda x: f'{x}.SA'
    )

    df_contributions = remove_sold_assets(df_contributions)
    df_contributions = df_contributions[df_contributions["type"] == "Ações"]
    tickers = df_contributions["ticker"].unique().tolist()

    all_data = pd.DataFrame()

    for ticker in tickers:
        ticker_data = df_contributions[df_contributions["ticker"] == ticker]
        start_date = ticker_data["date"].min().strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        data = download_data(ticker, start_date, end_date)
        data.name = ticker
        all_data = pd.merge(
            all_data,
            data,
            left_index=True,
            right_index=True,
            how="outer"
        )

    daily_returns = calculate_log_returns(all_data)
    daily_covariance = daily_returns.cov()
    wallet_returns = []
    stock_weights = []
    wallet_volatilities = []
    sharpe_ratios = []

    number_stocks = len(tickers)
    number_wallets = 500_000

    np.random.seed(101)

    for wallet in tqdm(range(number_wallets)):
        weights = np.random.random(number_stocks)
        weights /= np.sum(weights)

        returns = np.dot(weights, daily_returns.mean() * 250)

        volatility = np.sqrt(
            np.dot(weights.T, np.dot(daily_covariance * 250, weights)))

        sharpe = returns / volatility

        sharpe_ratios.append(sharpe)
        wallet_returns.append(returns)
        wallet_volatilities.append(volatility)
        stock_weights.append(weights)

    wallet = {'Return': wallet_returns,
              'Volatility': wallet_volatilities,
              'Sharpe Ratio': sharpe_ratios}

    for count, stock in enumerate(tickers):
        wallet[stock+' Weight'] = [Weight[count] for Weight in stock_weights]

    df_wallet = pd.DataFrame(wallet)

    columns = ['Return', 'Volatility', 'Sharpe Ratio'] + \
        [stock+' Weight' for stock in tickers]

    df_wallet = df_wallet[columns]
    min_volatility = df_wallet['Volatility'].min()
    max_sharpe = df_wallet['Sharpe Ratio'].max()

    wallet_sharpe = df_wallet.loc[df_wallet['Sharpe Ratio'] == max_sharpe]
    wallet_min_variance = df_wallet.loc[df_wallet['Volatility']
                                        == min_volatility]

    df_wallet.plot.scatter(x='Volatility', y='Return', c='Sharpe Ratio',
                           cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
    plt.scatter(x=wallet_sharpe['Volatility'],
                y=wallet_sharpe['Return'], c='red', marker='o', s=200)
    plt.scatter(x=wallet_min_variance['Volatility'],
                y=wallet_min_variance['Return'], c='blue', marker='o', s=200)
    plt.xlabel('Volatility')
    plt.ylabel('Expected Return')
    plt.title('Markowitz Efficient Frontier')
    plt.savefig(
        f"images/markowitz_efficient_frontier{datetime.now().strftime('%Y%m%d-%H%M')}.png"
    )

    print("This is the Minimum Variance Portfolio:", '\n', wallet_min_variance.T)
    print('\n')
    print("This is the Maximum Sharpe Ratio Portfolio:", '\n', wallet_sharpe.T)


if __name__ == "__main__":
    main()
