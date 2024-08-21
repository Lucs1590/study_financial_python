from datetime import datetime

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

CPU_COUNT = max(1, cpu_count() - 1)


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
    data = yf.download(ticker, start=start_date, end=end_date)["Adj Close"]
    return data


def calculate_log_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates log returns from adjusted close prices."""
    return np.log(data / data.shift(1))


def calculate_portfolio_performance(weights: np.array, daily_returns: pd.DataFrame, daily_covariance: pd.DataFrame) -> tuple:
    """Calculates portfolio returns, volatility, and Sharpe ratio."""
    returns = np.dot(weights, daily_returns.mean() * 250)
    volatility = np.sqrt(
        np.dot(weights.T, np.dot(daily_covariance * 250, weights))
    )
    sharpe_ratio = returns / volatility if volatility != 0 else float("inf")
    return returns, volatility, sharpe_ratio


def analyze_portfolio(args):
    """Helper function to analyze a single portfolio."""
    weights, daily_returns, daily_covariance = args
    return calculate_portfolio_performance(weights, daily_returns, daily_covariance)


def generate_portfolios(
        number_stocks: int,
        number_wallets: int,
        daily_returns: pd.DataFrame,
        daily_covariance: pd.DataFrame,
        tickers: list
) -> pd.DataFrame:
    """Generates random portfolios and calculates their performance."""
    stock_weights = [np.random.random(number_stocks)
                     for _ in range(number_wallets)]
    stock_weights = [weights / np.sum(weights) for weights in stock_weights]

    print("Generating and analyzing portfolios...")

    # Using multiprocessing to parallelize the portfolio analysis
    with Pool(CPU_COUNT) as pool:
        results = list(
            tqdm(pool.imap(
                analyze_portfolio,
                [(weights, daily_returns, daily_covariance)
                 for weights in stock_weights]
            ), total=number_wallets)
        )

    wallet_returns, wallet_volatilities, sharpe_ratios = zip(*results)

    wallet = {
        "Return": wallet_returns,
        "Volatility": wallet_volatilities,
        "Sharpe Ratio": sharpe_ratios,
    }

    for count, stock in enumerate(tickers):
        wallet[stock + " Weight"] = [
            weights[count]
            for weights in stock_weights
        ]

    df_wallet = pd.DataFrame(wallet)
    columns = ["Return", "Volatility", "Sharpe Ratio"] + [
        stock + " Weight" for stock in tickers
    ]
    df_wallet = df_wallet[columns]

    return df_wallet


def find_optimal_portfolios(df_wallet: pd.DataFrame) -> tuple:
    """Finds the portfolios with minimum volatility and maximum Sharpe ratio."""
    min_volatility = df_wallet["Volatility"].min()
    max_sharpe = df_wallet["Sharpe Ratio"].max()

    wallet_sharpe = df_wallet.loc[df_wallet["Sharpe Ratio"] == max_sharpe]
    wallet_min_variance = df_wallet.loc[df_wallet["Volatility"]
                                        == min_volatility]

    return wallet_sharpe, wallet_min_variance


def plot_efficient_frontier(df_wallet: pd.DataFrame, optimal_portfolios: tuple) -> None:
    """Plots the efficient frontier and highlights optimal portfolios."""
    wallet_sharpe, wallet_min_variance = optimal_portfolios

    plt.figure(figsize=(10, 8))
    plt.scatter(
        df_wallet["Volatility"],
        df_wallet["Return"],
        c=df_wallet["Sharpe Ratio"],
        cmap="RdYlGn",
        edgecolors="black",
    )
    plt.scatter(
        wallet_sharpe["Volatility"],
        wallet_sharpe["Return"],
        c="red",
        marker="o",
        s=200
    )
    plt.scatter(
        wallet_min_variance["Volatility"],
        wallet_min_variance["Return"],
        c="blue",
        marker="o",
        s=200,
    )
    plt.xlabel("Volatility (Annualized)")
    plt.ylabel("Expected Return (Annualized)")
    plt.title("Markowitz Efficient Frontier")
    plt.colorbar(label="Sharpe Ratio")
    plt.grid(True)
    plt.show()

    print("Minimum Variance Portfolio:\n", wallet_min_variance.T)
    print("\nMaximum Sharpe Ratio Portfolio:\n", wallet_sharpe.T)


def main():
    """Main function to calculate and print portfolio risk metrics."""
    df_contributions = pd.read_csv("data/contributions.csv")
    df_contributions["date"] = pd.to_datetime(
        df_contributions["date"],
        format="%d/%m/%Y"
    )
    df_contributions["ticker"] = df_contributions["ticker"].str.upper().apply(
        lambda x: f'{x}.SA'
    )

    df_contributions = remove_sold_assets(df_contributions)
    df_contributions = df_contributions[
        df_contributions["type"] == "A\u00e7\u00f5es"
    ]
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

    number_stocks = len(tickers)
    number_wallets = 100_000

    df_wallet = generate_portfolios(
        number_stocks, number_wallets, daily_returns, daily_covariance, tickers
    )
    optimal_portfolios = find_optimal_portfolios(df_wallet)
    plot_efficient_frontier(df_wallet, optimal_portfolios)


if __name__ == "__main__":
    main()
