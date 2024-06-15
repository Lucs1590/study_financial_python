from time import sleep
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from scipy.stats import skew, kurtosis


def remove_sold_assets(df: pd.DataFrame) -> pd.DataFrame:
    """Removes assets that have been sold from the portfolio."""
    sold_df = df[df["buy_sell"] == "Venda"][["ticker", "amount"]] \
        .groupby("ticker").sum().reset_index()
    bought_df = df[df["buy_sell"] == "Compra"][["ticker", "amount"]] \
        .groupby("ticker").sum().reset_index()

    df = df.merge(sold_df, on="ticker", how="left", suffixes=("", "_sold"))
    df = df.merge(bought_df, on="ticker", how="left", suffixes=("", "_bought"))
    df["amount_sold"] = df["amount_sold"].fillna(0)
    df["amount_bought"] = df["amount_bought"].fillna(0)
    df["amount"] = df["amount_bought"] - df["amount_sold"]
    df = df.drop(columns=["amount_sold", "amount_bought"])
    df = df[df["amount"] > 0]
    df["amount"] = df["amount"].astype(int)
    df = df.drop(columns=["buy_sell"])

    return df


def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Downloads adjusted close prices for a given ticker and date range."""
    sleep(1)
    data = yf.download(ticker, start=start_date, end=end_date)["Adj Close"]
    return data


def calculate_log_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates log returns from adjusted close prices."""
    return np.log(data / data.shift(1))


def calculate_annualized_statistics(
    returns: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculates annualized mean, standard deviation, skewness and kurtosis from daily returns."""
    skew_returns = {}
    kurt_returns = {}
    mean_returns = returns.mean() * 250
    std_returns = returns.std() * np.sqrt(250)
    for ticker in returns.columns:
        skew_returns[ticker] = skew(returns[ticker].dropna())
        kurt_returns[ticker] = kurtosis(returns[ticker].dropna())

    return mean_returns, std_returns, skew_returns, kurt_returns


def calculate_portfolio_metrics(
    returns: pd.DataFrame, weights: np.ndarray
) -> Tuple[float, float, float]:
    """Calculates portfolio variance, volatility, and diversifiable risk."""
    cov_matrix = returns.cov() * 250
    portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_volatility = np.sqrt(portfolio_var)

    # Diversifiable risk (assuming equal weighting)
    asset_variances = returns.var() * 250
    diversifiable_risk = np.dot(weights**2, asset_variances)
    return portfolio_var, portfolio_volatility, diversifiable_risk


def print_asset_statistics(tickers, mean_returns, std_returns, skew_returns, kurt_returns):
    """Prints the statistics of each asset"""
    for ticker in tickers:
        print(f"\n{ticker} Statistics:")
        print(f"  - Mean Annual Return: {mean_returns[ticker]:.2%}")
        print(f"  - Annual Volatility: {std_returns[ticker]:.2%}")
        print(f"  - Skewness: {skew_returns[ticker]:.4f}")
        print(f"  - Kurtosis: {kurt_returns[ticker]:.4f}")


def print_portfolio_statistics(
    portfolio_return, portfolio_volatility, sharpe_ratio, portfolio_var
):
    """Prints portfolio performance metrics."""
    print("\nPortfolio Performance:")
    print(f"  - Annual Return: {portfolio_return:.2%}")
    print(f"  - Annual Volatility: {portfolio_volatility:.2%}")
    print(f"  - Variance: {portfolio_var:.4f}")
    print(f"  - Sharpe Ratio: {sharpe_ratio:.2f}")


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
    tickers = df_contributions["ticker"].unique().tolist()

    all_data = pd.DataFrame()

    for ticker in tickers:
        ticker_data = df_contributions[df_contributions["ticker"] == ticker]
        start_date = ticker_data["date"].min().strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        data = download_data(ticker, start_date, end_date)
        data.name = ticker
        all_data = pd.merge(all_data, data, left_index=True, right_index=True, how="outer")

    returns = calculate_log_returns(all_data)
    returns.to_csv("data/historical_data.csv")

    mean_returns, std_returns, skew_returns, kurt_returns = calculate_annualized_statistics(
        returns
    )

    # --- Portfolio Analysis ---
    # weights = np.full(len(tickers), 1 / len(tickers))  # Equal weights
    weights = df_contributions.set_index(
        "ticker")["amount"] / df_contributions["amount"].sum()
    weights = weights.groupby("ticker").sum().reindex(tickers).fillna(0).values
    portfolio_var, portfolio_volatility, diversifiable_risk = calculate_portfolio_metrics(
        returns,
        weights
    )

    portfolio_return = np.dot(weights, mean_returns)
    sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else float(
        "inf"
    )
    non_diversifiable_risk = portfolio_var - diversifiable_risk

    # --- Results Presentation ---
    print_asset_statistics(
        tickers,
        mean_returns,
        std_returns,
        skew_returns,
        kurt_returns
    )

    print_portfolio_statistics(
        portfolio_return,
        portfolio_volatility,
        sharpe_ratio,
        portfolio_var
    )

    print("Risk Analysis:")
    print(f"  - Diversifiable Risk: {diversifiable_risk:.2%}")
    print(f"  - Non-Diversifiable Risk: {non_diversifiable_risk:.2%}")
    print("Note: Values are annualized.")

    # --- Save statistics to CSV ---
    stats = pd.DataFrame({
        "Mean Annual Return": mean_returns,
        "Annual Volatility": std_returns,
        "Skewness": skew_returns,
        "Kurtosis": kurt_returns,
    })
    stats.to_csv(
        f"data/asset_statistics_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    )


if __name__ == "__main__":
    main()
