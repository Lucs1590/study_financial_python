from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from scipy.stats import skew, kurtosis


def download_data(tickers: List[str], start_date: str, end_date: datetime) -> pd.DataFrame:
    """Downloads adjusted close prices for the given tickers and date range."""
    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
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
        skew_returns[ticker] = skew(returns[ticker].tolist())
        kurt_returns[ticker] = kurtosis(returns[ticker].tolist())

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
    tickers = ["VGIR11.SA", "BTCI11.SA", "RURA11.SA", "MXRF11.SA", "GALG11.SA"]
    start_date = "2023-06-01"
    end_date = datetime.now()

    data = download_data(tickers, start_date, end_date)
    returns = calculate_log_returns(data)
    data.to_csv("historical_data.csv")

    mean_returns, std_returns, skew_returns, kurt_returns = calculate_annualized_statistics(
        returns
    )

    # --- Portfolio Analysis ---
    weights = np.full(len(tickers), 1 / len(tickers))  # Equal weights
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
    stats = pd.DataFrame(
        {
            "Mean Annual Return": mean_returns,
            "Annual Volatility": std_returns,
            "Skewness": skew_returns,
            "Kurtosis": kurt_returns,
        }
    )
    stats.to_csv("asset_statistics.csv")


if __name__ == "__main__":
    main()
