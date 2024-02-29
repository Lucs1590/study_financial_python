from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf


def download_data(tickers, start_date, end_date):
    """
    Download adjusted close prices for the given tickers and date range.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data


def calculate_returns(data):
    """
    Calculate log returns from the adjusted close prices.
    """
    returns = np.log(data / data.shift(1))
    return returns


def calculate_statistics(returns):
    """
    Calculate mean, standard deviation, and other statistics from returns.
    """
    mean_returns = returns.mean()
    std_returns = returns.std()
    mean_annual_returns = mean_returns * 250
    std_annual_returns = std_returns * np.sqrt(250)
    return mean_returns, std_returns, mean_annual_returns, std_annual_returns


def calculate_portfolio_metrics(returns, weights):
    """
    Calculate portfolio variance and volatility.
    """
    cov_matrix = returns.cov() * 250
    pfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    pfolio_volatility = np.sqrt(pfolio_var)
    return pfolio_var, pfolio_volatility


def calculate_diversifiable_risk(returns, weights):
    """
    Calculate diversifiable risk of the portfolio.
    """
    asset_variances = returns.var() * 250
    diversifiable_risk = np.dot(weights**2, asset_variances)
    return diversifiable_risk


# Define tickers and date range
tickers = ['VGIR11.SA', 'BTCI11.SA']
start_date = '2023-01-01'
end_date = datetime.now()

# Download data
data = download_data(tickers, start_date, end_date)

# Calculate returns
returns = calculate_returns(data)

# Calculate statistics
mean_returns, std_returns, mean_annual_returns, std_annual_returns = calculate_statistics(
    returns
)

# Print statistics
for ticker in tickers:
    print(f'{ticker} mean: {mean_returns[ticker] * 100}')
    print(f'{ticker} mean annual: {mean_annual_returns[ticker] * 100}')
    print(f'{ticker} std: {std_returns[ticker] * 100}')
    print(f'{ticker} std annual: {std_annual_returns[ticker]}\n')

# Calculate portfolio metrics
# Equal weighting scheme dinamically
weights = np.full(len(tickers), 1 / len(tickers))
pfolio_var, pfolio_volatility = calculate_portfolio_metrics(returns, weights)
print(f'Portfolio Variance: {pfolio_var * 100}')
print(f'Portfolio Volatility: {pfolio_volatility * 100}')

# Calculate diversifiable risk
diversifiable_risk = calculate_diversifiable_risk(returns, weights)
print(f'Diversifiable Risk: {diversifiable_risk * 100}')
if diversifiable_risk > 0:
    non_diversifiable_risk = pfolio_var - diversifiable_risk
    print(f'Non-Diversifiable Risk: {non_diversifiable_risk * 100}')

print("""\n - If the diversifiable risk is greater than zero it means that the portfolio is not well diversified.
   In this case, the non-diversifiable risk is also calculated, which is the risk that cannot be diversified away.""")
print(""" - If the diversifiable risk is zero, it means that the portfolio is well diversified and the non-diversifiable risk is equal to the portfolio variance.""")
print(""" - The variance of the portfolio represents the total risk of the portfolio, which is the sum of the diversifiable and non-diversifiable risk.""")
print(""" - The volatility of the portfolio is the standard deviation of the portfolio returns, which is a measure of the total risk of the portfolio.""")


# Save data to a CSV file with the metrics
metrics = pd.DataFrame({
    'Mean': mean_returns,
    'Mean Annual': mean_annual_returns,
    'Std': std_returns,
    'Std Annual': std_annual_returns
})
metrics.to_csv('metrics.csv')
data.to_csv('data.csv')
