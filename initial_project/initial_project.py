import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime
from scipy.stats import norm


"""
phi is the standard normal probability density function and 
Phi is the standard normal cumulative distribution function. Note that the gamma and vega formulas are the same for calls and puts.

S: current stock price 
K: strike price
r: risk-free interest rate
q: annual dividend yield
t: time to expiration in years
sigma: volatility
"""

phi = norm.pdf
Phi = norm.cdf


def d1(S, K, r, q, t, sigma):
    return (np.log(S / K) + (r - q + np.power(sigma, 2) / 2) * t) / (sigma * np.sqrt(t))


def d2(S, K, r, q, t, sigma):

    return d1(**locals()) - sigma * np.sqrt(t)


def fair_value(S, K, r, q, t, sigma):
    return S * np.exp(-q * t) * Phi(d1(**locals())) - np.exp(-r * t) * K * Phi(
        d2(**locals())
    )


def delta(S, K, r, q, t, sigma):
    return np.exp(-q * t) * Phi(d1(**locals()))


def gamma(S, K, r, q, t, sigma):
    return np.exp(-q * t) * ((phi(d1(**locals()))) / (S * sigma * np.sqrt(t)))


def theta(S, K, r, q, t, sigma):
    return (
        -np.exp(-q * t) * ((S * phi(d1(**locals())) * sigma) / (2 * np.sqrt(t)))
        - r * K * np.exp(-r * t) * Phi(d2(**locals()))
        + q * S * np.exp(-q * t) * Phi(d1(**locals()))
    ) * 0.01


def vega(S, K, r, q, t, sigma):
    return S * np.exp(-q * t) * phi(d1(**locals())) * np.sqrt(t) * 0.01


def rho(S, K, r, q, t, sigma):
    return K * t * np.exp(-r * t) * Phi(d2(**locals())) * 0.01


def volatility(S, K, r, q, t, market_price):
    max_iter = 1000
    sigma = 1
    for _ in range(max_iter):
        bs_price = fair_value(S, K, r, q, t, sigma)
        _vega = vega(S, K, r, q, t, sigma) * 100
        C = bs_price - market_price
        sigma -= C / _vega
    return sigma


TICKER = "SPY"
EXPIRATION = "2024-03-28"
RISK_FREE_INTEREST_RATE = 0.05375  # 1 month T-bills because expiration in 1 month

ticker = yf.Ticker(TICKER)
options = ticker.option_chain(date=EXPIRATION)
calls = pd.DataFrame(options.calls)
df = calls[["strike", "lastPrice"]].copy()

ymd = [int(x) for x in EXPIRATION.split("-")]
t = float((datetime(*ymd) - datetime.now()).days) / 365
S = ticker.info["open"]
try:
    q = ticker.info["yield"]
except:
    q = 0

r = RISK_FREE_INTEREST_RATE

df["implied volatility"] = df.apply(
    lambda row: volatility(S, row["strike"], r, q, t, row["lastPrice"]), axis=1
)
df["delta"] = df.apply(
    lambda row: delta(S, row["strike"], r, q, t, row["implied volatility"]), axis=1
)
df["gamma"] = df.apply(
    lambda row: gamma(S, row["strike"], r, q, t, row["implied volatility"]), axis=1
)
df["theta"] = df.apply(
    lambda row: theta(S, row["strike"], r, q, t, row["implied volatility"]), axis=1
)
df["vega"] = df.apply(
    lambda row: vega(S, row["strike"], r, q, t, row["implied volatility"]), axis=1
)
df["rho"] = df.apply(
    lambda row: rho(S, row["strike"], r, q, t, row["implied volatility"]), axis=1
)

df.to_csv("options_book.csv", index=False)
