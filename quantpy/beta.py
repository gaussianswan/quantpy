import pandas as pd
import numpy as np

def compute_beta(series_x: pd.Series, series_y: pd.Series, y_on_x: bool = True) -> float:

    # Ensuring that the indices are going to match up
    df = pd.concat([series_x, series_y], axis = 1).dropna()

    if y_on_x:
        y_col = 1
        x_col = 0
    else:
        y_col = 0
        x_col = 1

    y = df.iloc[:, y_col]
    x = df.iloc[:, x_col]

    covariance = y.cov(other = x)
    variance = x.var()

    beta = covariance / variance

    return beta

def compute_rolling_beta(series_x: pd.Series, series_y: pd.Series, window: int = 90, y_on_x: bool = True) -> pd.Series:
    assert series_x.shape[0] > window, f"The series_x you pass has to have more data points than the window size. The current size is {series_x.shape[0]}. The window size is {window}"
    assert series_y.shape[0] > window, f"The series_y you pass has to have more data points than the window size. The current size is {series_y.shape[0]}. The window size is {window}"
    df = pd.concat([series_x, series_y], axis = 1).dropna()

    if y_on_x:
        y_col = 1
        x_col = 0
    else:
        y_col = 0
        x_col = 1

    y = df.iloc[:, y_col]
    x = df.iloc[:, x_col]

    rolling_correlation = y.rolling(window=window).corr(x)

    y_rolling_vol = y.rolling(window).std()
    x_rolling_vol = x.rolling(window).std()
    rolling_covariance = rolling_correlation * y_rolling_vol * x_rolling_vol
    rolling_beta = rolling_covariance / (x_rolling_vol ** 0.5)

    return rolling_beta

def compute_rolling_ewm_beta(series_x: pd.Series, series_y: pd.Series, halflife: int = 40, min_periods: int = 0, y_on_x: bool = True) -> pd.Series:

    df = pd.concat([series_x, series_y], axis = 1).dropna()

    if y_on_x:
        y_col = 1
        x_col = 0
    else:
        y_col = 0
        x_col = 1

    y = df.iloc[:, y_col]
    x = df.iloc[:, x_col]

    ewm_rolling_covariance = y.ewm(halflife = halflife, min_periods=min_periods).cov(x)
    x_ewm_vol = x.ewm(halflife=halflife, min_periods=min_periods).std()

    ewm_rolling_beta = ewm_rolling_covariance / (x_ewm_vol ** 2)

    return ewm_rolling_beta










