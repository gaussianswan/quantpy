import numpy as np
import pandas as pd

def log_returns(series: pd.Series, dropna: bool = True) -> pd.Series:

    log_value = series.apply(np.log)
    log_reg = log_value.diff()

    if dropna:
        return log_reg.dropna()
    else:
        return log_reg


def simple_returns(series: pd.Series, lag: int = 1, dropna: bool = True) -> pd.Series:

    pct_change = series.pct_change(periods = lag)

    if dropna:
        return pct_change.dropna()
    else:
        return pct_change



