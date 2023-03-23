import pandas as pd
import numpy as np

from typing import Literal
from arch import arch_model

def garch_conditional_vol(series: pd.Series, p: int = 1, q: int = 1, distribution: Literal['Normal', 'StudentsT'] = 'Normal') -> pd.Series:

    am = arch_model(y = series, p = p, q = q, dist=distribution)
    fit_result = am.fit(disp='off')
    return fit_result.conditional_volatility

def ewm_conditional_vol(series: pd.Series, halflife: int = 40, min_periods: int = 100) -> pd.Series:
    return series.ewm(halflife=halflife, min_periods=min_periods).std()

def rolling_conditional_vol(series: pd.Series, window: int = 90) -> pd.Series:
    return series.rolling(window=window).std()

