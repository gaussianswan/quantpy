import pandas as pd

from scipy.stats import norm, t, cauchy, laplace
from dataclasses import dataclass

@dataclass
class TFit:

    df: float
    loc: float
    scale: float

@dataclass
class NormalFit:

    loc: float
    scale: float

@dataclass
class CauchyFit:

    loc: float
    scale: float

@dataclass
class LaplaceFit:
    loc: float
    scale: float


def fit_t_distribution(data: pd.Series) -> TFit:

    fit = t.fit(data = data)
    tfit = TFit(df = fit[0], loc = fit[2], scale = fit[1])
    return tfit

def fit_normal_distribution(data: pd.Series) -> NormalFit:

    fit = norm.fit(data = data)
    norm_fit = NormalFit(loc = fit[0], scale = fit[1])

    return norm_fit

def fit_cauchy_distributoin(data: pd.Series) -> CauchyFit:

    fit = cauchy.fit(data = data)
    cauchy_fit = CauchyFit(loc = fit[0], scale = fit[1])

    return cauchy_fit

def fit_laplace_distribution(data: pd.Series) -> LaplaceFit:

    fit = laplace.fit(data = data)
    laplace_fit = LaplaceFit(loc = fit[0], scale = fit[1])

    return laplace_fit





