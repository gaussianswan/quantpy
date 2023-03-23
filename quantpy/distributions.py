import pandas as pd
import numpy as np

from enum import Enum
from scipy.stats import norm, t, cauchy, laplace
from dataclasses import dataclass

class DistributionType(Enum):

    NORMAL = 'normal'
    T = 'T'
    LAPLACE = 'laplace'
    CAUCHY = 'cauchy'
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

def pdf_fit(data: pd.Series, num_points: int = 100, distribution: DistributionType = DistributionType.NORMAL) -> pd.Series:

    support = np.linspace(data.min(), data.max(), num_points)

    if distribution == DistributionType.NORMAL:
        normal_fit = fit_normal_distribution(data = data)
        pdf = norm.pdf(x = support, loc = normal_fit.loc, scale = normal_fit.scale)

    elif distribution == DistributionType.CAUCHY:
        cauchy_fit = fit_cauchy_distributoin(data = data)
        pdf = norm.pdf(x = support, loc = cauchy_fit.loc, scale = cauchy_fit.scale)

    elif distribution == DistributionType.LAPLACE:
        laplace_fit = fit_laplace_distribution(data = data)
        pdf = laplace.pdf(x = support, loc = laplace_fit.loc, scale = laplace_fit.scale)


    elif distribution == DistributionType.T:
        t_fit = fit_t_distribution(data = data)
        pdf = t.pdf(x = support, df = t_fit.df, loc = t_fit.loc, scale = t_fit.scale)

    return pd.Series(pdf, index = support)








