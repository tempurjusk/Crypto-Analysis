import pandas as pd
import numpy as np

def compute_drawdown(return_series: pd.Series):
    '''
    Takes a time series of asset returns
    Computes and returns a drawdown that contains
    the wealth index
    the previous peaks
    percent drawdowns
    '''
    wealth_index = 1000*(1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
#     Checks if the first month return is negative. If it is, replaces previous_peaks to the invested value of 1000
    if(previous_peaks.loc[previous_peaks.first_valid_index()]) < 1000:
        previous_peaks.loc[previous_peaks.first_valid_index()] = 1000
#     
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        'Wealth':wealth_index,
        'Peaks':previous_peaks,
        'Drawdowns': drawdowns
    })

def compute_max_drawdown(return_series: pd.Series):
    return compute_drawdown(return_series).Drawdowns.min()

def get_ffme_returns():
    '''
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    '''
    me_m = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv', parse_dates = True, header = 0, index_col = 0, na_values = -99.99)
    rets = me_m[['Lo 20', 'Hi 20']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format = '%Y%m').to_period('M')
    return rets

def get_hfi_returns():
    '''
    Load and format the hedge fund index returns
    '''
    hfi = pd.read_csv('data/edhec-hedgefundindices.csv', parse_dates = True, header = 0, index_col = 0, na_values = -99.99)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def get_ind_returns():
    '''
    Load and format the Ken French 30 Industry portfolios value weighted monthly returns
    '''
    dataset = pd.read_csv('data/ind30_m_vw_rets.csv', index_col = 0, parse_dates = True, header = 0)
    dataset.index = pd.to_datetime(dataset.index, format = '%Y%m').to_period('M')
    dataset = dataset/100
    # dataset.columns = [re.sub('\s','',x) for x in dataset.columns]
    # This above line is equivalent to this
    dataset.columns = dataset.columns.str.strip()
    return dataset

def skewness(r):
    '''
    Alternative to scipy.stats.skew()
    Copmutes the skewness of the supplied series or dataframe
    Returns a float or a series
    skewness = Exp[(R - E(R))^3]/Std_dev(R)^3
    '''
    demeaned_r = r - r.mean()
#     Using the population std, so df = 0
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r**3).mean()
    skew = exp/sigma_r**3
    return skew

def kurtosis(r):
    '''
    Alternative to scipy.stats.kurtosis()
    Copmutes the kurtosis of the supplied series or dataframe
    Returns a float or a series
    Kurtosis = Exp[(R - E(R))^4]/Std_dev(R)^4
    '''
    demeaned_r = r - r.mean()
#     Using the population std, so df = 0
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r**4).mean()
    kurt = exp/sigma_r**4
    return kurt

import scipy.stats
def is_normal(r, level=0.01):
    '''
    Apples the Jarque-Bera test to determine if a series is normal or not
    Test is applied at the 1% level by default, Significance level is 1%
    Null hypothesis is the series is normal
    Returns True if the hypothesis of normality is accepted, False otherwise
    '''
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value>level

def semideviation(r):
    '''
    Returns the semidivation aka negative semideviation of r
    r must be a series or Pandas dataframe
    '''
    semi_deviation = r[r<0].std(ddof = 0)
    return semi_deviation

def var_historic(r, level = 5):
    '''
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that 'level' percent of the returns
    fall below that number, and the (100 minus level) percent at above
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        return TypeError('Expected r to be a Series or DataFrame')

from scipy.stats import norm
def var_gaussian(r, level=5, modified = False):
    '''
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If 'modified' is set to True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    '''
    # Compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # Modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 - 3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    var = -(r.mean() + z * r.std(ddof = 0))
    return var

def cvar_historic(r, level = 5):
    '''
    Computes the conditional VaR of a Series or DataFrame
    '''
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        return TypeError('Expected r to be a Series or Dataframe')

def annualised_vol(r, periods_per_year):
    '''
    Annualises the volatility of a set of returns
    Takes a pandas dataframe or series and Periods per year as input
    '''
    return r.std()*np.sqrt(periods_per_year)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    '''
    Computes the annualised sharpe ratio of a set of returns
    Takes as input a pandas Series or DataFrame, Risk-free rate (annual)
    and number of periods per year as inputs
    '''
    rf_per_period = (1 + riskfree_rate)**(1/periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_excess_ret = annualised_ret(excess_ret, periods_per_year)
    ann_vol = annualised_vol(r, periods_per_year)
    sharpe_ratio = ann_excess_ret/ann_vol
    return sharpe_ratio

def annualised_ret(r, periods_per_year):
    '''
    Annualises a set of returns
    Takes as input returns as Pandas series or Dataframe
    and the number of periods per year
    '''
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    annualised_ret = compounded_growth ** (periods_per_year/n_periods) - 1
    return annualised_ret