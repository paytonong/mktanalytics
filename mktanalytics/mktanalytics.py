import numpy as np
import pandas as pd
from scipy.stats import norm, percentileofscore
from tqdm import tqdm

def rv_cc_estimator(sample,n=22):
	"""
	Realized volatility close to close calculation. Returns a time series of the realized volatility.

	sample: series or dataframe of closing prices indexed by date
	n: sample size period for the volatility
	"""
	sample_clean = sample.dropna()
	returns = np.divide(sample_clean, sample_clean.shift(1))
	log_returns = np.log(returns)
	ann_log_returns = 252*np.power(log_returns,2)/n
	return 100 * np.sqrt(ann_log_returns.rolling(window=n,min_periods=n).sum())


def cc_estimator(sample,n=22,days=1):
	combined_rv = pd.Series()
	sample_clean = sample.dropna()
	for i in range(days):
		staggered_samples = sample_clean[i::days]
		returns = np.divide(staggered_samples, staggered_samples.shift(1))
		log_returns = np.log(returns)
		ann_log_returns = 252*np.power(log_returns,2)/n/days
		sample_rv = 100 * np.sqrt(ann_log_returns.rolling(window=n,min_periods=n).sum())
		combined_rv = pd.concat([combined_rv, sample_rv])
	return combined_rv.sort_index()


def calc_period_var(sample, return_period=22, lookback=66):
	"""
	A period return's move normalized. Calculated as the squared move (variance) scaled by the period

	sample: series or dataframe of closing prices indexed by date

	"""
	sample_clean = sample.dropna()
	lookback_ret = sample_clean.pct_change(periods=return_period)
	return (lookback_ret**2).rolling(window=lookback).mean() * 250 / return_period


def calc_var_ratio(sample, return_period=22, period_min=3, day_min=66):
	"""
	The variance ratio based on the normalized historical returns over a given rolling return period ratioed to the daily historical returns

	sample: series or dataframe of closing prices indexed by date
	return period: 	
	"""
	lookback = max(return_period * period_min, day_min)
	period_var = calc_period_var(sample, return_period=return_period, lookback=lookback)
	daily_var = calc_period_var(sample, return_period=1, lookback=lookback)
	return period_var / daily_var


def calc_lfev(sample, return_period=22, period_min=3, day_min=66):
	lookback = max(return_period * period_min, day_min)
	period_var = calc_period_var(sample, return_period=return_period, lookback=lookback)
	daily_var = calc_period_var(sample, return_period=1, lookback=lookback)
	return (np.sqrt(period_var) - np.sqrt(daily_var)) * 100


def move_volatility(prices, days=66):
	abs_move = (prices / prices.shift(days) - 1)
	high_low = (prices.rolling(days+1).max() - prices.rolling(days+1).min()) / prices.shift(days)
	return abs_move / high_low * np.abs(abs_move) * 100


def move_volatility_range(prices, days=66):
	abs_move = (prices / prices.shift(days) - 1)
	high_prices = prices.rolling(days+1).max()
	low_prices = prices.rolling(days+1).min()
	
	close_dist_high_low = ((high_prices - prices.shift(days)) + (low_prices - prices.shift(days))) / prices.shift(days)
	
	high_low = (high_prices - low_prices) / prices.shift(days)
	return close_dist_high_low * (0.5 * (np.abs(abs_move) + high_low)) / high_low * 100


def generate_returns_dict(prices, undl_list, return_days):
	returns = {}
	for u in undl_list:
		returns[u] = pd.DataFrame()
		for i in return_days:
			close_prices = prices[u, 'Close'].dropna()
			returns[u][i] = (close_prices / close_prices.shift(i) - 1) * 100
	return returns


def rolling_trend(prices, undl_list, return_days, smoothing=5):
	'''Determines the trend by blending the returns across different periods and smooths the results.'''
	avg_returns_dict = {}
	returns_summary = {}
	returns_dict = generate_returns_dict(prices, undl_list, return_days)
	for u in undl_list:
		avg_returns_dict[u] = pd.DataFrame()
		for i in return_days:
			avg_returns_dict[u][i] = returns_dict[u][i].dropna().rolling(smoothing).mean() / np.sqrt(i)
		avg_returns_dict[u]['Average'] = avg_returns_dict[u].dropna().mean(axis=1)
		if len(avg_returns_dict[u].dropna()) > 0:
			returns_summary[u] = avg_returns_dict[u]['Average'].dropna()[-1]
	returns_summary = pd.Series(returns_summary)
	return returns_summary, avg_returns_dict


def spot_stats(sample, n=260):
	"""
	Simple spot statistics returning the distance in % terms from the last spot to the max spot in the period, distance to min spot, and current percentile in min to max.

	sample: series or dataframe of closing prices
	n: historical lookback period.
	"""
	spot_window = sample.dropna()[-n:]
	percentile = percentileofscore(spot_window, spot_window[-1])
	high = spot_window.max()
	low = spot_window.min()
	max_pct = (high / spot_window[-1] - 1) * 100
	min_pct = (low / spot_window[-1] - 1) * 100
	return max_pct, min_pct, percentile


def past_spot_ranges(sample, n=22):
	'''Finds Returns the past n spot range based on max/min of the period'''
	sample_max = (sample['High'].rolling(n).max() / sample['Close'].shift(n) - 1) * 100
	sample_min = (sample['Low'].rolling(n).min() / sample['Close'].shift(n) - 1) * 100
	return pd.concat([abs(sample_max), abs(sample_min)], axis=1).max(axis=1) / np.sqrt(n)


def varvolbreakeven(var, vol):
	b = -1
	a = 1 / (2 * var)
	c = vol - var / 2
	breakeven1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
	breakeven2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
	return breakeven1, breakeven2


def zscore_calc(hist_data, live_data):
	return (live_data - hist_data.mean()) / hist_data.std()


def var_payout(realized,strike):
	return 0.5 * (realized**2 / strike - strike)


def vol_payout(realized, strike):
	return realized - strike






class BlackScholes:
	def __init__(self, s, k, r, q, vol, t, payoff):
		"""vol is expressed in %. eg enter 16v as 16, not 0.16"""
		self.s = s
		self.k = k
		self.r = r
		self.q = q
		self.vol = vol / 100
		self.t = t
		self.payoff = payoff
		
	def d1(self):
		return (np.log(self.s / self.k) + 
				(self.r - self.q + self.vol ** 2 / 2) * \
				self.t) / (self.vol * np.sqrt(self.t))

	def d2(self):
		return (np.log(self.s / self.k) + 
				(self.r - self.q - self.vol ** 2 / 2) * \
				self.t) / (self.vol * np.sqrt(self.t))

	def phi(self, x):
		return np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)

	def price(self):
		if self.payoff.lower() == 'put':
			return self.put_price()
		else:
			return self.call_price()
		
	def call_price(self):
		if self.t == 0:
			return 0
		return self.s * np.exp(-self.q * self.r) * norm.cdf(self.d1()) - np.exp(-self.r * self.t) * self.k * norm.cdf(self.d2())
	
	def put_price(self):
		if self.t == 0:
			return 0
		return np.exp(-self.r * self.t) * self.k * norm.cdf(-self.d2()) - self.s * np.exp(-self.q * self.r) * norm.cdf(-self.d1()) 
	
	def delta(self):
		if self.t == 0:
			return 0
		if self.payoff.lower() == 'put':
			return -np.exp(-self.q * self.r) * norm.cdf(-self.d1())
		else:
			return np.exp(-self.q * self.r) * norm.cdf(self.d1())
	
	def vega(self):
		if self.t == 0:
			return 0
		return self.s * np.exp(-self.q * self.r) * self.phi(self.d1()) * np.sqrt(self.t)
	
	def alt_vega(self):
		return self.k * np.exp(-self.r * self.t) * self.phi(self.d2()) * np.sqrt(self.t)
	
	def gamma(self):
		if self.t == 0:
			return 0
		return np.exp(-self.q * self.r) * self.phi(self.d1()) / (self.s * self.vol * np.sqrt(self.t))
	
	def theta(self):
		"""Price of the option one calendar day later compared to the price today, rather than black scholes theta"""
		t_decay = self.t - 1 / 365
		tomorrow_option = BlackScholes(self.s, self.k, self.r, self.q, self.vol * 100, t_decay, self.payoff)
		return tomorrow_option.price() - self.price()
	
	
def get_iv(mkt_price, payoff, s, k, r, q, t, start_guess=0.6):
	MAX_ITERATIONS = 100
	PRECISION = 1.0e-5
	
	vol = start_guess
	for i in range(0, MAX_ITERATIONS):
		option = BlackScholes(s, k, r, q, vol * 100, t, payoff)
		price = option.price()
		vega = option.vega()
		
		diff = mkt_price - price
		if abs(diff) < PRECISION:
			return vol * 100
		vol = vol + diff / vega
	return vol * 100