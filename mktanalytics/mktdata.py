import numpy as np
import pandas as pd
import yfinance as yf
from yahoo_earnings_calendar import YahooEarningsCalendar
import mktanalytics as ma
from tqdm.notebook import tqdm

def nearest(items, pivot):
	return min(items, key=lambda x: abs(x - pivot))

def get_atm_vol(undl_list, weeks=1, calc_strangle=False, target_price=1):
	atm_dict = {}
	date_dict = {}
	volume_dict = {}
	no_options_list = []
	if calc_strangle:
		target_put = {}
		target_call = {}
	for u in tqdm(undl_list):
		try:
			ticker = yf.Ticker(u)
			option_expiries = list(ticker.options)
			option_expiries = [pd.Timestamp(x) for x in option_expiries]
			target_date = pd.Timestamp.now() + pd.DateOffset(weeks=weeks)
			closest_expiry = nearest(option_expiries, target_date)
			options = ticker.option_chain(closest_expiry.strftime('%Y-%m-%d'))
			put_atm = options.puts[-options.puts['inTheMoney']]['impliedVolatility'].iloc[-1]
			call_atm = options.calls[-options.calls['inTheMoney']]['impliedVolatility'].iloc[0]
			atm_dict[u] = (put_atm + call_atm) / 2 * 100
			date_dict[u] = closest_expiry
			volume_dict[u] = options.puts['volume'].sum() + options.calls['volume'].sum()

			if calc_strangle:
				spot_price = ticker.history()['Close'][-1]
				target_put[u], target_call[u] = get_target_strangle(options, spot_price, target_price)
		except IndexError:
			no_options_list.append(u)
	if len(no_options_list) > 0:
		for u in no_options_list:
			print('No options for {}'.format(u))
	if calc_strangle:
		return atm_dict, date_dict, volume_dict, target_put, target_call
	return atm_dict, date_dict, volume_dict


def get_target_strangle(options, spot_price, target_price, steps=10):
	puts = options.puts[-options.puts['inTheMoney']].copy()
	calls = options.calls[-options.calls['inTheMoney']].copy()
	strangle_prices = []
	option_list = [len(puts), len(calls), steps]
	steps = np.min(option_list)
	for i in range(steps):
		put = puts.iloc[-(i+1)]
		call = calls.iloc[i]
		put_price = put['lastPrice']
		call_price = call['lastPrice']
		strangle_price = put_price + call_price
		strangle_price_pct = (strangle_price / spot_price) * 100
		strangle_prices.append(strangle_price_pct)
	idx = min(range(len(strangle_prices)), key=lambda i: abs(strangle_prices[i] - 1))
	target_put = puts.iloc[-(idx+1)].copy()
	target_call = calls.iloc[idx].copy()
	target_call['strike_otm_pct'] = (target_call['strike'] / spot_price - target_price) * 100
	target_put['strike_otm_pct'] = (1 - target_put['strike'] / spot_price) * 100
	target_call['price_pct'] = target_call['lastPrice'] / spot_price * 100
	target_put['price_pct'] = target_put['lastPrice'] / spot_price * 100
	return target_put, target_call


def get_historical_data(ticker_list, period='1y'):
	ticker_list_str = ' '.join(spx_tickers_list)
	data = yf.download(ticker_list_str, period=period, group_by='ticker', auto_adjust=True, thread=True)
	return data


def get_rv_historical(data, undl_list, periods = [5, 10, 22]):
	rv_dict = {}
	for u in tqdm(undl_list):
		rv_df = pd.DataFrame()
		for period in periods:
			rv_df['{}D RV'.format(period)] = ma.rv_cc_estimator(data[u]['Close'], n=period, days=1).dropna()
		rv_df['Average RV'] = rv_df.mean(axis=1)
		rv_dict[u] = rv_df
	return rv_dict


def get_rv_vol(data, undl_list, periods = [5, 10, 22]):
	rv_dict = {}
	for u in tqdm(undl_list):
		rv_df = pd.DataFrame()
		for period in periods:
			rv_df['{}D RV'.format(period)] = ma.cc_estimator(data[u]['Close'], n=period, days=1).dropna()
		rv_df['Average RV'] = rv_df.mean(axis=1)
		rv_dict[u] = rv_df
	return rv_dict


def get_rv_latest(data, undl_list, periods = [5, 10, 22]):
	rv_dict = {}
	max_period = np.max(periods)
	for u in tqdm(undl_list):
		latest_data = data[u]['Close'][-(max_period+5):].dropna()
		if len(latest_data.dropna()) < max_period:
			continue
		rv_df = pd.DataFrame()
		for period in periods:
			rv_df['{}D RV'.format(period)] = ma.rv_cc_estimator(latest_data, n=period).dropna()
		rv_df['Average RV'] = rv_df.mean(axis=1)
		rv_dict[u] = rv_df.iloc[-1]
	return rv_dict


def get_yf_earnings_dates(undl_list):
	earnings_dates = {}
	for u in tqdm(undl_list):
		ticker = yf.Ticker(u)
		try:
			cal_df = ticker.calendar.transpose()
			if len(cal_df) > 0:
				earnings_dates[u] = cal_df['Earnings Date'].iloc[0]
			else:
				print('Could not get {} earnings date'.format(u))
		except:
			print('Could not get {} earnings date'.format(u))
	return earnings_dates


def get_earnings_calendar(start_date, end_date):
	yec = YahooEarningsCalendar()
	print('Getting earnings calendar between {} - {}'.format(start_date, end_date))
	all_earnings_data = yec.earnings_between(start_date, end_date)
	all_earnings_df = pd.DataFrame(all_earnings_data)
	all_earnings_df = all_earnings_df.set_index('ticker')
	return all_earnings_df


def get_earnings_date(earnings_df, undl):
	if undl in earnings_df.index:
		earnings_data = earnings_df.loc[undl]
		if type(earnings_data) is pd.DataFrame:
			earnings_data = earnings_data.iloc[0]
		return pd.Timestamp(earnings_data['startdatetime']).tz_localize(None)
	return np.nan