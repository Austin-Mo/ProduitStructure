from scipy.optimize import fsolve
from datetime import datetime
import pandas as pd
import numpy as np
from backend.models import Models
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.dates
import json
from dateutil.relativedelta import relativedelta

pd.options.mode.chained_assignment = None


def split_equity_info(equity_info):
    components = equity_info.split()
    ticker = components[0].replace("('", "")
    maturity_date = components[2]
    option_type = components[3][0]  # 'C' or 'P'
    strike = components[3][1:]
    return pd.Series([ticker, option_type, maturity_date, strike])


def read_bloomberg_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.transpose()
    df = df.rename(columns={df.columns[0]: 'Last_Price'})
    df = df.reset_index()
    df[['Ticker', 'Option_Type', 'Maturity_Date', 'Strike']] = df['index'].apply(split_equity_info)
    df = df.drop(columns=['index'])
    df['Option_Type'] = df['Option_Type'].map({'C': 'call', 'P': 'put'})
    df['Strike'] = df['Strike'].astype(float)
    df['Maturity_Date'] = pd.to_datetime(df['Maturity_Date'], format='%m/%d/%y')
    df_dict = {ticker: group.drop(columns='Ticker') for ticker, group in df.groupby('Ticker')}
    return df_dict


def error_function(volatility, market_price, strike, time_to_maturity, risk_free_rate, dividend_yield,
                   spot_price, option_type):
    model = Models(spot_price, strike, risk_free_rate, time_to_maturity, dividend_yield, volatility)
    if option_type == 'call':
        model_price = model.black_scholes('call')
    else:
        model_price = model.black_scholes('put')
    error = model_price - market_price
    return error


class Volatility:
    def __init__(self, stock, pricing_date, rate):
        self.spot_price = stock.spot_price
        self.tickers = stock.ticker.split()[0]
        self.pricing_date = datetime.strptime(pricing_date, "%Y%m%d")
        self.rate = rate
        self.dividend_yield = stock.dividend_yield
        self.data = self.calculate_volatility_surface()

    def calculate_volatility_surface(self):
        bloomberg_data = read_bloomberg_data('backend/data/option.json')
        option_data = bloomberg_data[self.tickers]
        option_data['Moneyness'] = option_data['Strike'].apply(lambda x: x / self.spot_price)

        calls, puts = self.filter_moneyness(option_data)

        for date in option_data['Maturity_Date'].unique():

            if isinstance(date, np.datetime64):
                date = date.astype('M8[ms]').astype(datetime)

            r = self.rate.interpolate_rate(date=date)
            d = self.dividend_yield

            volatility_surface = pd.DataFrame()

            # Calculate implied volatility for calls and puts
            calls['Implied_Volatility'] = calls.apply(lambda row: fsolve(error_function, 0.2, args=(
            row['Last_Price'], row['Strike'], (date - self.pricing_date).days / 365.0, r, d, self.spot_price, 'call'))[
                0], axis=1)
            puts['Implied_Volatility'] = puts.apply(lambda row: fsolve(error_function, 0.2, args=(
            row['Last_Price'], row['Strike'], (date - self.pricing_date).days / 365.0, r, d, self.spot_price, 'put'))[
                0], axis=1)

            # Concatenate calls and puts and add to the volatility surface DataFrame
            volatility_surface = pd.concat([volatility_surface, pd.concat([puts, calls])])

            # Pour l'interpolation
            def calculate_years(date):
                return (date - self.pricing_date).days / 365.0

            volatility_surface['Dates_In_Years'] = volatility_surface['Maturity_Date'].apply(calculate_years)

        # Filtre par maturité
        three_months_later = self.pricing_date + relativedelta(months=3)
        volatility_surface = volatility_surface[volatility_surface['Maturity_Date'] > three_months_later]

        # On filtre les extrêmes
        q_low = volatility_surface['Implied_Volatility'].quantile(0.10)
        q_high = volatility_surface['Implied_Volatility'].quantile(0.90)
        volatility_surface = volatility_surface[(volatility_surface['Implied_Volatility'] >= q_low) & (
                volatility_surface['Implied_Volatility'] <= q_high)]

        return volatility_surface

    def filter_moneyness(self, option_df, moneyness_range=[0.85, 1.15]):
        option_df = option_df.groupby('Maturity_Date').filter(lambda group: group['Moneyness'].max() > 1.1)
        calls = option_df[option_df['Option_Type'] == 'call']
        calls = calls[(calls['Strike'] > self.spot_price) &
                      (calls['Strike'] < moneyness_range[1] * self.spot_price)]
        puts = option_df[option_df['Option_Type'] == 'put']
        puts = puts[(puts['Strike'] > moneyness_range[0] * self.spot_price) &
                    (puts['Strike'] < self.spot_price)]
        return calls, puts
