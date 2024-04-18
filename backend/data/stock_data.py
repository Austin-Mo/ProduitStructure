import pandas as pd
import json
from backend.data.volatility import Volatility
from backend.data.rate_curve import ZeroCouponCurve


class StockData:
    def __init__(self, ticker, pricing_date):
        """
        Initialisation des données du sous-jacent.
        :param ticker: Ticker du sous-jacent.
        :param spot_price: Prix spot du sous-jacent.
        :param dividend_yield: Rendement du dividende annuel.
        """
        self.ticker = ticker
        self.spot_price = self.get_spot_price()
        self.dividend_yield = self.get_dividend_yield()
        self.rate_curve = ZeroCouponCurve(date=pricing_date)
        self.volatility_surface = Volatility(self, pricing_date, self.rate_curve)

    def get_dividend_yield(self):
        with open('backend/data/dividend_yield_data.json', 'r') as file:
            dividend_yield = json.load(file)
        df = pd.DataFrame(dividend_yield['data'], columns=dividend_yield['columns'], index=dividend_yield['index'])
        df.index = pd.to_datetime(df.index, unit='ms')
        ticker_dividend_yield = df.loc[:, self.ticker].iloc[0]
        return ticker_dividend_yield / 100

    def get_spot_price(self):
        with open('backend/data/spot_data.json', 'r') as file:
            spot_data = json.load(file)
        df = pd.DataFrame(spot_data['data'], columns=spot_data['columns'], index=spot_data['index'])
        df.index = pd.to_datetime(df.index, unit='ms')
        ticker_spot_data = df.loc[:, self.ticker].iloc[0]
        return ticker_spot_data
