import numpy as np
from datetime import datetime
from scipy.interpolate import NearestNDInterpolator, interp1d
from backend.data.correlation import get_correlation
import pandas as pd


class MonteCarlo:
    def __init__(self, stocks, start_date, end_date, num_simu=10000, day_conv=360, seed=0,
                 observation_frequency='monthly'):
        """
        Initialisation avec prise en compte de la fréquence d'observation.
        """
        self.stocks = stocks
        self.spots = np.array([stock.spot_price for stock in stocks])
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.maturity = (self.end_date - self.start_date).days / day_conv
        self.dividend_yields = np.array([stock.dividend_yield for stock in stocks])
        self.correlation_matrix = get_correlation([stock.ticker for stock in stocks])
        self.num_simu = num_simu
        self.day_conv = day_conv
        self.num_time_steps = int(self.maturity * day_conv)
        self.delta_t = self.maturity / day_conv
        self.seed = seed

        self.simulation_dates = pd.date_range(start=self.start_date, end=self.end_date).normalize()
        self.num_steps = None
        self.observation_frequency = observation_frequency
        self.observation_dates = self.generate_observation_dates()

        self.generate_correlated_shocks()
        self.simulations = self.simulate_correlated_prices()
        self.stocks_nb = len(self.simulations)

    def generate_observation_dates(self):
        """
        Génère les dates d'observations basées sur la fréquence et ajuste selon les jours ouvrables.
        """
        if self.observation_frequency == 'monthly':
            freq = 'BM'
        elif self.observation_frequency == 'quarterly':
            freq = 'BQ'
        elif self.observation_frequency == 'semiannually':
            freq = 'BQ-FEB,AUG'
        elif self.observation_frequency == 'annually':
            freq = 'BA'
        else:
            raise ValueError("Fréquence d'observation non reconnue.")

        # Générer les dates d'observations
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq=freq).normalize()

        return dates

    def generate_correlated_shocks(self):
        """
        Génère des chocs corrélés pour tous les sous-jacents en utilisant la décomposition de Cholesky.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        L = np.linalg.cholesky(self.correlation_matrix)
        z_uncorrelated = np.random.normal(0.0, 1.0,
                                          (self.num_time_steps, self.num_simu, len(self.spots))) * self.delta_t ** 0.5
        self.z = np.einsum('ij, tkj -> tki', L, z_uncorrelated)

    def simulate_correlated_prices(self):
        """
        Simule les chemins de prix pour tous les sous-jacents en utilisant les chocs corrélés.
        """
        dt = self.delta_t
        simu = np.zeros((self.num_time_steps + 1, self.num_simu, len(self.spots)))
        simu[0, :, :] = self.spots

        # Create an interpolation function for the volatility and rate of each stock
        volatilities = []
        for stock in self.stocks:
            # Prepare the data for interpolation
            x = stock.volatility_surface.data['Dates_In_Years']
            y = stock.volatility_surface.data['Strike']
            z = stock.volatility_surface.data['Implied_Volatility']
            points = np.array([x, y]).T

            # Create the interpolator
            volatilities.append(NearestNDInterpolator(points, z))

        rates = [interp1d(stock.rate_curve.data['maturity_in_years'], stock.rate_curve.data['rates'],
                          fill_value="extrapolate") for stock in self.stocks]

        for t in range(1, self.num_time_steps + 1):
            t_in_years = t / self.day_conv
            for i in range(len(self.stocks)):
                volatility = volatilities[i]((t_in_years, simu[t - 1, :, i]))
                rate = rates[i](t_in_years)
                simu[t, :, i] = simu[t - 1, :, i] * np.exp(
                    (rate - self.dividend_yields[i] - 0.5 * volatility ** 2) * dt + volatility * self.z[t - 1, :, i])

        dataframes = []
        for asset_index in range(simu.shape[2]):
            asset_data = simu[:, :, asset_index]
            df = pd.DataFrame(asset_data, index=self.simulation_dates,
                              columns=[f'{sim + 1}' for sim in range(self.num_simu)])
            dataframes.append(df)

        return dataframes
