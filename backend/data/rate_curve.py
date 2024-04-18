import pandas as pd
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class ZeroCouponCurve:
    def __init__(self, date='20240301'):
        """
        Initialisation avec les tickers des bons du Trésor américain pour différentes maturités.
        """
        self.date = date
        self.data = self.get_data_from_json()

    def get_data_from_json(self):
        """
        Récupère les données de Bloomberg pour les bons du Trésor américain.
        """
        df = pd.read_json('backend/data/rate.json')

        start_date = datetime.strptime(self.date, '%Y%m%d')

        new_columns = []

        for col in df.columns:
            # Extraire la partie numérique et l'unité de temps
            match = re.match(r"\('S0023Z (\d+)([DWMY]) BLC2 Curncy', 'Last_Price'\)", col)
            if match:
                number, unit = match.groups()
                number = int(number)

                # Calculer la nouvelle date en fonction de l'unité de temps
                if unit == 'D':
                    new_date = start_date + timedelta(days=number)
                elif unit == 'W':
                    new_date = start_date + timedelta(weeks=number)
                elif unit == 'M':
                    new_date = start_date + pd.DateOffset(months=number)
                elif unit == 'Y':
                    new_date = start_date + pd.DateOffset(years=number)

                new_columns.append(new_date)
            else:
                new_columns.append(col)

        # Remplacer les colonnes du DataFrame
        df.columns = new_columns
        df = df.T
        df.rename(columns={df.columns[0]: 'rates'}, inplace=True)

        def date_to_years(date):
            return (date - datetime.strptime(self.date, "%Y%m%d")).days / 365.0
        df['maturity_in_years'] = df.index.map(date_to_years)

        return df/100

    def interpolate_rate(self, date):
        """
        Interpole la courbe des taux pour une date cible.
        """
        # Convertir la date en année
        days = (date - datetime.strptime(self.date, '%Y%m%d')).days
        date_in_year = days / 365.0

        # Créer une fonction d'interpolation
        interp_func = interp1d(self.data['maturity_in_years'], self.data['rates'], kind='linear', fill_value='extrapolate')

        # Utiliser la fonction d'interpolation pour obtenir le taux à la date cible
        interpolated_rate = interp_func(date_in_year).tolist()

        return interpolated_rate