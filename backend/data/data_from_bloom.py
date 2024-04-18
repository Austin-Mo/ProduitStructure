# Pour utiliser xbbg sur les postes de l'école voir : https://pypi.org/project/xbbg/
# pip install blpapi --index-url=https://bcms.bloomberg.com/pip/simple/
# Puis
# pip install xbbg

from xbbg import blp
from datetime import datetime, timedelta


def get_data_rate():
    tickers_us = ['S0023Z 1D BLC2 Curncy', 'S0023Z 1W BLC2 Curncy', 'S0023Z 1M BLC2 Curncy', 'S0023Z 3M BLC2 Curncy',
                  'S0023Z 6M BLC2 Curncy', 'S0023Z 9M BLC2 Curncy', 'S0023Z 1Y BLC2 Curncy', 'S0023Z 2Y BLC2 Curncy',
                  'S0023Z 3Y BLC2 Curncy', 'S0023Z 5Y BLC2 Curncy', 'S0023Z 7Y BLC2 Curncy', 'S0023Z 10Y BLC2 Curncy',
                  'S0023Z 15Y BLC2 Curncy', 'S0023Z 20Y BLC2 Curncy', 'S0023Z 25Y BLC2 Curncy',
                  'S0023Z 30Y BLC2 Curncy']

    # Récupérer les données de Bloomberg
    data = blp.bdh(tickers_us, ['Last_Price'], start_date='20240301', end_date='20240301')

    # Convertir les données en JSON
    data_json = data.to_json()

    # Enregistrer le JSON sur votre PC
    with open('rate.json', 'w') as f:
        f.write(data_json)


def get_data_options():
    tickers = ['AAPL US Equity', 'MSFT US Equity', 'GOOGL US Equity']

    # Récupérer les données de Bloomberg
    data = blp.bds(tickers, ['OPT_CHAIN'], END_DT='20340301')
    data = blp.bdh(data['security_description'], ['Last_Price'], start_date='20240301', end_date='20240301')
    # Convertir les données en JSON
    data_json = data.to_json()

    # Enregistrer le JSON sur votre PC
    with open('option.json', 'w') as f:
        f.write(data_json)


def get_data_stock():
    # Définir les tickers et les champs
    tickers = ['AAPL US Equity', 'MSFT US Equity', 'GOOGL US Equity']
    fields = ['Last_Price', 'Dividend_Yield']
    # Récupérer les données de Bloomberg
    data = blp.bdh(tickers, fields, start_date='20240301', end_date='20240301')
    # Extraire les données de spot et de dividende yield
    spot_data = data.xs('Last_Price', level=1, axis=1)
    dividend_yield_data = data.xs('Dividend_Yield', level=1, axis=1)

    # Matrtice de corrélation
    start_date_correl = datetime.strptime('20240301', '%Y%m%d') - timedelta(days=3 * 365)
    # Récupérer les données de spot pour les 3 dernières années
    data = blp.bdh(tickers, ['Last_Price'], start_date=start_date_correl.strftime('%Y%m%d'), end_date='20240301')
    # Calculer la matrice de corrélation
    correlation_matrix = data.corr()

    # Convertir les DataFrames en JSON
    correlation_matrix_json = correlation_matrix.to_json(orient='split')
    dividend_yield_data_json = dividend_yield_data.to_json(orient='split')
    spot_data_json = spot_data.to_json(orient='split')

    # Enregistrer les chaînes JSON dans des fichiers
    with open('correlation_matrix.json', 'w') as f:
        f.write(correlation_matrix_json)

    with open('dividend_yield_data.json', 'w') as f:
        f.write(dividend_yield_data_json)

    with open('spot_data.json', 'w') as f:
        f.write(spot_data_json)


get_data_rate()
get_data_stock()
get_data_options()
