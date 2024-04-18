import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import matplotlib.dates as mdates
from scipy.interpolate import griddata
import numpy as np
import plotly.graph_objects as go


def plot_volatility_surface_streamlit(stocks_list):
    for stock in stocks_list:
        volatility_df = stock.volatility_surface.data.copy()
        volatility_df['Maturity_Date'] = pd.to_datetime(volatility_df['Maturity_Date'])
        volatility_df['Dates_In_Years'] = volatility_df['Dates_In_Years'].astype(float)
        volatility_df['Strike'] = volatility_df['Strike'].astype(float)

        x = np.linspace(volatility_df['Dates_In_Years'].min(), volatility_df['Dates_In_Years'].max(),
                        len(volatility_df['Dates_In_Years'].unique()))
        y = np.linspace(volatility_df['Strike'].min(), volatility_df['Strike'].max(),
                        len(volatility_df['Strike'].unique()))
        X, Y = np.meshgrid(x, y)
        Z = griddata((volatility_df['Dates_In_Years'], volatility_df['Strike']),
                     volatility_df['Implied_Volatility'], (X, Y), method='cubic')

        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

        fig.update_layout(title=f'Implied Volatility Surface for {stock.ticker}', autosize=True,
                          scene=dict(
                              xaxis_title='Dates_In_Years',
                              yaxis_title='Strike',
                              zaxis_title='Implied_Volatility'))

        st.plotly_chart(fig)


def plot_simulations_streamlit(autocall):
    for actif_index, (df, stock) in enumerate(zip(autocall.monte_carlo.simulations, autocall.monte_carlo.stocks)):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Convertir l'index en datetime si ce n'est pas déjà le cas
        df.index = pd.to_datetime(df.index)

        # Tracer chaque simulation pour l'actif courant
        for sim_index in df.columns:
            ax.plot(df.index, df[sim_index], lw=1)

        # Ajouter une ligne horizontale pour la barrière de coupon et d'autocall
        ax.axhline(y=autocall.coupon_barrier * df.iloc[0, 0], color='g', linestyle='--',
                   label=f'Coupon Barrier ({round(autocall.coupon_barrier * df.iloc[0, 0], 1)})')
        ax.axhline(y=autocall.autocall_barrier * df.iloc[0, 0], color='r', linestyle='--',
                   label=f'Autocall Barrier ({round(autocall.autocall_barrier * df.iloc[0, 0], 1)})')
        ax.axhline(y=autocall.put_barrier * df.iloc[0, 0], color='orange', linestyle='--',
                   label=f'Put Barrier ({round(autocall.put_barrier * df.iloc[0, 0], 1)})')

        # Ajouter une ligne verticale pour chaque date d'observation
        for obs_date in autocall.monte_carlo.observation_dates:
            ax.axvline(x=obs_date, color='lightblue', linestyle='--', linewidth=1, alpha=0.5)

        # Formater l'axe des x pour afficher les dates de manière lisible
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)

        ax.set_title(f'Monte Carlo Simulation for {stock.ticker}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Process Value')
        ax.grid(True, which='both', axis='y', linestyle='--', color='grey')
        ax.grid(False, which='both', axis='x')
        ax.legend()

        st.pyplot(fig)


def plot_rate_curve(stock):
    """
    Tracer la courbe des taux.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(stock.rate_curve.data.index, stock.rate_curve.data['rates'], color='blue', linestyle='-',
             marker='o', markersize=4)
    plt.title('Courbe des taux', fontsize=20, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Taux', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(plt)