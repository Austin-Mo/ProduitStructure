import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backend.monte_carlo import MonteCarlo
from backend.models import Autocall
from backend.data.stock_data import StockData
from backend.data.rate_curve import ZeroCouponCurve
import matplotlib.dates as mdates

st.title("Simulation de produits autocallables")

# Sélection des dates de début et de fin
start_date = st.date_input("Date de début", pd.to_datetime("2024-03-01"))
end_date = st.date_input("Date de fin", pd.to_datetime("2025-03-01"))

# Vérification que la date de fin est postérieure à la date de début
if start_date >= end_date:
    st.error("La date de fin doit être postérieure à la date de début.")
else:

     # Liste prédéfinie des sous-jacents disponibles pour la sélection
    available_stocks_keys = ['Apple', 'Microsoft', 'Google']
    selected_stocks_keys = st.multiselect("Choisissez les sous-jacents", available_stocks_keys, default=available_stocks_keys)


    # Paramètres de simulation - Première ligne
    st.header("Paramètres de simulation")
    cols = st.columns([1, 1, 1, 1])

    with cols[1]:
        risk_free_rate = st.number_input("Taux sans risque annuel", value=0.02, key='risk_free_rate')
    with cols[2]:
        num_simu = st.number_input("Nombre de simulations", value=2, format="%d", key='num_simu')
    with cols[3]:
        day_conv = st.number_input("Nombre de steps", value=360, format="%d", key='day_conv')
        

    # Paramètres de simulation - Deuxième ligne
    cols2 = st.columns([1, 1, 1, 1])
    with cols2[0]:
        nominal = st.number_input("Valeur nominale", value=1000, key='nominal')
    with cols2[1]:
        coupon_rate = st.number_input("Taux de coupon", value=0.05, key='coupon_rate')
    with cols2[2]:
        coupon_barrier = st.number_input("Barrière de coupon", value=1.1, key='coupon_barrier')
    with cols2[3]:
        autocall_barrier = st.number_input("Barrière d'autocall", value=1.3, key='autocall_barrier')

    # Paramètres supplémentaires si nécessaire
    with st.expander("Plus de paramètres"):
        observation_frequency = st.selectbox(
        "Fréquence d'observation",
        ['monthly', 'quarterly', 'semiannually', 'annually'],
        index=0  # Choix par défaut à 'monthly'
    )
        seed = st.number_input("Seed ", value=24, format="%d", key='seed')
        

    # Bouton de simulation au centre
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            width: 100%;
            border: 2px solid #4CAF50;
            border-radius: 5px;
            font-size: 20px;
            color: white;
            background-color: #4CAF50;
            padding: 10px 24px;
            margin: 10px 0;
            cursor: pointer;
        }
        div.stButton > button:hover {
            background-color: #45a049;
        }
        </style>""", unsafe_allow_html=True)

    if st.button("Simuler les chemins de prix"):


         # Récupération de la courbe des taux après la sélection des sous-jacents
        US_rate = ZeroCouponCurve(date=start_date.strftime('%Y%m%d'))

        # Récupération des données des sous-jacents sélectionnés
        stock_data = {
            'Apple': ('AAPL US Equity', US_rate),
            'Microsoft': ('MSFT US Equity', US_rate),
            'Google': ('GOOGL US Equity', US_rate),
        }
        selected_stocks = {key: StockData(ticker=data[0], pricing_date=start_date.strftime('%Y%m%d'), rate=data[1]) for key, data in stock_data.items() if key in selected_stocks_keys}

        monte_carlo = MonteCarlo(stocks=list(selected_stocks.values()),
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    risk_free_rate=risk_free_rate,
                    num_simu=num_simu,
                    day_conv=day_conv,
                    seed=seed,
                    volatilities=[0.1, 0.2, 0.3],
                    observation_frequency=observation_frequency)


        autocall = Autocall(monte_carlo,
                             nominal=nominal,
                               coupon_rate=coupon_rate,
                                 coupon_barrier=coupon_barrier,
                                   autocall_barrier=autocall_barrier,
                                     risk_free=risk_free_rate)
        
        def plot_simulations_streamlit(autocall):
            for actif_index, df in enumerate(autocall.monte_carlo.simulations):
                fig, ax = plt.subplots(figsize=(10, 6))

                # Convertir l'index en datetime si ce n'est pas déjà le cas
                df.index = pd.to_datetime(df.index)

                # Tracer chaque simulation pour l'actif courant
                for sim_index in df.columns:
                    ax.plot(df.index, df[sim_index], lw=1)

                # Ajouter une ligne horizontale pour la barrière de coupon et d'autocall
                ax.axhline(y=autocall.coupon_barrier * df.iloc[0,0], color='g', linestyle='--', label=f'Coupon Barrier ({round(autocall.coupon_barrier*df.iloc[0,0], 1)})')
                ax.axhline(y=autocall.autocall_barrier * df.iloc[0,0], color='r', linestyle='--', label=f'Autocall Barrier ({round(autocall.autocall_barrier*df.iloc[0,0], 1)})')

                # Ajouter une ligne verticale pour chaque date d'observation
                for obs_date in autocall.monte_carlo.observation_dates:
                    ax.axvline(x=obs_date, color='lightblue', linestyle='--', linewidth=1, alpha=0.5)

                # Formater l'axe des x pour afficher les dates de manière lisible
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.xticks(rotation=45)

                ax.set_title(f'Monte carlo simulation for Asset {actif_index + 1}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Process Value')
                ax.grid(True, which='both', axis='y', linestyle='--', color='grey')
                ax.grid(False, which='both', axis='x')
                ax.legend()

                st.pyplot(fig) 

        plot_simulations_streamlit(autocall)

        autocall.calculate_average_present_value()

        st.markdown("---")  # Un séparateur visuel
            # Afficher les DataFrames des payoffs et des payoffs actualisés
        for i, df in enumerate(autocall.payoffs):
            st.write(f"Payoffs DataFrame for Asset {i+1}:")
            st.dataframe(df)

        st.markdown("---")  # Un séparateur visuel

        # Titre pour la section des prix moyens individuels
        st.markdown("#### Prix moyen final pour chaque actif:")

        # Créer une grille avec un certain nombre de colonnes équivalent au nombre d'actifs
        cols = st.columns(len(autocall.average_price))

        # Remplir chaque colonne avec le nom de l'actif et son prix moyen final
        for i, (col, value) in enumerate(zip(cols, autocall.average_price)):
            col.metric(label=f"Actif {i+1}", value=f"{value:.2f} €")

        st.markdown("---")  # Un séparateur visuel

        # Pour le prix moyen final sur tous les actifs, on utilise HTML pour une taille de police plus grande
        st.markdown(f"""
            <div style='text-align: center;'>
                <span style='font-size: 1.5em;'>Prix moyen final sur tous les actifs:</span>
                <br>
                <span style='font-size: 2.5em;'>{autocall.overall_average:.2f} €</span>
            </div>
            """, unsafe_allow_html=True)


        