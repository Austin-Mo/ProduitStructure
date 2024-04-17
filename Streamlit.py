import streamlit as st
import pandas as pd
from backend.monte_carlo import MonteCarlo
from backend.models import Autocall
from backend.data.stock_data import StockData
from frontend.display import plot_volatility_surface_streamlit, plot_simulations_streamlit

st.title("Simulation de produits autocallables")

# Sélection des dates de début et de fin
start_date = st.date_input("Date de début", pd.to_datetime("2024-03-01"))
end_date = st.date_input("Date de fin", pd.to_datetime("2025-03-01"))

# Vérification que la date de fin est postérieure à la date de début
if start_date > end_date:
    st.error("La date de fin doit être postérieure à la date de début.")
else:

    st.subheader("Stratégie") 

    name_to_ticker = {
    "Apple": "AAPL US Equity",
    "Microsoft": "MSFT US Equity",
    "Google": "GOOGL US Equity",
    }

    # Liste prédéfinie des sous-jacents disponibles pour la sélection

    selected_strat = st.selectbox("Choisissez la stratégie souhaitée", ["worst-off", "best-off", "mono-asset"], key='strat_choice')

    if selected_strat == "mono-asset":
        selected_stocks_keys = st.selectbox("Choisissez le sous-jacent pour la stratégie Mono-jacent", list(name_to_ticker.keys()), key='stock_mono')
    else:
        selected_stocks_keys = st.multiselect("Choisissez les sous-jacents pour la strat Worst-Of et Best-Of", list(name_to_ticker.keys()), default=list(name_to_ticker.keys()))
        

    # Paramètres de simulation - Première ligne
    st.header("Paramètres de simulation")
    cols = st.columns([1, 1, 1])

    with cols[0]:
        nominal = st.number_input("Valeur nominale", value=1000, key='nominal')
    with cols[1]:
        num_simu = st.number_input("Nombre de simulations", value=100, format="%d", key='num_simu')
    with cols[2]:
        day_conv = st.selectbox("Date de convention", [360, 365], key='day_conv')
        

    # Paramètres de simulation - Deuxième ligne
    cols2 = st.columns([1, 1, 1, 1])

    with cols2[0]:
        coupon_rate = st.number_input("Taux de coupon", value=0.05, key='coupon_rate')
    with cols2[1]:
        put_barrier = st.number_input("Barrière put", value=0.8, key='put_barrier')
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
        show_volatility = st.checkbox("Afficher les surfaces de volatilité implicite des sous-jacents")

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
        # Récupération des données des sous-jacents sélectionnés
        stock_data = {
            'Apple': ('AAPL US Equity'),
            'Microsoft': ('MSFT US Equity'),
            'Google': ('GOOGL US Equity'),
        }
        selected_stocks = [StockData(ticker=data, pricing_date=start_date.strftime('%Y%m%d')) for key, data in stock_data.items() if key in selected_stocks_keys]

        if show_volatility:
            plot_volatility_surface_streamlit(selected_stocks)

        monte_carlo = MonteCarlo(stocks=selected_stocks,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    num_simu=num_simu,
                    day_conv=day_conv,
                    seed=seed,
                    observation_frequency=observation_frequency)

        autocall = Autocall(monte_carlo=monte_carlo,
                            strat=selected_strat,
                            nominal=nominal,
                            coupon_rate=coupon_rate,
                            coupon_barrier=coupon_barrier,
                            autocall_barrier=autocall_barrier,
                            put_barrier=put_barrier)

        plot_simulations_streamlit(autocall)

        autocall.calculate_average_present_value()
        probas = autocall.calculate_autocall_probabilities()

        st.markdown("---")
        
        if selected_strat == "mono-asset":
        
            st.write(f"Payoffs DataFrame for stratégie{selected_strat} with stock {autocall.monte_carlo.stocks}:")
            st.dataframe(autocall.payoffs)
        else:
            st.write(f"Payoffs DataFrame for stratégie {selected_strat}:")
            st.dataframe(autocall.payoffs)


        st.markdown("---")
        st.markdown(f"""
            <div style='text-align: center;'>
                <span style='font-size: 3.5em;'>Prix final stratégie {selected_strat}:</span>
                <br>
                <span style='font-size: 2.5em;'>{autocall.average_price:.2f} %</span>
            </div>
            """, unsafe_allow_html=True)


        