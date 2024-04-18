import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from backend.data.rate_curve import ZeroCouponCurve
from datetime import timedelta


class Models:
    def __init__(self, spot_price, strike, risk_free_rate, maturity, dividend_yield, volatility):
        self.spot_price = spot_price
        self.strike = strike
        self.risk_free_rate = risk_free_rate
        self.maturity = maturity
        self.dividend_yield = dividend_yield
        self.volatility = volatility

    def black_scholes(self, call_or_put='call'):
        d1 = (np.log(self.spot_price / self.strike) + (
                self.risk_free_rate - self.dividend_yield + 0.5 * self.volatility ** 2) * self.maturity) / (
                     self.volatility * np.sqrt(self.maturity))
        d2 = d1 - self.volatility * np.sqrt(self.maturity)
        if call_or_put == 'call':
            return self.spot_price * np.exp(-self.dividend_yield * self.maturity) * norm.cdf(d1) - self.strike * np.exp(
                -self.risk_free_rate * self.maturity) * norm.cdf(d2)
        elif call_or_put == 'put':
            return self.strike * np.exp(-self.risk_free_rate * self.maturity) * norm.cdf(
                -d2) - self.spot_price * np.exp(-self.dividend_yield * self.maturity) * norm.cdf(-d1)


class Autocall:
    def __init__(self, monte_carlo, strat, nominal, coupon_rate, coupon_barrier, autocall_barrier, put_barrier):
        self.monte_carlo = monte_carlo
        self.nominal = nominal
        self.strat = strat
        self.coupon_rate = coupon_rate
        self.coupon_barrier = coupon_barrier
        self.autocall_barrier = autocall_barrier
        self.put_barrier = put_barrier
        self.risk_free = ZeroCouponCurve(date=self.monte_carlo.start_date.strftime("%Y%m%d"))
        self.payoffs, self.payoffs_discount = self.generate_payoffs()
        self.average_price = None
        self.figs = []

    def discount_factor(self, step, total_steps):
        time = step / total_steps * self.monte_carlo.maturity  # Convertir en fraction de la maturité totale
        date = self.monte_carlo.start_date + timedelta(days=time * self.monte_carlo.day_conv)
        return np.exp(-self.risk_free.interpolate_rate(date=date) * time)

    def generate_payoffs(self):

        # Obtenir le nombre d'étapes et de simulations pour l'actif actuel
        num_steps = len(self.monte_carlo.observation_dates)
        num_simulations = self.monte_carlo.num_simu

        # Initialiser des tableaux pour stocker les payoffs et les payoffs actualisés à chaque étape pour chaque simulation
        payoffs_actif = np.zeros((num_steps, num_simulations))
        discounted_payoffs_actif = np.zeros((num_steps, num_simulations))

        if self.strat == "mono":
            df = self.monte_carlo.simulations[0]
        else:
            df = self.choice_asset_worstoff_bestoff()

        for step, time_step in enumerate(self.monte_carlo.observation_dates):
            total_payment, no_redemption_condition = self.payoff_by_step(df, step, time_step)

            # Stocker le paiement total et le paiement total actualisé à l'étape courante
            payoffs_actif[step, :] = total_payment
            discount = self.discount_factor(step, num_steps)
            discounted_payoffs_actif[step, :] = (total_payment * discount)

        # --------------------------------------------------------------------------------------------------------------------------------
        # Part put barrière
        filter_df = df.loc[self.monte_carlo.observation_dates]
        # Je regarde le plus petit ratio de prix sur toutes les observations dates
        initial_prices = df.iloc[0].values
        min_price_ratios = filter_df.min(axis=0) / initial_prices
        # Si le plus petit des ratios est inférieur à la barrière put alors cette barrière a été franchit
        put_condition = min_price_ratios <= self.put_barrier
        final_price_ratios = filter_df.iloc[-1].values / initial_prices

        # Boucle sur toutes les simulations
        for i in range(len(no_redemption_condition)):
            # Si il n'y a pas eu déjà de redemption, que le barrière put a au moins été franchit une fois et que le dernier prix est inférieur au prix initial alors il faut imputer la perte
            if no_redemption_condition[i] and put_condition[i] and (final_price_ratios[i] < 1):
                # J'annule tous les paiements de coupons précédents
                payoffs_actif[:, i] = discounted_payoffs_actif[:, i] = 0
                # Inputer la perte sur le dernier payoff
                payoffs_actif[-1, i] = self.nominal * final_price_ratios[i]

                discount = self.discount_factor(num_steps, num_steps)
                discounted_payoffs_actif[-1, i] = (payoffs_actif[-1, i] * discount)
        # --------------------------------------------------------------------------------------------------------------------------------

        # Créer des DataFrames pour les payoffs et les payoffs actualisés et les ajouter aux listes
        df_payoffs = pd.DataFrame(payoffs_actif, index=self.monte_carlo.observation_dates,
                                  columns=[f'Simulation {sim + 1}' for sim in range(num_simulations)])
        df_discounted_payoffs = pd.DataFrame(discounted_payoffs_actif, index=self.monte_carlo.observation_dates,
                                             columns=[f'Simulation {sim + 1}' for sim in range(num_simulations)])
        return df_payoffs, df_discounted_payoffs

    def payoff_by_step(self, df, step, time_step):
        # Obtenir les prix courants et les prix initiaux pour calculer les ratios de prix
        current_prices = df.loc[time_step].values
        initial_prices = df.iloc[0].values
        price_ratios = current_prices / initial_prices

        # Calculer les conditions de paiement du coupon et d'autocall basées sur les ratios de prix
        coupon_condition = price_ratios >= self.coupon_barrier
        autocall_condition = price_ratios >= self.autocall_barrier

        # S'assurer que le prix n'a jamais dépassé l'autocall barrière dans le passé sinon fin de contrat
        if step > 0:
            # Filtrer df pour ces dates d'observation
            filter_df = df.loc[self.monte_carlo.observation_dates[:step]]
            # Calculer max_price_ratios en se basant sur les prix filtrés
            max_price_ratios = filter_df.max(axis=0) / initial_prices
            no_redemption_condition = max_price_ratios <= self.autocall_barrier
        else:
            # Si nous sommes à la première observation, utiliser les ratios actuels comme max_ratios
            max_price_ratios = price_ratios
            no_redemption_condition = True

        num_steps = len(self.monte_carlo.observation_dates)
        # À la dernière étape, s'assurer de payer le nominal si la barrière put n'a pas été franchit et si les conditions d'autocall ne sont pas remplies
        if step == (num_steps - 1):
            for i in range(len(no_redemption_condition)):
                if bool(no_redemption_condition[i]):
                    no_redemption_condition[i] = autocall_condition[i] = True

        # Calculer les paiements de coupon et de rachat, puis le paiement total pour chaque simulation
        coupon_payment = self.nominal * self.coupon_rate * coupon_condition * no_redemption_condition
        redemption_payment = self.nominal * autocall_condition * no_redemption_condition
        total_payment = coupon_payment + redemption_payment

        return total_payment, no_redemption_condition

    def choice_asset_worstoff_bestoff(self):
        # Normaliser chaque DataFrame par sa première valeur (ligne 0, colonne 0)
        normalized_dfs = [df / df.iloc[0, 0] for df in self.monte_carlo.simulations]

        # Utiliser np.maximum.reduce pour trouver le DataFrame avec les valeurs maximales pour "best-off"
        if self.strat == "best-off":
            max_df = np.maximum.reduce(normalized_dfs)
        # Utiliser np.minimum.reduce pour trouver le DataFrame avec les valeurs minimales pour "worst-off"
        else:
            max_df = np.minimum.reduce(normalized_dfs)

        # Convertir le résultat en DataFrame
        result_df = pd.DataFrame(max_df, index=self.monte_carlo.simulations[0].index,
                                 columns=self.monte_carlo.simulations[0].columns)

        return result_df

    def calculate_average_present_value(self):
        """Calcule la valeur présente moyenne pour chaque actif et la moyenne globale."""
        total_discounted = self.payoffs_discount.sum(
            axis=0)  # Sum along rows to get the sum of all discount flows for each simulation
        average_price = total_discounted.mean()  # Calculate the mean across all simulations for the current asset
        self.average_price = average_price / self.nominal * 100

    def print_average_present_values(self):
        """Affiche les valeurs présentes moyennes calculées pour chaque actif et la moyenne globale."""
        if self.average_price is None or self.overall_average is None:
            self.calculate_average_present_value()

        for stock, value in zip(self.monte_carlo.stocks, self.average_price):
            print(f"Prix moyen final pour {stock.ticker}: {value:.2f} €")  # Utilisation de `stock.name`

        print(f"Prix moyen final sur tous les actifs: {self.overall_average:.2f} €")

    def calculate_autocall_probabilities(self):
        autocall_occurrences = [0] * len(self.monte_carlo.observation_dates)
        num_simulations = self.monte_carlo.simulations[0].shape[1]

        for step, time_step in enumerate(self.monte_carlo.observation_dates):
            for df in self.monte_carlo.simulations:
                current_prices = df.loc[time_step].values
                initial_prices = df.iloc[0].values
                price_ratios = current_prices / initial_prices
                autocall_condition = price_ratios >= self.autocall_barrier
                autocall_occurrences[step] += np.sum(autocall_condition)

        autocall_probabilities = [occ / num_simulations for occ in autocall_occurrences]
        autocall_probabilities_dict = {date.strftime('%Y-%m-%d'): prob for date, prob in
                                       zip(self.monte_carlo.observation_dates, autocall_probabilities)}

        return autocall_probabilities_dict
