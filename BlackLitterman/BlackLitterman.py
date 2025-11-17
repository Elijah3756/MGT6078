import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize

class BlackLitterman:
    def __init__(self, data_path):
        self.data_path = data_path

        # Prior / market state
        self.cov_matrix_prior = None
        self.market_weights = None
        self.market_variance = None
        self.prior_returns = None
        self.covariance_tau = None
        self.risk_aversion = None
        self.tau_for_covariance = None
        self.std_dev = None

        # Views
        self.P_matrix = None
        self.Q_vector = None
        self.tau_omega = None
        self.relative_confidence = None
        self.Omega_matrix = None
        self.prior_precision_of_views = None

        # Posterior
        self.posterior_returns = None
        self.posterior_cov_matrix = None

    # -----------------------------
    # Top-level API
    # -----------------------------
    def black_litterman_weights(self, risk_aversion,
                                tau_for_covariance,
                                market_weights,
                                P_matrix,
                                Q_vector,
                                tau_omega,
                                relative_confidence):
        # 1. Load data & set parameters
        bl_data = self._load_data()
        self._set_hyperparameters(
            risk_aversion=risk_aversion,
            tau_for_covariance=tau_for_covariance,
            tau_omega=tau_omega,
            relative_confidence=relative_confidence,
            market_weights=market_weights
        )

        # 2. Compute prior quantities from market data
        self._compute_prior_from_market(bl_data)

        # 3. Set up views (P, Q, Omega, prior precision of views)
        self._setup_views(P_matrix, Q_vector)

        # 4. Compute posterior returns & covariance
        self._compute_posterior()

        # 5. Optimize portfolio using posterior
        initial_weights = np.array(np.ones_like(market_weights))
        opt_weights, sharpe = self.optimize_portfolio(initial_weights)

        return opt_weights, sharpe

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _load_data(self):
        bl_data = pd.read_csv(self.data_path)
        return bl_data

    def _set_hyperparameters(self,
                             risk_aversion,
                             tau_for_covariance,
                             tau_omega,
                             relative_confidence,
                             market_weights):
        self.risk_aversion = risk_aversion
        self.tau_omega = tau_omega
        self.tau_for_covariance = tau_for_covariance
        self.relative_confidence = relative_confidence
        self.market_weights = np.array(market_weights).reshape(len(market_weights), 1)

    def _compute_prior_from_market(self, bl_data):
        # Prior covariance
        self.cov_matrix_prior = bl_data.cov()
        print(self.cov_matrix_prior)

        # Market variance
        self.market_variance = np.matmul(
            self.market_weights.T,
            np.matmul(self.cov_matrix_prior, self.market_weights)
        )[0][0]

        # Implied equilibrium (prior) returns
        self.prior_returns = self.risk_aversion * np.matmul(
            self.cov_matrix_prior,
            self.market_weights
        )

        # Tau-adjusted covariance
        self.covariance_tau = self.cov_matrix_prior * self.tau_for_covariance

        # Market expected excess return (scalar) and std dev
        self.market_exp_excess_return = self.risk_aversion * self.market_variance
        self.std_dev = math.sqrt(self.market_variance)

    def _setup_views(self, P_matrix, Q_vector):
        self.P_matrix = P_matrix
        self.Q_vector = Q_vector

        # Omega matrix based on scaling with tau_omega
        self.Omega_matrix = self.tau_omega * np.matmul(
            self.P_matrix,
            np.matmul(self.cov_matrix_prior, self.P_matrix.T)
        )

        # Prior precision of views
        self.prior_precision_of_views = np.matmul(
            self.P_matrix,
            np.matmul(self.covariance_tau, self.P_matrix.T)
        )

    def _compute_posterior(self):
        # (P Στ P' + Ω)^(-1)
        middle_inv = np.linalg.inv(self.prior_precision_of_views + self.Omega_matrix)

        # Posterior expected returns
        self.posterior_returns = (
            self.prior_returns
            + np.matmul(
                np.matmul(
                    np.matmul(self.covariance_tau, self.P_matrix.T),
                    middle_inv
                ),
                (self.Q_vector - np.matmul(self.P_matrix, self.prior_returns))
            )
        )

        # Posterior covariance
        self.posterior_cov_matrix = (
            self.cov_matrix_prior
            + self.covariance_tau
            - np.matmul(
                np.matmul(
                    np.matmul(self.covariance_tau, self.P_matrix.T),
                    middle_inv
                ),
                np.matmul(self.P_matrix, self.covariance_tau)
            )
        )

    # -----------------------------
    # Optimization utilities
    # -----------------------------
    def neg_sharpe_ratio(self, weights):
        weights = np.array(weights).reshape(-1, 1)
        expected_return = float((weights.T @ self.posterior_returns)[0][0])
        variance = float((weights.T @ self.posterior_cov_matrix @ weights)[0][0])
        return -expected_return / math.sqrt(variance)

    def optimize_portfolio(self, initial_weights):
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        result = minimize(self.neg_sharpe_ratio, initial_weights,
                          method='SLSQP', constraints=constraints)

        # Optimal weights and Sharpe ratio
        opt_weights = result.x
        max_sharpe = -result.fun  # negate back

        print("Optimal Weights:", opt_weights)
        print("Maximum Sharpe Ratio:", max_sharpe)
        return opt_weights, max_sharpe
