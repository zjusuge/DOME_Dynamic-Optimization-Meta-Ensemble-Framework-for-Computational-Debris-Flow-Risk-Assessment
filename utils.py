import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import entropy
import itertools


class ICWCMWeighting:
    """Information-Correlation Weighted Combination Method (ICWCM)"""

    def __init__(self):
        self.weights = None
        self.method_weights = None

    def coefficient_of_variation_weights(self, X):
        """Calculate weights using coefficient of variation"""
        weights = []
        for col in X.columns:
            std = X[col].std()
            mean = X[col].mean()
            cv = std / (mean + 1e-10)
            weights.append(cv)

        weights = np.array(weights)
        weights = weights / np.sum(weights)
        return weights

    def entropy_weights(self, X):
        """Calculate weights using entropy method"""
        X_normalized = X / X.sum()
        weights = []
        n = len(X)

        for col in X_normalized.columns:
            p = X_normalized[col].values
            p = p[p > 0]
            if len(p) > 0:
                h = -np.sum(p * np.log(p + 1e-10))
                e = h / np.log(n)
                w = (1 - e)
            else:
                w = 0
            weights.append(w)

        weights = np.array(weights)
        weights = weights / (np.sum(weights) + 1e-10)
        return weights

    def critic_weights(self, X):
        """Calculate weights using CRITIC method"""
        X_std = (X - X.mean()) / X.std()
        weights = []

        for i, col in enumerate(X.columns):
            std_i = X_std[col].std()

            correlations = []
            for j, other_col in enumerate(X.columns):
                if i != j:
                    corr = abs(X_std[col].corr(X_std[other_col]))
                    correlations.append(corr)

            if correlations:
                conflict = 1 - np.mean(correlations)
            else:
                conflict = 1

            weight = std_i * conflict
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / np.sum(weights)
        return weights

    def game_theory_optimization(self, weight_matrix):
        """Optimize final weights using game theory (Eq. 4)"""
        n_methods, n_features = weight_matrix.shape
        lambda_k = np.ones(n_methods) / n_methods

        max_iterations = 100
        tolerance = 1e-6

        for iteration in range(max_iterations):
            old_lambda = lambda_k.copy()

            for k in range(n_methods):
                numerator = 0
                denominator = 0

                for j in range(n_features):
                    for l in range(n_methods):
                        if l != k:
                            diff = weight_matrix[k, j] - weight_matrix[l, j]
                            numerator += diff * weight_matrix[k, j]
                            denominator += diff ** 2

                if denominator > 0:
                    lambda_k[k] = numerator / denominator
                else:
                    lambda_k[k] = 1.0 / n_methods

            lambda_k = np.abs(lambda_k)
            lambda_k = lambda_k / np.sum(lambda_k)

            if np.allclose(lambda_k, old_lambda, atol=tolerance):
                break

        final_weights = np.zeros(n_features)
        for k in range(n_methods):
            final_weights += lambda_k[k] * weight_matrix[k, :]

        self.method_weights = lambda_k
        return final_weights

    def calculate_weights(self, X, y):
        """Calculate optimal weights using ICWCM method"""
        print("  Calculating feature weights using ICWCM...")

        cv_weights = self.coefficient_of_variation_weights(X)
        entropy_weights = self.entropy_weights(X)
        critic_weights = self.critic_weights(X)

        weight_matrix = np.vstack([cv_weights, entropy_weights, critic_weights])
        final_weights = self.game_theory_optimization(weight_matrix)

        self.weights = final_weights
        return final_weights


def calculate_vif(X):
    """Calculate Variance Inflation Factor for multicollinearity detection"""
    vif_scores = {}

    for i, feature in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_scores[feature] = vif
        except:
            vif_scores[feature] = float('inf')

    return vif_scores