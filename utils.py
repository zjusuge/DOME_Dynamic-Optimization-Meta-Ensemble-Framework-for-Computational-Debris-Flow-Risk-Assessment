import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ----------------------------------------------------------------------
# Basic data / metric utilities
# ----------------------------------------------------------------------
def coerce_numeric_dataframe(X):
    """
    Convert dataframe columns to numeric where possible, coercing invalid values to NaN.
    """
    X = X.copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


def safe_mape(y_true, y_pred, epsilon=1e-8):
    """
    Safely compute MAPE while avoiding division by zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def safe_spearman_correlation(y_true, y_pred):
    """
    Safely compute Spearman correlation.
    Returns 0.0 if the result is invalid or undefined.
    """
    try:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 2:
            return 0.0

        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            return 0.0

        corr, _ = spearmanr(y_true, y_pred)
        if np.isnan(corr):
            return 0.0
        return float(corr)
    except Exception:
        return 0.0


def evaluate_regression(y_true, y_pred):
    """
    Compute a standard set of regression metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    spearman_corr = safe_spearman_correlation(y_true, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Spearman_Correlation": spearman_corr
    }


# ----------------------------------------------------------------------
# Multicollinearity analysis
# ----------------------------------------------------------------------
def calculate_vif(X):
    """
    Calculate VIF for each feature.

    Parameters
    ----------
    X : pandas.DataFrame

    Returns
    -------
    dict
        {feature_name: vif_value}
    """
    X = coerce_numeric_dataframe(X)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill missing with column medians
    fill_values = X.median(numeric_only=True).to_dict()
    for col in X.columns:
        fill_val = fill_values.get(col, 0.0)
        if pd.isna(fill_val) or not np.isfinite(fill_val):
            fill_val = 0.0
        X[col] = X[col].fillna(fill_val)

    vif_scores = {}
    if X.shape[1] == 0:
        return vif_scores

    for i, feature in enumerate(X.columns):
        try:
            vif_value = variance_inflation_factor(X.values.astype(float), i)
            if not np.isfinite(vif_value):
                vif_value = float("inf")
            vif_scores[feature] = float(vif_value)
        except Exception:
            vif_scores[feature] = float("inf")

    return vif_scores


def drop_high_vif_features(X, vif_threshold=10.0, corr_threshold=0.90, min_features=3):
    """
    Remove high-VIF features. If VIF is unstable, fall back to correlation filtering.

    Returns
    -------
    X_filtered : pandas.DataFrame
    removed_features : list
    """
    X = X.copy()
    removed_features = []

    if X.shape[1] <= min_features:
        return X, removed_features

    vif_scores = calculate_vif(X)
    high_vif = [k for k, v in vif_scores.items() if np.isfinite(v) and v > vif_threshold]

    if high_vif and (X.shape[1] - len(high_vif) >= min_features):
        removed_features = high_vif
        X = X.drop(columns=removed_features)
        return X, removed_features

    # Fallback: correlation-based filtering
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    corr_removed = []
    for column in upper_tri.columns:
        if any(upper_tri[column] > corr_threshold):
            corr_removed.append(column)

    if corr_removed and (X.shape[1] - len(corr_removed) >= min_features):
        removed_features = corr_removed
        X = X.drop(columns=removed_features)

    return X, removed_features


# ----------------------------------------------------------------------
# Feature selection: RFE + VIF
# ----------------------------------------------------------------------
def rfe_vif_feature_selection(
    X,
    y,
    estimator=None,
    select_ratio=0.70,
    min_features=5,
    vif_threshold=10.0,
    corr_threshold=0.90,
    random_state=42
):
    """
    Feature selection using:
    1. RFE
    2. VIF / correlation-based multicollinearity filtering

    Returns
    -------
    X_selected : pandas.DataFrame
    selected_features : list
    metadata : dict
    """
    X = coerce_numeric_dataframe(X)
    X = X.replace([np.inf, -np.inf], np.nan)

    fill_values = X.median(numeric_only=True).to_dict()
    for col in X.columns:
        fill_val = fill_values.get(col, 0.0)
        if pd.isna(fill_val) or not np.isfinite(fill_val):
            fill_val = 0.0
        X[col] = X[col].fillna(fill_val)

    y = pd.Series(pd.to_numeric(pd.Series(y), errors="coerce")).fillna(pd.Series(y).median())

    if estimator is None:
        estimator = RandomForestRegressor(
            n_estimators=60,
            random_state=random_state
        )

    if X.shape[1] <= min_features:
        selected_features = list(X.columns)
        metadata = {
            "rfe_selected": selected_features,
            "removed_multicollinearity_features": [],
            "final_selected": selected_features
        }
        return X[selected_features], selected_features, metadata

    n_features_to_select = min(
        max(int(np.ceil(X.shape[1] * select_ratio)), min_features),
        X.shape[1]
    )

    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features_to_select
    )
    rfe.fit(X, y)

    rfe_selected = X.columns[rfe.support_].tolist()
    X_rfe = X[rfe_selected].copy()

    X_final, removed_features = drop_high_vif_features(
        X_rfe,
        vif_threshold=vif_threshold,
        corr_threshold=corr_threshold,
        min_features=max(3, min_features // 2)
    )

    final_selected = list(X_final.columns)

    metadata = {
        "rfe_selected": rfe_selected,
        "removed_multicollinearity_features": removed_features,
        "final_selected": final_selected
    }

    return X_final, final_selected, metadata


# ----------------------------------------------------------------------
# ICWCM weighting
# ----------------------------------------------------------------------
class ICWCMWeighting:
    """
    Information-Correlation Weighted Combination Method (ICWCM)

    This implementation is designed to be numerically stable for practical
    machine-learning workflows. It combines several weighting strategies:

    - Coefficient of variation weights
    - Entropy weights
    - CRITIC weights
    - Optional target-correlation weights

    Then it aggregates them through a game-theory-inspired consensus weighting.
    """

    def __init__(self, epsilon=1e-10):
        self.epsilon = float(epsilon)
        self.weights = None
        self.method_weights = None
        self.method_names = None
        self.weight_matrix_ = None

    def _prepare_X(self, X):
        X = coerce_numeric_dataframe(X)
        X = X.replace([np.inf, -np.inf], np.nan)

        fill_values = X.median(numeric_only=True).to_dict()
        for col in X.columns:
            fill_val = fill_values.get(col, 0.0)
            if pd.isna(fill_val) or not np.isfinite(fill_val):
                fill_val = 0.0
            X[col] = X[col].fillna(fill_val)

        return X.astype(float)

    def _normalize_nonnegative(self, weights):
        weights = np.asarray(weights, dtype=float)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        weights = np.maximum(weights, 0.0)

        if np.allclose(weights.sum(), 0.0):
            weights = np.ones_like(weights, dtype=float)

        return weights / weights.sum()

    def coefficient_of_variation_weights(self, X):
        """
        Coefficient of variation weighting.
        """
        X = self._prepare_X(X)

        means = np.abs(X.mean(axis=0).values.astype(float))
        stds = X.std(axis=0, ddof=0).values.astype(float)

        cv = stds / (means + self.epsilon)
        return self._normalize_nonnegative(cv)

    def entropy_weights(self, X):
        """
        Entropy weighting with robust min-max normalization to positive space.
        """
        X = self._prepare_X(X)
        X_pos = X.copy()

        for col in X_pos.columns:
            values = X_pos[col].values.astype(float)
            vmin = np.min(values)
            vmax = np.max(values)

            if np.isclose(vmax - vmin, 0.0):
                X_pos[col] = 1.0
            else:
                X_pos[col] = (values - vmin) / (vmax - vmin) + self.epsilon

        n = max(len(X_pos), 2)
        entropy_scores = []

        for col in X_pos.columns:
            col_sum = X_pos[col].sum()
            if np.isclose(col_sum, 0.0):
                p = np.full(len(X_pos), 1.0 / len(X_pos))
            else:
                p = X_pos[col].values / col_sum

            p = np.clip(p, self.epsilon, None)
            e = -np.sum(p * np.log(p)) / np.log(n)
            d = 1.0 - e
            entropy_scores.append(d)

        return self._normalize_nonnegative(entropy_scores)

    def critic_weights(self, X):
        """
        CRITIC weighting.
        """
        X = self._prepare_X(X)

        stds = X.std(axis=0, ddof=0).values.astype(float)
        corr_matrix = X.corr().fillna(0.0).values.astype(float)

        scores = []
        for i in range(X.shape[1]):
            conflict = np.sum(1.0 - np.abs(corr_matrix[i, :])) - (1.0 - abs(corr_matrix[i, i]))
            score = stds[i] * max(conflict, 0.0)
            scores.append(score)

        return self._normalize_nonnegative(scores)

    def target_correlation_weights(self, X, y):
        """
        Absolute Spearman correlation between each feature and target.
        """
        X = self._prepare_X(X)
        y = pd.Series(pd.to_numeric(pd.Series(y), errors="coerce")).fillna(pd.Series(y).median())

        scores = []
        for col in X.columns:
            corr = safe_spearman_correlation(X[col].values, y.values)
            scores.append(abs(corr))

        return self._normalize_nonnegative(scores)

    def game_theory_optimization(self, weight_matrix, max_iterations=200, tolerance=1e-8):
        """
        Game-theory-inspired consensus weighting across multiple methods.

        Rather than using a numerically fragile closed-form update, this method
        iteratively assigns larger method weights to weighting vectors that are
        closer to the evolving consensus.

        Parameters
        ----------
        weight_matrix : np.ndarray, shape (n_methods, n_features)

        Returns
        -------
        final_weights : np.ndarray
        """
        weight_matrix = np.asarray(weight_matrix, dtype=float)

        # Row-normalize safely
        normalized_rows = []
        for row in weight_matrix:
            normalized_rows.append(self._normalize_nonnegative(row))
        weight_matrix = np.vstack(normalized_rows)

        n_methods = weight_matrix.shape[0]
        lambda_k = np.ones(n_methods, dtype=float) / n_methods
        consensus = np.mean(weight_matrix, axis=0)
        consensus = self._normalize_nonnegative(consensus)

        for _ in range(max_iterations):
            old_lambda = lambda_k.copy()

            distances = np.linalg.norm(weight_matrix - consensus, axis=1) + self.epsilon
            lambda_k = 1.0 / distances
            lambda_k = self._normalize_nonnegative(lambda_k)

            consensus = np.sum(lambda_k[:, None] * weight_matrix, axis=0)
            consensus = self._normalize_nonnegative(consensus)

            if np.allclose(lambda_k, old_lambda, atol=tolerance):
                break

        self.method_weights = lambda_k
        return consensus

    def calculate_weights(self, X, y=None, include_target_correlation=True):
        """
        Calculate final ICWCM-style feature weights.

        Parameters
        ----------
        X : pandas.DataFrame
        y : array-like, optional
        include_target_correlation : bool, default=True

        Returns
        -------
        np.ndarray
            Final feature weights
        """
        X = self._prepare_X(X)

        methods = []
        method_names = []

        cv_weights = self.coefficient_of_variation_weights(X)
        methods.append(cv_weights)
        method_names.append("coefficient_of_variation")

        entropy_weights = self.entropy_weights(X)
        methods.append(entropy_weights)
        method_names.append("entropy")

        critic_weights = self.critic_weights(X)
        methods.append(critic_weights)
        method_names.append("critic")

        if include_target_correlation and y is not None:
            corr_weights = self.target_correlation_weights(X, y)
            methods.append(corr_weights)
            method_names.append("target_correlation")

        weight_matrix = np.vstack(methods)
        final_weights = self.game_theory_optimization(weight_matrix)

        self.weights = final_weights
        self.method_names = method_names
        self.weight_matrix_ = weight_matrix

        return final_weights

    def get_feature_weights(self, feature_names):
        """
        Return feature weights as a dictionary.
        """
        if self.weights is None:
            raise ValueError("Weights have not been calculated yet.")

        if len(feature_names) != len(self.weights):
            raise ValueError("feature_names length must match number of weights.")

        return {
            str(feature): float(weight)
            for feature, weight in zip(feature_names, self.weights)
        }
