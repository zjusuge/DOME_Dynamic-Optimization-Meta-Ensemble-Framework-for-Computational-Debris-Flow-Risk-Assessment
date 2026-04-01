import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_val_predict,
    KFold
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def safe_mape(y_true, y_pred, epsilon=1e-8):
    """Safely compute MAPE while avoiding division by zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def safe_spearman_correlation(y_true, y_pred):
    """Safely compute Spearman correlation and return 0.0 when invalid."""
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


def resolve_target_column(df, preferred=None, fallback_to_last=True):
    """
    Resolve the target column from a dataframe.

    Preferred behavior:
    1. Use user-specified column if present.
    2. Check common susceptibility/risk target names.
    3. Fall back to the last column if requested.

    This keeps compatibility with historical spreadsheets using 'Risk_index'
    while aligning terminology with susceptibility assessment.
    """
    if preferred is not None:
        if preferred in df.columns:
            return preferred
        raise ValueError(
            f"Preferred target column '{preferred}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    candidate_names = [
        "Susceptibility_index",
        "Susceptibility",
        "SusceptibilityScore",
        "Susceptibility_score",
        "Risk_index",      # legacy compatibility
        "label",
        "Label",
        "target",
        "Target",
        "y"
    ]

    for col in candidate_names:
        if col in df.columns:
            return col

    if fallback_to_last:
        return df.columns[-1]

    raise ValueError(
        "Unable to resolve target column automatically. "
        "Please specify it explicitly."
    )


class DOMEModel:
    """
    DOME (Dynamic Optimization Meta-Ensemble) model for
    inventory-based regional debris-flow susceptibility assessment.

    Notes
    -----
    - The code remains backward-compatible with legacy target naming such as
      'Risk_index', but the current manuscript framing is susceptibility-oriented.
    - SHAP interpretation is optional and requires the 'shap' package.
    """

    def __init__(
        self,
        alpha=0.34,
        beta=0.04,
        gamma=0.01,
        random_state=42,
        cv_splits=5,
        test_size=0.3,
        n_candidate_base=5,
        n_selected_base=3,
        gcra_population_size=20,
        gcra_max_iterations=20,
        verbose=True
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.random_state = random_state
        self.cv_splits = cv_splits
        self.test_size = test_size
        self.n_candidate_base = n_candidate_base
        self.n_selected_base = n_selected_base
        self.gcra_population_size = gcra_population_size
        self.gcra_max_iterations = gcra_max_iterations
        self.verbose = verbose

        # Training-related artifacts
        self.scaler = StandardScaler()
        self.original_feature_names = None
        self.training_feature_fill_values_ = None

        self.selected_features = None
        self.feature_weights = None

        self.candidate_base_learners_ = None
        self.selected_base_learners = None
        self.selected_meta_learner = None

        self.optimized_selection_weights_ = None
        self.learner_weights = None

        self.trained_base_learners = {}
        self.trained_meta_learner = None

        self.last_metrics = None
        self.last_results = None
        self.optimization_fitness_ = None
        self.shap_summary_ = None

        self.is_fitted = False

        # Initialize model pools
        self._initialize_base_learners()
        self._initialize_meta_learners()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _log(self, message):
        if self.verbose:
            print(message)

    def _adaptive_cv(self, n_samples, preferred_splits=None):
        """
        Create a safe KFold object for the current sample size.
        Ensures at least 2 folds and at most n_samples folds.
        """
        preferred = preferred_splits if preferred_splits is not None else self.cv_splits
        n_splits = max(2, min(int(preferred), int(n_samples)))
        return KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

    def _coerce_numeric_dataframe(self, X):
        """Convert dataframe columns to numeric where possible."""
        X = X.copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        return X

    def _sanitize_fill_values(self, fill_values):
        cleaned = {}
        for key, value in fill_values.items():
            if pd.isna(value) or not np.isfinite(value):
                cleaned[key] = 0.0
            else:
                cleaned[key] = float(value)
        return cleaned

    # ------------------------------------------------------------------
    # Learner initialization
    # ------------------------------------------------------------------
    def _initialize_base_learners(self):
        """Initialize pool of base learners."""
        self.base_learners_pool = {
            "RF": RandomForestRegressor(
                n_estimators=120,
                random_state=self.random_state
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=120,
                random_state=self.random_state
            ),
            "DecisionTree": DecisionTreeRegressor(
                random_state=self.random_state
            ),
            "SVR": SVR(kernel="rbf"),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "MLP": MLPRegressor(
                hidden_layer_sizes=(80,),
                max_iter=500,
                random_state=self.random_state
            ),
            "AdaBoost": AdaBoostRegressor(
                n_estimators=120,
                random_state=self.random_state
            ),
            "Bagging": BaggingRegressor(
                n_estimators=80,
                random_state=self.random_state
            ),
        }

        if XGBOOST_AVAILABLE:
            self.base_learners_pool["XGBoost"] = xgb.XGBRegressor(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                verbosity=0
            )

        if LIGHTGBM_AVAILABLE:
            self.base_learners_pool["LightGBM"] = lgb.LGBMRegressor(
                n_estimators=120,
                learning_rate=0.05,
                random_state=self.random_state,
                verbosity=-1
            )

    def _initialize_meta_learners(self):
        """Initialize pool of meta learners."""
        self.meta_learners_pool = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.01),
            "ElasticNet": ElasticNet(alpha=0.01),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=80,
                random_state=self.random_state
            ),
            "AdaBoost": AdaBoostRegressor(
                n_estimators=80,
                random_state=self.random_state
            ),
            "Bagging": BaggingRegressor(
                n_estimators=60,
                random_state=self.random_state
            ),
            "MLP": MLPRegressor(
                hidden_layer_sizes=(50,),
                max_iter=400,
                random_state=self.random_state
            ),
        }

        if XGBOOST_AVAILABLE:
            self.meta_learners_pool["XGBoost"] = xgb.XGBRegressor(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=self.random_state,
                verbosity=0
            )

    # ------------------------------------------------------------------
    # Workflow steps
    # ------------------------------------------------------------------
    def step1_data_preprocessing(self, X, y):
        """Step 1: Data preprocessing."""
        self._log("Step 1: Data preprocessing")
        self._log(f"  Raw feature shape: {X.shape}")
        self._log(f"  Number of samples: {len(X)}")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        elif not isinstance(y, pd.Series):
            y = pd.Series(y, name="target")

        X = self._coerce_numeric_dataframe(X)
        y = pd.to_numeric(y, errors="coerce")

        # Replace inf with nan
        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)

        # Fill missing values
        feature_fill_values = X.median(numeric_only=True).to_dict()
        feature_fill_values = self._sanitize_fill_values(feature_fill_values)

        for col in X.columns:
            fill_val = feature_fill_values.get(col, 0.0)
            X[col] = X[col].fillna(fill_val)

        y_fill = y.median()
        if pd.isna(y_fill) or not np.isfinite(y_fill):
            y_fill = 0.0
        y = y.fillna(float(y_fill))

        self.training_feature_fill_values_ = feature_fill_values

        self._log("  Missing values handled successfully.")
        return X, y

    def step2_train_test_split(self, X, y):
        """Step 2: Train-test split."""
        self._log("Step 2: Training/testing partitioning")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self._log(f"  Training samples: {len(X_train)}")
        self._log(f"  Testing samples: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def step3_feature_analysis_selection(self, X_train, y_train):
        """
        Step 3: Feature analysis and selection using
        RFE + multicollinearity screening + pragmatic ICWCM-style weighting.
        """
        self._log("Step 3: Feature analysis and selection")

        if X_train.shape[1] <= 3:
            final_features = list(X_train.columns)
            feature_weights = self._calculate_icwcm_weights(X_train[final_features], y_train)
            self._log(f"  Feature count small; using all features: {final_features}")
            return X_train[final_features], final_features, feature_weights

        n_features_to_select = min(
            max(int(np.ceil(X_train.shape[1] * 0.7)), 5),
            X_train.shape[1]
        )

        self._log("  Running Recursive Feature Elimination (RFE)...")
        rfe_estimator = RandomForestRegressor(
            n_estimators=60,
            random_state=self.random_state
        )
        rfe = RFE(
            estimator=rfe_estimator,
            n_features_to_select=n_features_to_select
        )
        rfe.fit(X_train, y_train)
        rfe_selected_features = X_train.columns[rfe.support_].tolist()
        self._log(f"  RFE retained {len(rfe_selected_features)} features.")

        # Multicollinearity screening
        X_rfe = X_train[rfe_selected_features].copy()
        high_vif_features = []

        try:
            if X_rfe.shape[1] >= 2:
                for i, feature in enumerate(X_rfe.columns):
                    vif = variance_inflation_factor(X_rfe.values, i)
                    if np.isfinite(vif) and vif > 10:
                        high_vif_features.append(feature)
        except Exception:
            corr_matrix = X_rfe.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_vif_features = [
                column for column in upper_tri.columns
                if any(upper_tri[column] > 0.90)
            ]

        if high_vif_features:
            tentative_features = [
                f for f in rfe_selected_features
                if f not in high_vif_features
            ]
            if len(tentative_features) >= 3:
                final_features = tentative_features
                self._log(f"  Removed high-VIF/high-correlation features: {high_vif_features}")
            else:
                final_features = rfe_selected_features
                self._log("  High-VIF removal would leave too few features; retaining RFE set.")
        else:
            final_features = rfe_selected_features

        X_selected = X_train[final_features]
        feature_weights = self._calculate_icwcm_weights(X_selected, y_train)

        self._log(f"  Final selected features ({len(final_features)}): {final_features}")
        return X_selected, final_features, feature_weights

    def _calculate_icwcm_weights(self, X_selected, y_train):
        """
        Pragmatic ICWCM-style feature weighting.

        This implementation combines:
        - information proxy: feature dispersion (standard deviation)
        - correlation proxy: absolute Spearman correlation with target
        """
        X_selected = X_selected.copy()

        info_scores = X_selected.std(axis=0, ddof=0).values.astype(float)
        if np.allclose(info_scores.sum(), 0):
            info_scores = np.ones(len(info_scores), dtype=float)
        info_scores = info_scores / info_scores.sum()

        corr_scores = []
        for feature in X_selected.columns:
            corr = safe_spearman_correlation(X_selected[feature].values, y_train.values)
            corr_scores.append(abs(corr))
        corr_scores = np.asarray(corr_scores, dtype=float)

        if np.allclose(corr_scores.sum(), 0):
            corr_scores = np.ones(len(corr_scores), dtype=float)
        corr_scores = corr_scores / corr_scores.sum()

        weights = 0.5 * info_scores + 0.5 * corr_scores
        weights = weights / weights.sum()
        return weights

    def _build_oof_matrix(self, X, y, learner_names, cv=None):
        """
        Build out-of-fold prediction matrix for selected learners.
        """
        X = X.copy()
        y = y.copy()
        cv = cv if cv is not None else self._adaptive_cv(len(X), self.cv_splits)

        oof_columns = []
        for learner_name in learner_names:
            estimator = clone(self.base_learners_pool[learner_name])
            try:
                preds = cross_val_predict(
                    estimator,
                    X,
                    y,
                    cv=cv,
                    method="predict",
                    n_jobs=None
                )
            except Exception:
                # Fallback to in-sample predictions if OOF fails
                estimator.fit(X, y)
                preds = estimator.predict(X)
            oof_columns.append(np.asarray(preds, dtype=float))

        return np.column_stack(oof_columns)

    def step4_initial_learner_selection(self, X_train, y_train):
        """
        Step 4: Initial learner screening via CV for base learners and OOF meta-features.
        """
        self._log("Step 4: Initial learner screening")

        cv = self._adaptive_cv(len(X_train), self.cv_splits)
        base_learner_scores = {}

        self._log("  Evaluating base learners with cross-validation...")
        for name, learner in self.base_learners_pool.items():
            try:
                scores = cross_val_score(
                    clone(learner),
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="neg_mean_squared_error",
                    n_jobs=None
                )
                rmse = float(np.mean(np.sqrt(-scores)))
                base_learner_scores[name] = rmse
                self._log(f"    {name}: CV RMSE = {rmse:.6f}")
            except Exception as e:
                base_learner_scores[name] = float("inf")
                self._log(f"    {name}: Failed ({e})")

        candidate_count = min(self.n_candidate_base, len(base_learner_scores))
        sorted_base = sorted(base_learner_scores.items(), key=lambda x: x[1])
        candidate_base_learners = [name for name, _ in sorted_base[:candidate_count]]
        self.candidate_base_learners_ = candidate_base_learners
        self._log(f"  Candidate base learners: {candidate_base_learners}")

        self._log("  Building OOF meta-features for meta-learner screening...")
        oof_meta_features = self._build_oof_matrix(
            X_train,
            y_train,
            candidate_base_learners,
            cv=cv
        )

        meta_learner_scores = {}
        self._log("  Evaluating meta learners...")
        for name, learner in self.meta_learners_pool.items():
            try:
                scores = cross_val_score(
                    clone(learner),
                    oof_meta_features,
                    y_train,
                    cv=cv,
                    scoring="neg_mean_squared_error",
                    n_jobs=None
                )
                rmse = float(np.mean(np.sqrt(-scores)))
                meta_learner_scores[name] = rmse
                self._log(f"    {name}: CV RMSE = {rmse:.6f}")
            except Exception as e:
                meta_learner_scores[name] = float("inf")
                self._log(f"    {name}: Failed ({e})")

        best_meta_learner = min(meta_learner_scores.items(), key=lambda x: x[1])[0]
        self._log(f"  Selected candidate meta learner: {best_meta_learner}")

        return candidate_base_learners, best_meta_learner

    def _decode_solution(self, solution, candidate_base_learners):
        """
        Decode GCRA solution into selected learners and normalized weights.
        The solution format is:
          [selection_scores..., raw_weight_scores...]
        """
        n_candidates = len(candidate_base_learners)
        if solution is None or len(solution) != 2 * n_candidates:
            selected = candidate_base_learners[:min(self.n_selected_base, n_candidates)]
            weights = np.ones(len(selected), dtype=float)
            weights = weights / weights.sum()
            return selected, weights

        selection_scores = np.asarray(solution[:n_candidates], dtype=float)
        raw_weights = np.asarray(solution[n_candidates:], dtype=float)

        selected_count = min(self.n_selected_base, n_candidates)
        selected_indices = np.argsort(selection_scores)[-selected_count:]
        selected_indices = selected_indices[np.argsort(selection_scores[selected_indices])[::-1]]

        selected_learners = [candidate_base_learners[i] for i in selected_indices]
        selected_weights = raw_weights[selected_indices]

        if np.allclose(selected_weights.sum(), 0):
            selected_weights = np.ones(len(selected_learners), dtype=float)

        selected_weights = selected_weights / selected_weights.sum()
        return selected_learners, selected_weights

    def _evaluate_learner_combination(
        self,
        individual,
        X_train,
        y_train,
        candidate_base_learners,
        candidate_meta_learner
    ):
        """
        Objective function for GCRA optimization.

        Fitness = mean CV RMSE
                  + alpha * stability penalty
                  + beta  * complexity penalty
                  + gamma * redundancy penalty
        """
        try:
            selected_learners, selected_weights = self._decode_solution(
                individual,
                candidate_base_learners
            )

            if len(selected_learners) == 0:
                return float("inf")

            outer_cv = self._adaptive_cv(
                len(X_train),
                preferred_splits=min(3, self.cv_splits)
            )

            fold_rmses = []
            fold_redundancy = []

            for train_idx, valid_idx in outer_cv.split(X_train):
                X_tr = X_train.iloc[train_idx]
                X_val = X_train.iloc[valid_idx]
                y_tr = y_train.iloc[train_idx]
                y_val = y_train.iloc[valid_idx]

                inner_cv = self._adaptive_cv(
                    len(X_tr),
                    preferred_splits=min(3, self.cv_splits)
                )

                # OOF predictions on training fold for meta-learner fitting
                oof_matrix = self._build_oof_matrix(
                    X_tr,
                    y_tr,
                    selected_learners,
                    cv=inner_cv
                )
                weighted_oof = oof_matrix * selected_weights

                meta_learner = clone(self.meta_learners_pool[candidate_meta_learner])
                meta_learner.fit(weighted_oof, y_tr)

                # Fit base learners on full training fold and predict validation fold
                val_columns = []
                for learner_name, weight in zip(selected_learners, selected_weights):
                    base_model = clone(self.base_learners_pool[learner_name])
                    base_model.fit(X_tr, y_tr)
                    val_pred = np.asarray(base_model.predict(X_val), dtype=float) * weight
                    val_columns.append(val_pred)

                val_matrix = np.column_stack(val_columns)
                val_pred_final = meta_learner.predict(val_matrix)

                rmse = np.sqrt(mean_squared_error(y_val, val_pred_final))
                fold_rmses.append(float(rmse))

                if val_matrix.shape[1] > 1:
                    corr = np.corrcoef(val_matrix.T)
                    upper = corr[np.triu_indices_from(corr, k=1)]
                    upper = upper[np.isfinite(upper)]
                    redundancy = float(np.mean(np.abs(upper))) if upper.size > 0 else 0.0
                else:
                    redundancy = 0.0
                fold_redundancy.append(redundancy)

            mean_rmse = float(np.mean(fold_rmses))
            stability_penalty = float(np.std(fold_rmses))
            complexity_penalty = float(len(selected_learners) / max(len(candidate_base_learners), 1))
            redundancy_penalty = float(np.mean(fold_redundancy))

            fitness = (
                mean_rmse
                + self.alpha * stability_penalty
                + self.beta * complexity_penalty
                + self.gamma * redundancy_penalty
            )
            return float(fitness)

        except Exception:
            return float("inf")

    def step5_dynamic_learner_selection_optimization(
        self,
        X_train,
        y_train,
        candidate_base_learners,
        candidate_meta_learner
    ):
        """
        Step 5: Dynamic learner selection and optimization using GCRA.
        Falls back gracefully if GCRA is unavailable.
        """
        self._log("Step 5: Dynamic learner optimization")

        try:
            from gcra_optimizer import GCRA
            gcra_available = True
        except Exception:
            gcra_available = False

        if not gcra_available:
            self._log("  GCRA unavailable; using fallback learner selection.")
            self.selected_base_learners = candidate_base_learners[:min(self.n_selected_base, len(candidate_base_learners))]
            self.selected_meta_learner = candidate_meta_learner
            self.optimized_selection_weights_ = np.ones(len(self.selected_base_learners), dtype=float)
            self.optimized_selection_weights_ /= self.optimized_selection_weights_.sum()
            self.optimization_fitness_ = None
            return None, None

        def objective_function(individual):
            return self._evaluate_learner_combination(
                individual,
                X_train,
                y_train,
                candidate_base_learners,
                candidate_meta_learner
            )

        n_candidates = len(candidate_base_learners)
        dimensions = 2 * n_candidates
        bounds = [(0.0, 1.0)] * dimensions

        self._log(
            f"  Running GCRA "
            f"(population={self.gcra_population_size}, iterations={self.gcra_max_iterations})..."
        )

        try:
            gcra = GCRA(
                objective_function=objective_function,
                population_size=self.gcra_population_size,
                max_iterations=self.gcra_max_iterations,
                dimensions=dimensions,
                bounds=bounds
            )
            best_solution, best_fitness = gcra.optimize()
        except Exception as e:
            self._log(f"  GCRA failed ({e}); using fallback learner selection.")
            self.selected_base_learners = candidate_base_learners[:min(self.n_selected_base, len(candidate_base_learners))]
            self.selected_meta_learner = candidate_meta_learner
            self.optimized_selection_weights_ = np.ones(len(self.selected_base_learners), dtype=float)
            self.optimized_selection_weights_ /= self.optimized_selection_weights_.sum()
            self.optimization_fitness_ = None
            return None, None

        selected_learners, selected_weights = self._decode_solution(
            best_solution,
            candidate_base_learners
        )

        self.selected_base_learners = selected_learners
        self.selected_meta_learner = candidate_meta_learner
        self.optimized_selection_weights_ = selected_weights
        self.optimization_fitness_ = best_fitness

        self._log(f"  Optimized base learners: {self.selected_base_learners}")
        self._log(f"  Optimized prior weights: {np.round(self.optimized_selection_weights_, 4)}")
        self._log(f"  Best fitness: {best_fitness:.6f}")

        return best_solution, best_fitness

    def step6_model_training_optimization(self, X_train, y_train):
        """
        Step 6: Train final base learners and meta learner using OOF stacking.
        """
        self._log("Step 6: Final stacked model training")

        if self.selected_base_learners is None or len(self.selected_base_learners) == 0:
            raise ValueError("No base learners selected before final training.")

        cv = self._adaptive_cv(len(X_train), self.cv_splits)

        # Build OOF meta-features for final meta-learner training
        oof_matrix = self._build_oof_matrix(
            X_train,
            y_train,
            self.selected_base_learners,
            cv=cv
        )

        # Performance-derived learner weights
        performance_weights = []
        for i in range(oof_matrix.shape[1]):
            rmse_i = np.sqrt(mean_squared_error(y_train, oof_matrix[:, i]))
            performance_weights.append(1.0 / (rmse_i + 1e-8))
        performance_weights = np.asarray(performance_weights, dtype=float)
        performance_weights = performance_weights / performance_weights.sum()

        # Combine optimization priors with performance-derived weights
        if (
            self.optimized_selection_weights_ is not None
            and len(self.optimized_selection_weights_) == len(performance_weights)
        ):
            combined_weights = (
                self.alpha * self.optimized_selection_weights_
                + (1.0 - self.alpha) * performance_weights
            )
        else:
            combined_weights = performance_weights

        combined_weights = combined_weights / combined_weights.sum()
        self.learner_weights = combined_weights

        self._log(f"  Final learner weights: {np.round(self.learner_weights, 4)}")

        weighted_oof = oof_matrix * self.learner_weights
        meta_model = clone(self.meta_learners_pool[self.selected_meta_learner])
        meta_model.fit(weighted_oof, y_train)
        self.trained_meta_learner = meta_model

        # Train final base learners on the full training data
        self.trained_base_learners = {}
        for learner_name in self.selected_base_learners:
            self._log(f"  Training final base learner: {learner_name}")
            model = clone(self.base_learners_pool[learner_name])
            model.fit(X_train, y_train)
            self.trained_base_learners[learner_name] = model

        self._log("  Final stacked model trained successfully.")

    def step7_model_validation(self, X_test, y_test):
        """Step 7: Model validation."""
        self._log("Step 7: Model validation")

        y_pred = self._predict_internal(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        mape = safe_mape(y_test, y_pred)
        spearman_corr = safe_spearman_correlation(y_test, y_pred)

        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "Spearman_Correlation": spearman_corr
        }

        self._log(f"  RMSE: {rmse:.6f}")
        self._log(f"  MAE: {mae:.6f}")
        self._log(f"  MAPE: {mape:.2f}%")
        self._log(f"  Spearman Correlation: {spearman_corr:.6f}")

        self.last_metrics = metrics
        return metrics

    def step8_optimization_analysis(self):
        """Step 8: Training summary."""
        self._log("Step 8: Optimization and training summary")
        self._log(f"  Selected base learners: {self.selected_base_learners}")
        self._log(f"  Selected meta learner: {self.selected_meta_learner}")
        if self.learner_weights is not None:
            self._log(f"  Final learner weights: {np.round(self.learner_weights, 4)}")
        if self.optimization_fitness_ is not None:
            self._log(f"  Optimization fitness: {self.optimization_fitness_:.6f}")

    # ------------------------------------------------------------------
    # Feature preparation and prediction
    # ------------------------------------------------------------------
    def _prepare_features(self, X):
        """
        Prepare raw feature dataframe for prediction.

        Rules:
        - Extra columns are ignored
        - Missing non-selected training columns are filled with training medians
        - Missing selected columns raise an error
        """
        if self.original_feature_names is None:
            raise ValueError("Model has no stored feature schema. Fit the model first.")

        if isinstance(X, np.ndarray):
            if X.ndim != 2 or X.shape[1] != len(self.original_feature_names):
                raise ValueError(
                    "When passing a NumPy array to predict(), it must have the same "
                    "number of columns as the original training data."
                )
            X = pd.DataFrame(X, columns=self.original_feature_names)

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input features must be a pandas DataFrame or compatible 2D NumPy array.")

        X = X.copy()

        # Ensure required selected features exist
        for feature in self.selected_features:
            if feature not in X.columns:
                raise ValueError(f"Missing required selected feature: {feature}")

        # Add missing non-selected original columns
        for col in self.original_feature_names:
            if col not in X.columns:
                fill_val = self.training_feature_fill_values_.get(col, 0.0)
                X[col] = fill_val

        # Keep only original training columns in original order
        X = X[self.original_feature_names]
        X = self._coerce_numeric_dataframe(X)

        for col in X.columns:
            fill_val = self.training_feature_fill_values_.get(col, 0.0)
            X[col] = X[col].fillna(fill_val)

        return X

    def transform_selected_features(self, X):
        """
        Public helper: transform raw input into scaled selected-feature space.
        """
        X_prepared = self._prepare_features(X)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_prepared),
            columns=self.original_feature_names,
            index=X_prepared.index
        )
        return X_scaled[self.selected_features]

    def _predict_from_selected_scaled(self, X_selected_scaled):
        """Predict from already scaled selected-feature space."""
        if isinstance(X_selected_scaled, np.ndarray):
            X_selected_scaled = pd.DataFrame(
                X_selected_scaled,
                columns=self.selected_features
            )

        base_predictions = []
        for learner_name, weight in zip(self.selected_base_learners, self.learner_weights):
            learner = self.trained_base_learners[learner_name]
            pred = np.asarray(learner.predict(X_selected_scaled), dtype=float) * weight
            base_predictions.append(pred)

        meta_features = np.column_stack(base_predictions)
        final_pred = self.trained_meta_learner.predict(meta_features)
        return np.asarray(final_pred, dtype=float)

    def _predict_internal(self, X):
        """Internal prediction without fitted-state check."""
        X_selected_scaled = self.transform_selected_features(X)
        return self._predict_from_selected_scaled(X_selected_scaled)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X, y):
        """
        Fit the DOME model.

        Parameters
        ----------
        X : pandas.DataFrame
            Conditioning factors / explanatory variables.
        y : pandas.Series or array-like
            Susceptibility target (legacy 'Risk_index' also supported).

        Returns
        -------
        dict
            Training and validation summary.
        """
        self._log("=" * 80)
        self._log("DOME Training Workflow")
        self._log("Inventory-Based Regional Debris-Flow Susceptibility Assessment")
        self._log("=" * 80)

        # Step 1
        X, y = self.step1_data_preprocessing(X, y)

        self.original_feature_names = list(X.columns)
        if self.training_feature_fill_values_ is None:
            fills = X.median(numeric_only=True).to_dict()
            self.training_feature_fill_values_ = self._sanitize_fill_values(fills)

        # Step 2
        X_train, X_test, y_train, y_test = self.step2_train_test_split(X, y)

        # Scale full feature space
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        # Step 3
        X_train_selected, selected_features, feature_weights = self.step3_feature_analysis_selection(
            X_train_scaled,
            y_train
        )
        X_test_selected = X_test_scaled[selected_features]

        self.selected_features = selected_features
        self.feature_weights = feature_weights

        # Step 4
        candidate_base_learners, candidate_meta_learner = self.step4_initial_learner_selection(
            X_train_selected,
            y_train
        )

        # Step 5
        self.step5_dynamic_learner_selection_optimization(
            X_train_selected,
            y_train,
            candidate_base_learners,
            candidate_meta_learner
        )

        # Step 6
        self.step6_model_training_optimization(X_train_selected, y_train)

        # Mark fitted before external-style validation helpers if needed
        self.is_fitted = True

        # Step 7
        validation_metrics = self.step7_model_validation(X_test, y_test)

        # Step 8
        self.step8_optimization_analysis()

        results = {
            "selected_base_learners": self.selected_base_learners,
            "selected_meta_learner": self.selected_meta_learner,
            "candidate_base_learners": candidate_base_learners,
            "optimized_prior_weights": (
                self.optimized_selection_weights_.tolist()
                if self.optimized_selection_weights_ is not None
                else None
            ),
            "learner_weights": (
                self.learner_weights.tolist()
                if self.learner_weights is not None
                else None
            ),
            "selected_features": self.selected_features,
            "feature_weights": self.feature_weights.tolist() if self.feature_weights is not None else None,
            "performance_metrics": validation_metrics,
            "training_samples": int(len(X_train)),
            "testing_samples": int(len(X_test)),
            "total_features": int(len(X.columns)),
            "selected_feature_count": int(len(self.selected_features)),
            "optimization_fitness": (
                float(self.optimization_fitness_)
                if self.optimization_fitness_ is not None
                else None
            )
        }

        self.last_results = results

        self._log("=" * 80)
        self._log("DOME training completed successfully.")
        self._log("=" * 80)

        return results

    def predict(self, X):
        """
        Predict susceptibility values for new samples.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        return self._predict_internal(X)

    def get_feature_importance(self):
        """
        Return selected-feature importance/weight dictionary.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance.")

        return {
            feature: float(weight)
            for feature, weight in zip(self.selected_features, self.feature_weights)
        }

    def get_model_summary(self):
        """
        Return model configuration and fitted summary.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before requesting summary.")

        return {
            "model_type": "DOME (Dynamic Optimization Meta-Ensemble)",
            "application": "Inventory-based regional debris-flow susceptibility assessment",
            "selected_base_learners": self.selected_base_learners,
            "selected_meta_learner": self.selected_meta_learner,
            "candidate_base_learners": self.candidate_base_learners_,
            "learner_weights": self.learner_weights.tolist() if self.learner_weights is not None else None,
            "selected_features": self.selected_features,
            "feature_weights": self.feature_weights.tolist() if self.feature_weights is not None else None,
            "optimization_fitness": self.optimization_fitness_,
            "hyperparameters": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "random_state": self.random_state,
                "cv_splits": self.cv_splits,
                "test_size": self.test_size,
                "n_candidate_base": self.n_candidate_base,
                "n_selected_base": self.n_selected_base,
                "gcra_population_size": self.gcra_population_size,
                "gcra_max_iterations": self.gcra_max_iterations
            },
            "shap_available": SHAP_AVAILABLE,
            "last_metrics": self.last_metrics
        }

    # ------------------------------------------------------------------
    # SHAP interpretation
    # ------------------------------------------------------------------
    def _predict_selected_scaled_array(self, X_array):
        """
        Wrapper for SHAP KernelExplainer.
        Input is assumed to already be in the scaled selected-feature space.
        """
        X_df = pd.DataFrame(X_array, columns=self.selected_features)
        return self._predict_from_selected_scaled(X_df)

    def get_shap_explanations(self, X, background_size=50, explain_size=None, nsamples="auto"):
        """
        Compute model-agnostic SHAP explanations in the selected scaled-feature space.

        Parameters
        ----------
        X : pandas.DataFrame
            Raw input dataframe.
        background_size : int
            Number of background samples for KernelExplainer.
        explain_size : int or None
            Number of rows to explain. If None, explain all rows in X.
        nsamples : int or "auto"
            SHAP sampling budget.

        Returns
        -------
        dict
            {
              "summary": pandas.DataFrame,
              "shap_values": np.ndarray,
              "expected_value": float,
              "feature_names": list
            }
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "The 'shap' package is not installed. "
                "Run `pip install shap` to enable SHAP-based interpretation."
            )

        if not self.is_fitted:
            raise ValueError("Model must be fitted before SHAP explanation.")

        X_selected_scaled = self.transform_selected_features(X)

        if len(X_selected_scaled) == 0:
            raise ValueError("No samples available for SHAP explanation.")

        background_n = max(1, min(int(background_size), len(X_selected_scaled)))
        explain_n = len(X_selected_scaled) if explain_size is None else max(1, min(int(explain_size), len(X_selected_scaled)))

        background_df = X_selected_scaled.sample(
            n=background_n,
            random_state=self.random_state
        )
        explain_df = X_selected_scaled.iloc[:explain_n].copy()

        explainer = shap.KernelExplainer(
            self._predict_selected_scaled_array,
            background_df.values
        )

        shap_values = explainer.shap_values(
            explain_df.values,
            nsamples=nsamples
        )

        # For regression, shap_values is usually a 2D array.
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values, dtype=float)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        summary_df = pd.DataFrame({
            "feature": self.selected_features,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(np.asarray(expected_value).reshape(-1)[0])
        else:
            expected_value = float(expected_value)

        self.shap_summary_ = summary_df

        return {
            "summary": summary_df,
            "shap_values": shap_values,
            "expected_value": expected_value,
            "feature_names": self.selected_features
        }


def main():
    """
    Example usage.
    """
    print("DOME Example")
    print("=" * 40)

    try:
        df = pd.read_excel("CPEC_debris_flow_dataset_3447.xlsx")
        target_col = resolve_target_column(df)
        print(f"Detected target column: {target_col}")
        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        model = DOMEModel(
            alpha=0.34,
            beta=0.04,
            gamma=0.01,
            random_state=42,
            cv_splits=5,
            verbose=True
        )

        results = model.fit(X, y)

        print("\nTraining Results")
        print("-" * 40)
        print(f"Selected base learners: {results['selected_base_learners']}")
        print(f"Selected meta learner: {results['selected_meta_learner']}")
        print(f"Metrics: {results['performance_metrics']}")

        sample_X = X.head(5)
        predictions = model.predict(sample_X)
        print(f"\nSample predictions: {predictions}")

        if SHAP_AVAILABLE:
            print("\nGenerating SHAP summary on a small sample...")
            shap_result = model.get_shap_explanations(
                X.head(min(30, len(X))),
                background_size=min(15, len(X)),
                explain_size=min(15, len(X)),
                nsamples=50
            )
            print(shap_result["summary"].head())
        else:
            print("\nSHAP not installed; skipping SHAP summary.")

    except FileNotFoundError:
        print("Dataset file not found: CPEC_debris_flow_dataset_3447.xlsx")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
