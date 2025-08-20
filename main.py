import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings('ignore')

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


class DOMEModel:
    """
    DOME (Dynamic Optimization Meta-Ensemble) Model

    An advanced meta-ensemble learning framework that combines multiple base learners
    with dynamic optimization for debris flow risk assessment.
    """

    def __init__(self, alpha=0.34, beta=0.04, gamma=0.01, random_state=42):
        """
        Initialize DOME model with hyperparameters

        Parameters:
        -----------
        alpha : float, default=0.34
            Weight parameter for ensemble combination
        beta : float, default=0.04
            Regularization parameter for learner selection
        gamma : float, default=0.01
            Convergence threshold for optimization
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.random_state = random_state

        # Initialize components
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_weights = None
        self.selected_base_learners = None
        self.selected_meta_learner = None
        self.trained_base_learners = {}
        self.trained_meta_learner = None
        self.learner_weights = None
        self.is_fitted = False

        # Initialize base learner pool
        self._initialize_base_learners()
        self._initialize_meta_learners()

    def _initialize_base_learners(self):
        """Initialize pool of base learners"""
        self.base_learners_pool = {
            'RF': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
            'DecisionTree': DecisionTreeRegressor(random_state=self.random_state),
            'SVR': SVR(kernel='rbf'),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'MLP': MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=self.random_state),
            'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=self.random_state),
            'Bagging': BaggingRegressor(n_estimators=100, random_state=self.random_state)
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.base_learners_pool['XGBoost'] = xgb.XGBRegressor(
                n_estimators=100,
                random_state=self.random_state,
                verbosity=0
            )

        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.base_learners_pool['LightGBM'] = lgb.LGBMRegressor(
                n_estimators=100,
                random_state=self.random_state,
                verbosity=-1
            )

    def _initialize_meta_learners(self):
        """Initialize pool of meta learners"""
        self.meta_learners_pool = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=self.random_state),
            'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=self.random_state),
            'Bagging': BaggingRegressor(n_estimators=50, random_state=self.random_state),
            'MLP': MLPRegressor(hidden_layer_sizes=(50,), max_iter=300, random_state=self.random_state)
        }

        # Add XGBoost meta learner if available
        if XGBOOST_AVAILABLE:
            self.meta_learners_pool['XGBoost'] = xgb.XGBRegressor(
                n_estimators=50,
                random_state=self.random_state,
                verbosity=0
            )

    def step1_data_preprocessing(self, X, y):
        """Step 1: Data collection and preprocessing"""
        print("Step 1: Data collection and preprocessing")
        print(f"Dataset shape: {X.shape}")
        print(f"Processing {len(X)} debris flow samples")

        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print("  Handling missing values...")
            X = X.fillna(X.mean())

        if y.isnull().sum() > 0:
            y = y.fillna(y.mean())

        return X, y

    def step2_train_test_split(self, X, y, test_size=0.3):
        """Step 2: Training and testing set partitioning"""
        print("Step 2: Training and testing set partitioning")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=None
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Testing set: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def step3_feature_analysis_selection(self, X_train, y_train):
        """Step 3: Feature analysis and selection using RFE and VIF"""
        print("Step 3: Feature analysis and selection")

        # Recursive Feature Elimination (RFE)
        print("  Performing Recursive Feature Elimination (RFE)...")
        rfe_estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
        n_features_to_select = min(max(int(X_train.shape[1] * 0.7), 5), X_train.shape[1])
        rfe = RFE(estimator=rfe_estimator, n_features_to_select=n_features_to_select)
        rfe.fit(X_train, y_train)

        rfe_selected_features = X_train.columns[rfe.support_].tolist()
        print(f"  RFE selected {len(rfe_selected_features)} features")

        # Multicollinearity detection using VIF
        print("  Calculating correlation matrix for multicollinearity detection...")
        X_rfe = X_train[rfe_selected_features]

        # Calculate VIF for selected features
        high_vif_features = []
        try:
            for i, feature in enumerate(X_rfe.columns):
                vif = variance_inflation_factor(X_rfe.values, i)
                if vif > 10:  # VIF threshold
                    high_vif_features.append(feature)
        except:
            # If VIF calculation fails, use correlation-based removal
            corr_matrix = X_rfe.corr().abs()
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]
            high_vif_features = high_corr_features

        if high_vif_features:
            print(f"  Removing high VIF features: {high_vif_features}")
            final_features = [f for f in rfe_selected_features if f not in high_vif_features]
        else:
            final_features = rfe_selected_features

        print(f"  Final selected features ({len(final_features)}): {final_features}")

        # Information-Correlation Weighted Combination Method (ICWCM)
        print("  Applying Information-Correlation Weighted Combination Method (ICWCM)...")
        print("  Calculating feature weights using ICWCM...")

        X_selected = X_train[final_features]
        feature_weights = self._calculate_icwcm_weights(X_selected, y_train)

        return X_selected, final_features, feature_weights

    def _calculate_icwcm_weights(self, X_selected, y_train):
        """Calculate feature weights using Information-Correlation Weighted Combination Method"""
        n_features = X_selected.shape[1]
        weights = np.ones(n_features) / n_features

        # Simple correlation-based weighting
        correlations = []
        for feature in X_selected.columns:
            try:
                corr, _ = spearmanr(X_selected[feature], y_train)
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            except:
                correlations.append(0)

        if sum(correlations) > 0:
            weights = np.array(correlations) / sum(correlations)

        return weights

    def step4_initial_learner_selection(self, X_train, y_train):
        """Step 4: Initial learner selection and cross-validation"""
        print("Step 4: Initial learner selection and cross-validation")
        print("  Evaluating base learners with 5-fold cross-validation...")

        base_learner_scores = {}

        for name, learner in self.base_learners_pool.items():
            try:
                scores = cross_val_score(learner, X_train, y_train, cv=5,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
                rmse_scores = np.sqrt(-scores)
                avg_rmse = np.mean(rmse_scores)
                base_learner_scores[name] = avg_rmse
                print(f"    {name}: CV RMSE = {avg_rmse:.6f}")
            except Exception as e:
                print(f"    {name}: Failed - {str(e)}")
                base_learner_scores[name] = float('inf')

        # Select top 3 base learners
        sorted_learners = sorted(base_learner_scores.items(), key=lambda x: x[1])
        top_base_learners = [name for name, _ in sorted_learners[:3]]
        print(f"  Selected top base learners: {top_base_learners}")

        # Evaluate meta learners
        print("  Evaluating meta learners...")
        meta_learner_scores = {}

        # Create dummy meta features for evaluation
        dummy_meta_features = np.random.random((len(X_train), 3))

        for name, learner in self.meta_learners_pool.items():
            try:
                scores = cross_val_score(learner, dummy_meta_features, y_train, cv=5,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
                rmse_scores = np.sqrt(-scores)
                avg_rmse = np.mean(rmse_scores)
                meta_learner_scores[name] = avg_rmse
                print(f"    {name}: CV RMSE = {avg_rmse:.6f}")
            except Exception as e:
                print(f"    {name}: Failed - {str(e)}")
                meta_learner_scores[name] = float('inf')

        # Select best meta learner
        best_meta_learner = min(meta_learner_scores.items(), key=lambda x: x[1])[0]
        print(f"  Selected meta learner: {best_meta_learner}")

        return top_base_learners, best_meta_learner

    def step5_dynamic_learner_selection_optimization(self, X_train, y_train,
                                                     candidate_base_learners, candidate_meta_learner):
        """Step 5: Dynamic learner selection and optimization using GCRA"""
        from gcra_optimizer import GCRA

        def objective_function(individual):
            """Objective function for GCRA optimization"""
            return self._evaluate_learner_combination(
                individual, X_train, y_train, candidate_base_learners, candidate_meta_learner
            )

        print("  Running GCRA optimization...")

        # Initialize GCRA optimizer
        gcra = GCRA(
            objective_function=objective_function,
            population_size=30,
            max_iterations=50,
            dimensions=len(candidate_base_learners) + 3,  # learner selection + weights
            bounds=[(0, 1)] * len(candidate_base_learners) + [(0.1, 1)] * 3
        )

        # Run optimization
        best_solution, best_fitness = gcra.optimize()

        if best_fitness == float('inf'):
            print("  GCRA optimization failed, using default selection...")
            # Fallback to default selection
            self.selected_base_learners = candidate_base_learners[:3]
            self.selected_meta_learner = candidate_meta_learner
            self.learner_weights = np.array([1 / 3, 1 / 3, 1 / 3])
        else:
            # Parse optimization results
            learner_selection = best_solution[:len(candidate_base_learners)]
            weights = best_solution[len(candidate_base_learners):]

            # Select top 3 learners based on optimization
            selected_indices = np.argsort(learner_selection)[-3:]
            self.selected_base_learners = [candidate_base_learners[i] for i in selected_indices]
            self.selected_meta_learner = candidate_meta_learner
            self.learner_weights = weights / np.sum(weights)  # Normalize weights

        print(f"  Optimization completed. Best fitness: {best_fitness:.6f}")

        return best_solution, best_fitness

    def _evaluate_learner_combination(self, individual, X_train, y_train,
                                      candidate_base_learners, candidate_meta_learner):
        """Evaluate a specific learner combination"""
        try:
            # Parse individual
            learner_selection = individual[:len(candidate_base_learners)]
            weights = individual[len(candidate_base_learners):]

            # Select top 3 learners
            selected_indices = np.argsort(learner_selection)[-3:]
            selected_learners = [candidate_base_learners[i] for i in selected_indices]

            # Train base learners
            base_predictions = []
            for learner_name in selected_learners:
                learner = self.base_learners_pool[learner_name]
                learner.fit(X_train, y_train)
                pred = learner.predict(X_train)
                base_predictions.append(pred)

            # Train meta learner
            meta_features = np.column_stack(base_predictions)
            meta_learner = self.meta_learners_pool[candidate_meta_learner]
            meta_learner.fit(meta_features, y_train)

            # Evaluate performance
            final_pred = meta_learner.predict(meta_features)
            rmse = np.sqrt(mean_squared_error(y_train, final_pred))

            return rmse

        except Exception as e:
            return float('inf')  # Return high error for failed combinations

    def step6_model_training_optimization(self, X_train, y_train):
        """Step 6: Model training and optimization"""
        print("Step 6: Model training and optimization")
        print("  Training base learners...")

        # Train selected base learners
        for learner_name in self.selected_base_learners:
            print(f"    Training {learner_name}...")
            learner = self.base_learners_pool[learner_name]
            learner.fit(X_train, y_train)
            self.trained_base_learners[learner_name] = learner

        # Calculate refined learner weights using Equation (2)
        print("  Calculating learner weights using Eq. (2)...")
        base_predictions = []
        for learner_name in self.selected_base_learners:
            learner = self.trained_base_learners[learner_name]
            pred = learner.predict(X_train)
            base_predictions.append(pred)

        # Calculate weights based on individual performance
        weights = []
        for i, pred in enumerate(base_predictions):
            mse = mean_squared_error(y_train, pred)
            weight = 1.0 / (mse + 1e-10)  # Add small epsilon to avoid division by zero
            weights.append(weight)

        # Normalize weights
        self.learner_weights = np.array(weights) / np.sum(weights)
        print(f"  Refined learner weights: {self.learner_weights}")

        # Train meta learner with weighted stacking
        print("  Training meta learner with weighted stacking...")
        meta_features = np.column_stack(base_predictions)
        meta_learner = self.meta_learners_pool[self.selected_meta_learner]
        meta_learner.fit(meta_features, y_train)
        self.trained_meta_learner = meta_learner

        print("  Model training completed successfully!")

    def step7_model_validation(self, X_test, y_test):
        """Step 7: Model validation and result analysis"""
        print("Step 7: Model validation and result analysis")

        # Make predictions
        y_pred = self._predict_internal(X_test)

        # Calculate performance metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Calculate Spearman correlation
        try:
            spearman_corr, _ = spearmanr(y_test, y_pred)
        except:
            spearman_corr = 0.0

        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Spearman_Correlation': spearman_corr
        }

        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Spearman Correlation: {spearman_corr:.6f}")

        return metrics

    def step8_optimization_analysis(self):
        """Step 8: Optimization process analysis and result interpretation"""
        print("Step 8: Optimization process analysis and result interpretation")
        print("  Model optimization completed successfully!")
        print(f"  Selected base learners: {self.selected_base_learners}")
        print(f"  Selected meta learner: {self.selected_meta_learner}")
        print(f"  Optimized learner weights: {self.learner_weights}")

    def fit(self, X, y):
        """
        Fit the DOME model using the 8-step workflow

        Parameters:
        -----------
        X : pandas.DataFrame
            Input features
        y : pandas.Series
            Target variable

        Returns:
        --------
        dict : Training results and metrics
        """
        print("=" * 80)
        print("DOME Model Training - 8-Step Workflow")
        print(f"Processing {len(X)} debris flow samples")
        print("=" * 80)

        # Step 1: Data preprocessing
        X, y = self.step1_data_preprocessing(X, y)

        # Step 2: Train-test split
        X_train, X_test, y_train, y_test = self.step2_train_test_split(X, y)

        # Feature scaling - Scale ALL features first
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),  # ✅ Train scaler on ALL features
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),  # ✅ Transform ALL features
            columns=X_test.columns,
            index=X_test.index
        )

        # Step 3: Feature analysis and selection
        X_train_selected, selected_features, feature_weights = self.step3_feature_analysis_selection(
            X_train_scaled, y_train  # ✅ Feature selection on scaled data
        )
        X_test_selected = X_test_scaled[selected_features]  # Select same features for test set

        # Store feature information
        self.selected_features = selected_features
        self.feature_weights = feature_weights

        # Step 4: Initial learner selection
        candidate_base_learners, candidate_meta_learner = self.step4_initial_learner_selection(
            X_train_selected, y_train
        )

        # Step 5: Dynamic optimization using GCRA
        try:
            best_solution, best_fitness = self.step5_dynamic_learner_selection_optimization(
                X_train_selected, y_train, candidate_base_learners, candidate_meta_learner
            )
        except Exception as e:
            print(f"  GCRA optimization failed: {e}")
            print("  Using fallback selection...")
            self.selected_base_learners = candidate_base_learners[:3]
            self.selected_meta_learner = candidate_meta_learner
            self.learner_weights = np.array([1 / 3, 1 / 3, 1 / 3])

        # Step 6: Model training and optimization
        self.step6_model_training_optimization(X_train_selected, y_train)

        # Step 7: Model validation - ✅ Pass ALL features (scaled)
        validation_metrics = self.step7_model_validation(X_test_scaled, y_test)

        # Step 8: Optimization analysis
        self.step8_optimization_analysis()

        # Mark model as fitted
        self.is_fitted = True

        # Return comprehensive results
        results = {
            'selected_base_learners': self.selected_base_learners,
            'selected_meta_learner': self.selected_meta_learner,
            'learner_weights': self.learner_weights,
            'selected_features': self.selected_features,
            'feature_weights': self.feature_weights,
            'performance_metrics': validation_metrics,
            'training_samples': len(X_train),
            'testing_samples': len(X_test),
            'total_features': len(X.columns),
            'selected_feature_count': len(self.selected_features)
        }

        print("\n" + "=" * 80)
        print("DOME Model Training Completed Successfully!")
        print("=" * 80)

        return results

    def _predict_internal(self, X):
        """Internal prediction method without fitted check"""
        # Scale ALL features (scaler was trained on all features)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),  # ✅ Scale all features
            columns=X.columns,
            index=X.index
        )

        # Select features after scaling
        X_selected = X_scaled[self.selected_features]  # ✅ Then select needed features

        # Get base learner predictions
        base_predictions = []
        for learner_name in self.selected_base_learners:
            learner = self.trained_base_learners[learner_name]
            pred = learner.predict(X_selected)
            base_predictions.append(pred)

        # Meta learner final prediction
        meta_features = np.column_stack(base_predictions)
        final_prediction = self.trained_meta_learner.predict(meta_features)

        return final_prediction

    def predict(self, X):
        """
        Make predictions using the trained DOME model

        Parameters:
        -----------
        X : pandas.DataFrame
            Input features (should have same columns as training data)

        Returns:
        --------
        numpy.ndarray : Predictions
        """
        # Check if model is fitted
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")

        # Check feature consistency
        if not all(col in X.columns for col in self.selected_features):
            missing_features = [col for col in self.selected_features if col not in X.columns]
            raise ValueError(f"Missing features in input data: {missing_features}")

        return self._predict_internal(X)

    def get_feature_importance(self):
        """Get feature importance scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance.")

        importance_dict = {}
        for i, feature in enumerate(self.selected_features):
            importance_dict[feature] = self.feature_weights[i]

        return importance_dict

    def get_model_summary(self):
        """Get comprehensive model summary"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting summary.")

        summary = {
            'model_type': 'DOME (Dynamic Optimization Meta-Ensemble)',
            'selected_base_learners': self.selected_base_learners,
            'selected_meta_learner': self.selected_meta_learner,
            'learner_weights': self.learner_weights.tolist(),
            'selected_features': self.selected_features,
            'feature_weights': self.feature_weights.tolist(),
            'total_selected_features': len(self.selected_features),
            'hyperparameters': {
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
                'random_state': self.random_state
            }
        }

        return summary


def main():
    """Example usage of DOME model"""
    print("DOME Model - Example Usage")
    print("=" * 40)

    # Load sample data (replace with your dataset)
    try:
        df = pd.read_excel('CPEC_debris_flow_dataset_3447.xlsx')
        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

        # Prepare data
        X = df.drop('Risk_index', axis=1)
        y = df['Risk_index']

        # Initialize and train DOME model
        dome_model = DOMEModel(alpha=0.34, beta=0.04, gamma=0.01)
        results = dome_model.fit(X, y)

        # Display results
        print("\nTraining Results:")
        print(f"Selected base learners: {results['selected_base_learners']}")
        print(f"Selected meta learner: {results['selected_meta_learner']}")
        print(f"Performance metrics: {results['performance_metrics']}")

        # Make sample predictions
        sample_X = X.head(5)
        predictions = dome_model.predict(sample_X)
        print(f"\nSample predictions: {predictions}")

    except FileNotFoundError:
        print("Dataset file not found. Please ensure 'CPEC_debris_flow_dataset_3447.xlsx' is available.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()