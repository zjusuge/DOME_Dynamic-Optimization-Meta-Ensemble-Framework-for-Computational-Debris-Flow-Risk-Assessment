import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from main import DOMEModel, resolve_target_column, SHAP_AVAILABLE

warnings.filterwarnings("ignore")


def convert_numpy_types(obj):
    """Recursively convert numpy types into JSON-serializable Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj


def load_dataset(data_path):
    """Load CSV/XLS/XLSX dataset."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    return df


def shrink_model_pools_for_testing(model):
    """
    Reduce learner pools for faster test execution while preserving diversity.
    """
    preferred_base = ["RF", "GradientBoosting", "XGBoost", "LightGBM", "SVR"]
    preferred_meta = ["Ridge", "LinearRegression", "GradientBoosting", "XGBoost"]

    original_base = model.base_learners_pool.copy()
    original_meta = model.meta_learners_pool.copy()

    new_base = {
        k: v for k, v in original_base.items()
        if k in preferred_base
    }
    if len(new_base) < 3:
        for k, v in original_base.items():
            if k not in new_base:
                new_base[k] = v
            if len(new_base) >= 3:
                break

    new_meta = {
        k: v for k, v in original_meta.items()
        if k in preferred_meta
    }
    if len(new_meta) < 2:
        for k, v in original_meta.items():
            if k not in new_meta:
                new_meta[k] = v
            if len(new_meta) >= 2:
                break

    model.base_learners_pool = new_base
    model.meta_learners_pool = new_meta
    return model


def test_dome_model(
    data_path="CPEC_debris_flow_dataset_3447.xlsx",
    subset_size=120,
    generate_shap=False,
    verbose=True
):
    """
    Primary validation routine for the susceptibility-oriented DOME implementation.
    """
    print("=" * 70)
    print("DOME Validation Test")
    print("Inventory-Based Regional Debris-Flow Susceptibility Assessment")
    print("=" * 70)

    try:
        start_time = time.time()

        # Load dataset
        df = load_dataset(data_path)
        print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

        target_col = resolve_target_column(df)
        print(f"Detected target column: {target_col}")

        if subset_size is not None:
            subset_size = min(int(subset_size), len(df))
            df = df.sample(n=subset_size, random_state=42).reset_index(drop=True)
            print(f"Using sampled subset: {df.shape[0]} rows")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        print(f"Feature count: {X.shape[1]}")
        print(f"Target min/max: {float(y.min()):.6f} / {float(y.max()):.6f}")

        # Initialize faster test model
        model = DOMEModel(
            alpha=0.34,
            beta=0.04,
            gamma=0.01,
            random_state=42,
            cv_splits=3,
            test_size=0.3,
            n_candidate_base=4,
            n_selected_base=3,
            gcra_population_size=10,
            gcra_max_iterations=10,
            verbose=verbose
        )

        model = shrink_model_pools_for_testing(model)

        print(f"Base learner pool for test: {list(model.base_learners_pool.keys())}")
        print(f"Meta learner pool for test: {list(model.meta_learners_pool.keys())}")

        # Train
        results = model.fit(X, y)
        train_time = time.time() - start_time

        # Predict on a small sample
        pred_count = min(10, len(X))
        X_pred = X.head(pred_count)
        y_true = y.head(pred_count).values
        y_pred = model.predict(X_pred)

        mean_error = float(np.mean(np.abs(y_pred - y_true)))
        max_error = float(np.max(np.abs(y_pred - y_true)))
        relative_error = float(
            mean_error / (np.mean(np.abs(y_true)) + 1e-8) * 100.0
        )

        metrics = results.get("performance_metrics", {})

        # Optional SHAP
        shap_summary_path = None
        shap_status = "not_requested"
        if generate_shap:
            if SHAP_AVAILABLE:
                try:
                    shap_result = model.get_shap_explanations(
                        X.head(min(30, len(X))),
                        background_size=min(15, len(X)),
                        explain_size=min(15, len(X)),
                        nsamples=50
                    )
                    shap_summary = shap_result["summary"]
                    shap_summary_path = "test_shap_summary.csv"
                    shap_summary.to_csv(shap_summary_path, index=False)
                    shap_status = "generated"
                    print(f"SHAP summary saved to: {shap_summary_path}")
                except Exception as e:
                    shap_status = f"failed: {e}"
            else:
                shap_status = "skipped_shap_not_installed"

        checks = {
            "dataset_loaded": True,
            "target_detected": target_col is not None,
            "model_trained": results is not None,
            "model_is_fitted": model.is_fitted,
            "selected_base_learners_nonempty": bool(results.get("selected_base_learners")),
            "selected_meta_learner_nonempty": bool(results.get("selected_meta_learner")),
            "selected_features_nonempty": bool(results.get("selected_features")),
            "predictions_length_valid": len(y_pred) == pred_count,
            "predictions_finite": bool(np.isfinite(y_pred).all()),
            "rmse_finite": np.isfinite(metrics.get("RMSE", np.nan)),
            "mae_finite": np.isfinite(metrics.get("MAE", np.nan)),
            "mape_finite": np.isfinite(metrics.get("MAPE", np.nan)),
            "spearman_valid_range": (
                np.isfinite(metrics.get("Spearman_Correlation", np.nan))
                and -1.0 <= metrics.get("Spearman_Correlation", 0.0) <= 1.0
            ),
        }

        if generate_shap:
            checks["shap_status_ok"] = shap_status in {"generated", "skipped_shap_not_installed"}

        all_passed = all(checks.values())

        print("\nValidation checks:")
        for key, passed in checks.items():
            print(f"  {'✅' if passed else '❌'} {key}")

        print("\nSample predictions:")
        for i in range(min(5, pred_count)):
            print(
                f"  Sample {i + 1}: "
                f"Pred={float(y_pred[i]):.6f}, "
                f"Actual={float(y_true[i]):.6f}, "
                f"AbsError={float(abs(y_pred[i] - y_true[i])):.6f}"
            )

        summary = {
            "test_info": {
                "dataset_path": str(data_path),
                "rows_used": int(len(df)),
                "feature_count": int(X.shape[1]),
                "target_column": target_col,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_scope": "susceptibility_assessment_alignment"
            },
            "model_config": {
                "alpha": model.alpha,
                "beta": model.beta,
                "gamma": model.gamma,
                "cv_splits": model.cv_splits,
                "gcra_population_size": model.gcra_population_size,
                "gcra_max_iterations": model.gcra_max_iterations,
                "base_learners_pool": list(model.base_learners_pool.keys()),
                "meta_learners_pool": list(model.meta_learners_pool.keys())
            },
            "training_results": results,
            "prediction_sample": {
                "actual": y_true.tolist(),
                "predicted": y_pred.tolist(),
                "mean_absolute_error_sample": mean_error,
                "max_absolute_error_sample": max_error,
                "relative_error_percent_sample": relative_error
            },
            "checks": checks,
            "training_time_seconds": float(train_time),
            "shap_status": shap_status,
            "shap_summary_path": shap_summary_path,
            "test_status": "PASSED" if all_passed else "FAILED"
        }

        with open("test_results.json", "w", encoding="utf-8") as f:
            json.dump(convert_numpy_types(summary), f, indent=2, ensure_ascii=False)

        print(f"\nTest results saved to: test_results.json")
        print(f"\n{'🎉 ALL CHECKS PASSED' if all_passed else '⚠️ SOME CHECKS FAILED'}")

        return all_passed

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")

        failure_summary = {
            "test_status": "FAILED",
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open("test_results.json", "w", encoding="utf-8") as f:
            json.dump(failure_summary, f, indent=2, ensure_ascii=False)

        return False


def run_extended_tests(
    data_path="CPEC_debris_flow_dataset_3447.xlsx",
    sample_sizes=(50, 100, 200),
    verbose=False
):
    """
    Extended validation across multiple sample sizes.
    """
    print("\n" + "=" * 70)
    print("Extended Validation")
    print("=" * 70)

    df = load_dataset(data_path)
    target_col = resolve_target_column(df)
    X_all = df.drop(columns=[target_col])
    y_all = df[target_col]

    extended_results = {}

    for size in sample_sizes:
        size = min(int(size), len(df))
        print(f"\nTesting sample size: {size}")

        df_sub = df.sample(n=size, random_state=42).reset_index(drop=True)
        X = df_sub.drop(columns=[target_col])
        y = df_sub[target_col]

        model = DOMEModel(
            alpha=0.34,
            beta=0.04,
            gamma=0.01,
            random_state=42,
            cv_splits=3,
            test_size=0.3,
            n_candidate_base=4,
            n_selected_base=3,
            gcra_population_size=8,
            gcra_max_iterations=8,
            verbose=verbose
        )
        model = shrink_model_pools_for_testing(model)

        start = time.time()
        try:
            results = model.fit(X, y)
            preds = model.predict(X.head(min(10, len(X))))

            extended_results[str(size)] = {
                "status": "PASSED",
                "training_time_seconds": float(time.time() - start),
                "rmse": float(results["performance_metrics"]["RMSE"]),
                "mae": float(results["performance_metrics"]["MAE"]),
                "mape": float(results["performance_metrics"]["MAPE"]),
                "spearman": float(results["performance_metrics"]["Spearman_Correlation"]),
                "selected_feature_count": int(results["selected_feature_count"]),
                "selected_base_learners": results["selected_base_learners"],
                "selected_meta_learner": results["selected_meta_learner"],
                "prediction_sample_count": int(len(preds))
            }
            print(
                f"  ✅ size={size}: "
                f"RMSE={results['performance_metrics']['RMSE']:.6f}, "
                f"time={extended_results[str(size)]['training_time_seconds']:.2f}s"
            )
        except Exception as e:
            extended_results[str(size)] = {
                "status": "FAILED",
                "error": str(e)
            }
            print(f"  ❌ size={size}: {e}")

    with open("extended_test_results.json", "w", encoding="utf-8") as f:
        json.dump(convert_numpy_types(extended_results), f, indent=2, ensure_ascii=False)

    print("\nExtended results saved to: extended_test_results.json")
    return extended_results


def main():
    parser = argparse.ArgumentParser(
        description="DOME validation script for susceptibility-oriented manuscript alignment."
    )
    parser.add_argument(
        "--data",
        default="CPEC_debris_flow_dataset_3447.xlsx",
        help="Path to CSV/XLS/XLSX dataset"
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=120,
        help="Subset size for primary validation"
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Generate optional SHAP summary during test"
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Run extended multi-size validation"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce training logs"
    )
    args = parser.parse_args()

    success = test_dome_model(
        data_path=args.data,
        subset_size=args.subset_size,
        generate_shap=args.shap,
        verbose=not args.quiet
    )

    if success and args.extended:
        run_extended_tests(
            data_path=args.data,
            sample_sizes=(50, 100, 200),
            verbose=False
        )

    return success


if __name__ == "__main__":
    main()
