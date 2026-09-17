import argparse
import os
import warnings
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def ape_percent(y_true, y_pred, eps: float = 1e-8) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return np.abs((y_true - y_pred) / denom) * 100.0


def mape_percent(y_true, y_pred) -> float:
    return float(np.mean(ape_percent(y_true, y_pred)))


def median_ape_percent(y_true, y_pred) -> float:
    return float(np.median(ape_percent(y_true, y_pred)))

def safe_r2(y_true, y_pred) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")

def _norm_col(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")


def find_column(df: pd.DataFrame, candidates: List[str], arg_value: Optional[str], role: str) -> str:
    if arg_value is not None:
        if arg_value not in df.columns:
            raise ValueError(f"Column passed for {role} not found: {arg_value}\nAvailable columns: {list(df.columns)}")
        return arg_value

    norm_to_original = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        if _norm_col(cand) in norm_to_original:
            return norm_to_original[_norm_col(cand)]

    raise ValueError(
        f"Could not auto-detect {role} column. Pass it explicitly with --{role}-col.\n"
        f"Available columns: {list(df.columns)}"
    )


class ClassSpecificKNNRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        length_col: str,
        width_col: str,
        class_col: str,
        scale: str = "standard",
        target_transform: str = "none",
        n_neighbors_grid: Tuple[int, ...] = (3, 5, 7, 9, 11),
        weights_grid: Tuple[str, ...] = ("distance",),
        p_grid: Tuple[int, ...] = (2,),
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 1,
        random_state: int = 42,
    ):
        self.length_col = length_col
        self.width_col = width_col
        self.class_col = class_col
        self.scale = scale
        self.target_transform = target_transform
        self.n_neighbors_grid = n_neighbors_grid
        self.weights_grid = weights_grid
        self.p_grid = p_grid
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _make_pipeline(self):
        steps = []
        if self.scale == "standard":
            steps.append(("scaler", StandardScaler()))
        elif self.scale == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        elif self.scale == "none":
            pass
        else:
            raise ValueError("--scale must be one of: none, standard, minmax")

        steps.append(("knn", KNeighborsRegressor()))
        model = Pipeline(steps)

        if self.target_transform == "log1p":
            model = TransformedTargetRegressor(
                regressor=model,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=False,
            )
        elif self.target_transform == "none":
            pass
        else:
            raise ValueError("--target-transform must be one of: none, log1p")

        return model

    def _param_grid_for_n(self, n_samples: int) -> Optional[Dict[str, List]]:
        cv_eff = min(self.cv, n_samples)
        if cv_eff < 2:
            return None

        # In each CV split, the training fold is smaller than n_samples.
        # Keep only k values that are valid in all folds.
        max_test_fold = int(np.ceil(n_samples / cv_eff))
        min_train_fold = n_samples - max_test_fold
        valid_k = [k for k in self.n_neighbors_grid if 1 <= k <= min_train_fold]
        if not valid_k:
            valid_k = [1]

        prefix = "regressor__" if self.target_transform == "log1p" else ""
        return {
            f"{prefix}knn__n_neighbors": valid_k,
            f"{prefix}knn__weights": list(self.weights_grid),
            f"{prefix}knn__p": list(self.p_grid),
        }

    def fit(self, X: pd.DataFrame, y):
        X = X.copy()
        y = np.asarray(y, dtype=float)

        self.classes_ = list(pd.Series(X[self.class_col]).astype(str).unique())
        self.models_: Dict[str, object] = {}
        self.best_params_: Dict[str, Dict] = {}
        self.train_counts_: Dict[str, int] = {}

        print("\n--- Optimizing KNN with 'ks' metric ---")
        print("Ks mode: training one independent KNN for each vessel class")

        for cls in self.classes_:
            mask = X[self.class_col].astype(str).values == cls
            X_cls = X.loc[mask, [self.length_col, self.width_col]]
            y_cls = y[mask]
            n_cls = len(y_cls)
            self.train_counts_[cls] = n_cls

            print(f"\nClass '{cls}' | training samples: {n_cls}")

            if n_cls < 2:
                warnings.warn(f"Class {cls!r} has less than 2 samples. Using constant median predictor.")
                self.models_[cls] = float(np.median(y_cls))
                self.best_params_[cls] = {"constant_median": True}
                continue

            base_model = self._make_pipeline()
            param_grid = self._param_grid_for_n(n_cls)

            if param_grid is None:
                self.models_[cls] = float(np.median(y_cls))
                self.best_params_[cls] = {"constant_median": True}
                continue

            cv_eff = min(self.cv, n_cls)
            cv = KFold(n_splits=cv_eff, shuffle=True, random_state=self.random_state)

            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring="neg_mean_absolute_percentage_error",
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                refit=True,
                error_score="raise",
            )
            search.fit(X_cls, y_cls)

            self.models_[cls] = search.best_estimator_
            self.best_params_[cls] = search.best_params_
            print(f"Best params for class '{cls}': {search.best_params_}")
            print(f"Best CV MAPE for class '{cls}': {-search.best_score_ * 100:.4f}%")

        # Fallback only for unseen classes. It is NOT used for normal Ks evaluation.
        self.global_fallback_ = self._make_pipeline()
        self.global_fallback_.set_params(**self._default_knn_params_for_global(len(y)))
        self.global_fallback_.fit(X[[self.length_col, self.width_col]], y)
        return self

    def _default_knn_params_for_global(self, n_samples: int) -> Dict:
        k = min(5, max(1, n_samples))
        prefix = "regressor__" if self.target_transform == "log1p" else ""
        return {
            f"{prefix}knn__n_neighbors": k,
            f"{prefix}knn__weights": "distance",
            f"{prefix}knn__p": 2,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        preds = np.zeros(len(X), dtype=float)

        for cls in pd.Series(X[self.class_col]).astype(str).unique():
            mask = X[self.class_col].astype(str).values == cls
            X_cls = X.loc[mask, [self.length_col, self.width_col]]

            model = self.models_.get(cls, None)
            if model is None:
                warnings.warn(f"Unseen class at prediction time: {cls!r}. Using global fallback KNN.")
                preds[mask] = self.global_fallback_.predict(X_cls)
            elif isinstance(model, float):
                preds[mask] = model
            else:
                preds[mask] = model.predict(X_cls)

        return np.maximum(preds, 0.0)


def performance_by_class(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_col: str,
    train_counts: Dict[str, int],
) -> pd.DataFrame:
    rows = []
    classes = sorted(pd.Series(df_test[class_col]).astype(str).unique())

    for cls in classes:
        mask = df_test[class_col].astype(str).values == cls
        rows.append({
            "Class_Name": cls,
            "Mean_APE": mape_percent(y_true[mask], y_pred[mask]),
            "Median_APE": median_ape_percent(y_true[mask], y_pred[mask]),
            "MAE": float(mean_absolute_error(y_true[mask], y_pred[mask])),
            "Test_Samples": int(mask.sum()),
            "Train_Samples": int(train_counts.get(cls, 0)),
        })

    return pd.DataFrame(rows)


def print_results(metric_name: str, y_true, y_pred, per_class_df: pd.DataFrame):
    print("\n" + "=" * 50)
    print(f"PERFORMANCE ANALYSIS PER CLASS (Metric: {metric_name})")
    print("=" * 50)
    print(per_class_df.to_string(index=False))

    print("\n" + "=" * 50)
    print(f"OVERALL PERFORMANCE (Metric: {metric_name})")
    print("=" * 50)
    print(f"Mean_APE / MAPE:   {mape_percent(y_true, y_pred):.6f}%")
    print(f"Median_APE:        {median_ape_percent(y_true, y_pred):.6f}%")
    print(f"MAE:               {mean_absolute_error(y_true, y_pred):.6f}")
    print(f"R2:                {safe_r2(y_true, y_pred):.6f}")

def parse_int_tuple(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def parse_str_tuple(s: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate Ks: class-specific KNN GT regressor.")
    parser.add_argument("--dataset", type=str, required=True, help="Input CSV dataset.")
    parser.add_argument("--output-dir", type=str, default="models_result", help="Directory where outputs are saved.")
    parser.add_argument("--model-name", type=str, default="knn_model_ks.joblib", help="Saved model filename.")

    parser.add_argument("--length-col", type=str, default=None)
    parser.add_argument("--width-col", type=str, default=None)
    parser.add_argument("--class-col", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=None)

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=1)

    parser.add_argument("--scale", choices=["none", "standard", "minmax"], default="standard")
    parser.add_argument("--target-transform", choices=["none", "log1p"], default="none")

    parser.add_argument("--n-neighbors-grid", type=str, default="3,5,7,9,11")
    parser.add_argument("--weights-grid", type=str, default="distance")
    parser.add_argument("--p-grid", type=str, default="2")

    args = parser.parse_args()

    df = pd.read_csv(args.dataset)

    length_col = find_column(
        df,
        ["length", "Length", "l", "ship_length", "vessel_length", "Length_m", "length_m", "LOA", "loa"],
        args.length_col,
        "length",
    )
    width_col = find_column(
        df,
        ["width", "Width", "w", "ship_width", "vessel_width", "Width_m", "width_m", "beam", "Beam"],
        args.width_col,
        "width",
    )
    class_col = find_column(
        df,
        ["Class_Name", "class_name", "class", "vessel_class", "ship_class", "label", "category", "lambda"],
        args.class_col,
        "class",
    )
    target_col = find_column(
        df,
        ["GT", "gt", "Gross_Tonnage", "gross_tonnage", "gross tonnage", "tonnage", "Tonnage", "target", "t"],
        args.target_col,
        "target",
    )

    print("Detected columns:")
    print(f"  length_col = {length_col}")
    print(f"  width_col  = {width_col}")
    print(f"  class_col  = {class_col}")
    print(f"  target_col = {target_col}")

    use_cols = [length_col, width_col, class_col, target_col]
    data = df[use_cols].copy()
    data[length_col] = pd.to_numeric(data[length_col], errors="coerce")
    data[width_col] = pd.to_numeric(data[width_col], errors="coerce")
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data[class_col] = data[class_col].astype(str)

    before = len(data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[(data[length_col] > 0) & (data[width_col] > 0) & (data[target_col] > 0)]
    after = len(data)
    if after < before:
        print(f"Dropped {before - after} invalid rows.")

    print("\nClass distribution:")
    print(data[class_col].value_counts().to_string())

    X = data[[length_col, width_col, class_col]].copy()
    y = data[target_col].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = ClassSpecificKNNRegressor(
        length_col=length_col,
        width_col=width_col,
        class_col=class_col,
        scale=args.scale,
        target_transform=args.target_transform,
        n_neighbors_grid=parse_int_tuple(args.n_neighbors_grid),
        weights_grid=parse_str_tuple(args.weights_grid),
        p_grid=parse_int_tuple(args.p_grid),
        cv=args.cv,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        random_state=args.random_state,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    per_class = performance_by_class(X_test, y_test, y_pred, class_col, model.train_counts_)
    print_results("ks", y_test, y_pred, per_class)

    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(args.output_dir, args.model_name)
    joblib.dump({
        "model_type": "Ks_class_specific_knn",
        "model": model,
        "columns": {
            "length_col": length_col,
            "width_col": width_col,
            "class_col": class_col,
            "target_col": target_col,
        },
        "args": vars(args),
        "per_class_results": per_class,
        "overall_results": {
            "MAPE": mape_percent(y_test, y_pred),
            "Median_APE": median_ape_percent(y_test, y_pred),
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "R2": safe_r2(y_test, y_pred),
        },
    }, model_path)

    results_csv = os.path.join(args.output_dir, "knn_ks_per_class_results.csv")
    per_class.to_csv(results_csv, index=False)

    predictions_csv = os.path.join(args.output_dir, "knn_ks_predictions.csv")
    pred_df = X_test.copy()
    pred_df["y_true"] = y_test
    pred_df["y_pred"] = y_pred
    pred_df["APE"] = ape_percent(y_test, y_pred)
    pred_df.to_csv(predictions_csv, index=False)

    print("\nSaved outputs:")
    print(f"  model:       {model_path}")
    print(f"  per-class:   {results_csv}")
    print(f"  predictions: {predictions_csv}")


if __name__ == "__main__":
    main()

