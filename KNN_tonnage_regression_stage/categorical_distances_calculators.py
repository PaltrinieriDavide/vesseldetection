import logging
import os
from abc import ABC, abstractmethod

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad, dblquad
from scipy.stats import wasserstein_distance

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KernelDensity, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

logger = logging.getLogger(__name__)

class DistributionalDistanceCalculator(ABC):
    def __init__(self, numerical_features, categorical_feature, target_variable, n_classes):
        self.numerical_features = numerical_features
        self.categorical_feature = categorical_feature
        self.target_variable = target_variable
        self.n_classes = n_classes

    @abstractmethod
    def calculate(self, train_df: pd.DataFrame) -> np.ndarray:
        pass

    def _normalize_matrix(self, dist_matrix: np.ndarray) -> np.ndarray:
        max_observed_dist = np.nanmax(dist_matrix)
        if np.isnan(max_observed_dist) or max_observed_dist == 0:
            max_observed_dist = 1.0
        
        dist_matrix[np.isnan(dist_matrix)] = max_observed_dist

        max_val = dist_matrix.max()
        if max_val > 0:
            dist_matrix = dist_matrix / max_val
        
        return dist_matrix
    
    def _normalize_matrix_sqrt(self, dist_matrix: np.ndarray) -> np.ndarray:
        if np.any(dist_matrix < 0):
            valid_distances = dist_matrix[dist_matrix >= 0]
            if len(valid_distances) > 0:
                max_valid_dist = np.max(valid_distances)
                penalty = max_valid_dist * 1.1 if max_valid_dist > 0 else 1.0
                dist_matrix[dist_matrix < 0] = penalty
            else:
                dist_matrix[dist_matrix < 0] = 1.0
                
        transformed_matrix = np.sqrt(dist_matrix)
        
        max_val = np.max(transformed_matrix)
        if max_val > 0:
            normalized_matrix = transformed_matrix / max_val
        else:
            normalized_matrix = transformed_matrix
            
        return normalized_matrix

class WassersteinDistanceCalculator(DistributionalDistanceCalculator):
    """Calculates distance based on the Wasserstein Distance (Earth Mover's Distance)."""
    def calculate(self, train_df: pd.DataFrame) -> np.ndarray:
        logger.info("Starting distance matrix calculation with Wasserstein...")
        dist_matrix = np.zeros((self.n_classes, self.n_classes))
        
        samples_by_class = {
            i: train_df[train_df[self.categorical_feature] == i][self.target_variable].values
            for i in range(self.n_classes)
        }
        
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                samples_i = samples_by_class.get(i, np.array([]))
                samples_j = samples_by_class.get(j, np.array([]))
                
                if len(samples_i) == 0 or len(samples_j) == 0:
                    dist = 1.0
                else:
                    dist = wasserstein_distance(samples_i, samples_j)
                
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        normalized_matrix = self._normalize_matrix(dist_matrix)
        logger.info("Wasserstein distance matrix calculated and normalized.")
        return normalized_matrix


class OverlapDistanceCalculator(DistributionalDistanceCalculator):
    """Calculates distance based on the non-overlapping area of KDEs."""
    def calculate(self, train_df: pd.DataFrame) -> np.ndarray:
        logger.info("Starting distance matrix calculation with Overlap (KDE)...")
        dist_matrix = np.zeros((self.n_classes, self.n_classes))
        
        samples_by_class = {
            i: train_df[train_df[self.categorical_feature] == i][self.target_variable].values
            for i in range(self.n_classes)
        }

        min_val = train_df[self.target_variable].min()
        max_val = train_df[self.target_variable].max()
        grid = np.linspace(min_val, max_val, 1000).reshape(-1, 1)

        kdes = {}
        for i in range(self.n_classes):
            samples = samples_by_class.get(i)
            if samples is not None and len(samples) > 0:
                kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(samples.reshape(-1, 1))
                kdes[i] = kde
        
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                kde_i, kde_j = kdes.get(i), kdes.get(j)
                
                if kde_i is None or kde_j is None:
                    dist = 1.0
                else:
                    pdf_i = np.exp(kde_i.score_samples(grid))
                    pdf_j = np.exp(kde_j.score_samples(grid))
                    intersection_area = np.trapezoid(np.minimum(pdf_i, pdf_j), x=grid.ravel())
                    dist = 1.0 - intersection_area
                
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        normalized_matrix = self._normalize_matrix(dist_matrix)
        logger.info("Overlap distance matrix calculated and normalized.")
        return normalized_matrix


#############################################################################################################
#
#   CROSS PREDICTION ERROR DISTANCE (CPED)
#   compute the mean absolute error between models trained on each class using polynomial regression
#
#############################################################################################################
class CrossPredictionErrorDistance(DistributionalDistanceCalculator):
    """Calculates distance based on cross-prediction error (CPED) using polynomial models."""
    def __init__(self, numerical_features, categorical_feature, target_variable, n_classes, degree=2):
        super().__init__(numerical_features, categorical_feature, target_variable, n_classes)
        self.degree = degree

    def calculate(self, train_df: pd.DataFrame) -> np.ndarray:
        logger.info(f"Starting CPED matrix calculation (Polynomial degree {self.degree})...")
        
        models, class_data = self._train_models_per_class(train_df)
        error_matrix = self._calculate_error_matrix(models, class_data)
        
        
        dist_matrix = np.zeros((self.n_classes, self.n_classes))
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                err_ij, err_ji = error_matrix[i, j], error_matrix[j, i]
                if err_ij < 0 or err_ji < 0:
                    dist = np.nan
                else:
                    dist = (err_ij + err_ji) / 2.0
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        normalized_matrix = self._normalize_matrix_sqrt(dist_matrix)
        
        logger.info("CPED distance matrix calculated and normalized.")
        return normalized_matrix
    
    def _train_models_per_class(self, train_df):
        models, class_data = {}, {}
        min_samples = PolynomialFeatures(degree=self.degree).fit_transform(np.zeros((1, len(self.numerical_features)))).shape[1] + 1
        
        for i in range(self.n_classes):
            df_class = train_df[train_df[self.categorical_feature] == i]
            if len(df_class) >= min_samples:
                X_class, y_class = df_class[self.numerical_features], df_class[self.target_variable]
                pipeline = Pipeline([
                    ('poly', PolynomialFeatures(degree=self.degree, include_bias=False)),
                    ('lin_reg', LinearRegression())
                ])
                pipeline.fit(X_class, y_class)
                models[i] = pipeline
                class_data[i] = (X_class, y_class)
            else:
                logger.warning(f"Class {i} has {len(df_class)} samples, which is not enough to train a model. It will be skipped.")
        return models, class_data

    def _calculate_error_matrix(self, models, class_data):
        error_matrix = np.full((self.n_classes, self.n_classes), -1.0)
        for i in models.keys():
            for j in class_data.keys():
                model_i = models[i]
                X_j, y_j = class_data[j]
                y_pred = model_i.predict(X_j)
                
                #r2 = r2_score(y_j, y_pred)
                #error = 1 - r2
                
                error = mean_absolute_error(y_j, y_pred)
                
                error_matrix[i, j] = error
        return error_matrix
    
    
#############################################################################################################
#
#   CROSS PREDICTION ERROR DISTANCE KNN (CPED-KNN)
#   compute the mean absolute error between models trained on each class using KNN
#
#############################################################################################################
class CrossPredictionErrorDistanceKNN(DistributionalDistanceCalculator):
    def __init__(self, numerical_features, categorical_feature, target_variable, n_classes, n_neighbors=5):
        super().__init__(numerical_features, categorical_feature, target_variable, n_classes)
        self.n_neighbors = n_neighbors

    def calculate(self, train_df: pd.DataFrame) -> np.ndarray:
        logger.info(f"Starting CPED matrix calculation (KNN con k={self.n_neighbors})")
        
        models, class_data = self._train_models_per_class(train_df)
        
        error_matrix = self._calculate_error_matrix(models, class_data)
        
        dist_matrix = np.full((self.n_classes, self.n_classes), np.nan)
        for i in range(self.n_classes):
            for j in range(i, self.n_classes):
                if i == j:
                    dist_matrix[i, j] = 0.0
                    continue

                if i in models and j in models:
                    err_ij = error_matrix[i, j]
                    err_ji = error_matrix[j, i]
                    
                    dist = (err_ij + err_ji) / 2.0
                    dist_matrix[i, j] = dist
                    dist_matrix[j, i] = dist
                else:
                    dist_matrix[i, j] = np.nan
                    dist_matrix[j, i] = np.nan
        
        normalized_matrix = self._normalize_matrix_sqrt(dist_matrix)
        
        return normalized_matrix
    
    def _train_models_per_class(self, train_df):
        models, class_data = {}, {}
        
        for i in range(self.n_classes):
            df_class = train_df[train_df[self.categorical_feature] == i]
            
            if len(df_class) > self.n_neighbors:
                X_class, y_class = df_class[self.numerical_features], df_class[self.target_variable]
                
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('knn', KNeighborsRegressor(n_neighbors=self.n_neighbors))
                ])
                
                pipeline.fit(X_class, y_class)
                models[i] = pipeline
                class_data[i] = (X_class, y_class)
        return models, class_data

    def _calculate_error_matrix(self, models, class_data):
        error_matrix = np.full((self.n_classes, self.n_classes), -1.0)
        
        for i in models.keys():
            for j in class_data.keys():
                model_i = models[i]
                X_j, y_j = class_data[j]
                
                y_pred = model_i.predict(X_j)
                
                error = mean_absolute_error(y_j, y_pred)
                
                error_matrix[i, j] = error
                
        return error_matrix

#############################################################################################################
#
#   RESIDUAL DISTRIBUTION DISTANCE (RDD)
#   compute the Wasserstein distance between the distributions of residuals of models trained on each class
#
#############################################################################################################

class ResidualDistributionDistance(DistributionalDistanceCalculator):
    """
    Calculates distance based on the Wasserstein distance between
    the distributions of model residuals (Residual Distribution Distance - RDD).
    """
    def __init__(self, numerical_features, categorical_feature, target_variable, n_classes, model_params=None):
        super().__init__(numerical_features, categorical_feature, target_variable, n_classes)
        if model_params is None:
            self.model_class = GradientBoostingRegressor
            self.model_params = {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1, 'loss': 'squared_error'}
        else:
            self.model_class = model_params.get('class', GradientBoostingRegressor)
            self.model_params = model_params.get('params', {})

    def calculate(self, train_df: pd.DataFrame) -> np.ndarray:
        logger.info(f"Starting RDD matrix calculation with {self.model_class.__name__}...")
        
        models, class_data = self._train_models_per_class(train_df)
        
        residual_distributions = self._calculate_all_residuals(models, class_data)
        
        dist_matrix = np.zeros((self.n_classes, self.n_classes))
        
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                if (i, i) not in residual_distributions or \
                   (i, j) not in residual_distributions or \
                   (j, j) not in residual_distributions or \
                   (j, i) not in residual_distributions:
                    
                    dist = np.nan
                
                else:
                    R_ii = residual_distributions[(i, i)]
                    R_ij = residual_distributions[(i, j)]
                    R_jj = residual_distributions[(j, j)]
                    R_ji = residual_distributions[(j, i)]
                    
                    d_ij = wasserstein_distance(R_ii, R_ij)
                    d_ji = wasserstein_distance(R_jj, R_ji)
                    
                    dist = (d_ij + d_ji) / 2.0
                    
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        normalized_matrix = self._normalize_matrix(dist_matrix)
        logger.info("RDD distance matrix calculated and normalized.")
        return normalized_matrix
    
    def _train_models_per_class(self, train_df):
        models, class_data = {}, {}
        min_samples = self.model_params.get('min_samples_leaf', 1) * 2 + 10 
        
        for i in range(self.n_classes):
            df_class = train_df[train_df[self.categorical_feature] == i]
            if len(df_class) >= min_samples:
                X_class, y_class = df_class[self.numerical_features], df_class[self.target_variable]
                
                model = self.model_class(**self.model_params)
                model.fit(X_class, y_class)
                
                models[i] = model
                class_data[i] = (X_class, y_class)
            else:
                logger.warning(f"Class {i} has {len(df_class)} samples, not enough to train. It will be skipped.")
        return models, class_data

    def _calculate_all_residuals(self, models, class_data):
        residual_distributions = {}
        for i in models.keys():
            model_i = models[i]
            for j in class_data.keys():
                X_j, y_j = class_data[j]
                
                y_pred = model_i.predict(X_j)
                residuals = y_j - y_pred
                
                residual_distributions[(i, j)] = residuals
                
        return residual_distributions
    
#############################################################################################################
#
#   GAUSSIAN PROCESS REGRESSION (GPR) FITTING
#   ∫from a to b |f1(a) - f2(a)| da
#   f(a) = tonnage
#
#############################################################################################################
class GPRPredictor:
    def __init__(self, model, x_scaler, y_scaler):
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def __call__(self, new_x):
        x_arr = np.atleast_1d(new_x).reshape(-1, 1)
        x_arr_scaled = self.x_scaler.transform(x_arr)
        y_pred_scaled = self.model.predict(x_arr_scaled)
        y_pred_original = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        return y_pred_original.ravel()

def fit_gpr(x, y):
    x_reshaped = x.reshape(-1, 1)
    y_reshaped = y.reshape(-1, 1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaled = x_scaler.fit_transform(x_reshaped)
    y_scaled = y_scaler.fit_transform(y_reshaped)

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
             + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e+1))

    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, alpha=1e-10)
    
    try:
        gpr.fit(x_scaled, y_scaled)
    except Exception as e:
        logger.error(f"GPR fitting failed for data with shape {x.shape}: {e}")
        return None

    predictor_obj = GPRPredictor(model=gpr, x_scaler=x_scaler, y_scaler=y_scaler)
    
    model_info = {
        'name': 'GPR',
        'predictor': predictor_obj,
        'params': gpr.kernel_.get_params()
    }
    
    return model_info
class CrossIntegralDifference(DistributionalDistanceCalculator):
    def __init__(self, numerical_features, categorical_feature, target_variable, n_classes, class_mapping: dict, min_samples=10, output_dir="results/fitted_curves"):
        super().__init__(numerical_features, categorical_feature, target_variable, n_classes)
        self.output_dir = output_dir
        self.min_samples = min_samples
        self.inverse_class_mapping = {v: k for k, v in class_mapping.items()}

    def _train_function_per_class(self, train_df: pd.DataFrame):
        funcs, class_data = {}, {}
        
        for class_id in range(self.n_classes):
            class_name = self.inverse_class_mapping.get(class_id, f"Unknown_ID_{class_id}")
            
            df_class = train_df[train_df[self.categorical_feature] == class_id]
            
            if len(df_class) < self.min_samples:
                logger.warning(f"Class '{class_name}' (ID: {class_id}) has only {len(df_class)} samples (min: {self.min_samples}). Skipping.")
                continue

            width = df_class['Width'].values
            length = df_class['Length'].values
            x = width * length
            y = df_class['Tonnage'].values
            
            sort_indices = np.argsort(x)
            x, y = x[sort_indices], y[sort_indices]
            
            model_info = fit_gpr(x, y)
                        
            if model_info is not None:
                funcs[class_id] = model_info['predictor']
                class_data[class_id] = (x, y)
                self._save_fitted_curve(class_id, model_info, x, y)
                logger.info(f"Successfully fitted GPR for class '{class_name}' (ID: {class_id}).")
            else:
                logger.warning(f"Could not fit a curve for class '{class_name}' (ID: {class_id}).")
        
        return funcs, class_data
    
    def _save_fitted_curve(self, class_id: int, model_info: dict, x_data: np.ndarray, y_data: np.ndarray):
        os.makedirs(self.output_dir, exist_ok=True)
        
        class_name = self.inverse_class_mapping.get(class_id)
        
        safe_class_name = class_name.replace(' ', '_').replace('/', '_')
        
        predictor = model_info['predictor']
        model_obj = predictor.model
        x_scaler = predictor.x_scaler
        y_scaler = predictor.y_scaler

        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, label='Original Data', color='black', s=15, zorder=10)
        
        x_plot = np.linspace(min(x_data), max(x_data), 400).reshape(-1, 1)
        x_plot_scaled = x_scaler.transform(x_plot)
        
        y_plot_scaled, sigma_scaled = model_obj.predict(x_plot_scaled, return_std=True)
        
        y_plot = y_scaler.inverse_transform(y_plot_scaled.reshape(-1, 1)).ravel()
        sigma = (sigma_scaled * y_scaler.scale_).ravel()
        
        lower_bound = y_plot - 1.96 * sigma
        upper_bound = y_plot + 1.96 * sigma

        plt.plot(x_plot.ravel(), y_plot, label='GPR Fit', color='red', linewidth=2)
        plt.fill_between(x_plot.ravel(), lower_bound, upper_bound,
                         alpha=0.2, color='red', label='95% Confidence Interval')
        
        plt.title(f"Gaussian Process Fit for Class: {class_name}")
        plt.xlabel('X (Width * Length)')
        plt.ylabel('Y (Tonnage)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plot_filename = os.path.join(self.output_dir, f"{safe_class_name}_fit.png")
        plt.savefig(plot_filename)
        plt.close()

        model_filename = os.path.join(self.output_dir, f"{safe_class_name}_model.joblib")
        joblib.dump(model_info, model_filename)
        
        logger.info(f"Saved GPR fit for class '{class_name}' to '{self.output_dir}'")
    
    def calculate(self, train_df: pd.DataFrame) -> np.ndarray:
        funcs, class_data = self._train_function_per_class(train_df)
        
        dist_matrix = np.full((self.n_classes, self.n_classes), -1.0)
        np.fill_diagonal(dist_matrix, 0)

        fitted_class_ids = set(funcs.keys())
        fitted_class_names = [self.inverse_class_mapping.get(i, i) for i in sorted(list(fitted_class_ids))]
        logger.info(f"Fitted curves for classes: {fitted_class_names}")
        if len(fitted_class_ids) < self.n_classes:
            unfitted_class_ids = set(range(self.n_classes)) - fitted_class_ids
            unfitted_class_names = [self.inverse_class_mapping.get(i, i) for i in sorted(list(unfitted_class_ids))]
            logger.warning(f"Failed to fit curves for: {unfitted_class_names}")

        logger.info("X-axis domains (Width * Length) for each fitted class:")
        for cid, (x_vals, _) in class_data.items():
            class_name = self.inverse_class_mapping.get(cid, cid)
            logger.info(f"  - Class '{class_name}': min={np.min(x_vals):.2f}, max={np.max(x_vals):.2f}")
        
        valid_integrals = []
        
        fitted_ids_list = sorted(list(fitted_class_ids))
        for i_idx, i in enumerate(fitted_ids_list):
            for j in fitted_ids_list[i_idx + 1:]:
                
                x_i, _ = class_data[i]
                x_j, _ = class_data[j]
                
                a = max(np.min(x_i), np.min(x_j))
                b = min(np.max(x_i), np.max(x_j))
                
                distance = -1.0
                
                if a < b:
                    try:
                        integral_val, _ = quad(lambda x: np.abs(funcs[i](x) - funcs[j](x)), a, b, limit=500)
                        distance = integral_val
                        valid_integrals.append(distance)
                    except Exception as e:
                        class_name_i = self.inverse_class_mapping.get(i, i)
                        class_name_j = self.inverse_class_mapping.get(j, j)
                        logger.warning(f"Integration failed for '{class_name_i}'-'{class_name_j}': {e}. Treating as non-overlapping.")
                
                dist_matrix[i, j] = dist_matrix[j, i] = distance

        if not valid_integrals:
            penalty = 1.0 
            logger.warning("No valid integrals calculated. Using default penalty of 1.0.")
        else:
            max_dist = np.max(valid_integrals)
            penalty = max_dist * 1.1 if max_dist > 0 else 1.0
        
        logger.info(f"Using penalty={penalty:.4f} for non-overlapping/unfitted pairs.")

        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                if dist_matrix[i, j] == -1.0:
                    dist_matrix[i, j] = dist_matrix[j, i] = penalty
        
        if np.any(dist_matrix < 0):
            logger.error("FATAL: Some distances in the matrix were not calculated correctly.")

        if hasattr(self, '_normalize_matrix') and callable(getattr(self, '_normalize_matrix', None)):
            logger.info("Normalizing the distance matrix.")
            dist_matrix = self._normalize_matrix_sqrt(dist_matrix)
        
        return dist_matrix
    
#############################################################################################################
#
#   GAUSSIAN PROCESS REGRESSION (GPR) FITTING
#   ∫from l1, w1 to l2, w2 |f1(l, w) - f2(l, w)| dl dw
#
#############################################################################################################

class GPRPredictor2D:
    def __init__(self, model, x_scaler, y_scaler):
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def __call__(self, new_x):
        x_arr = np.atleast_2d(new_x)
        x_arr_scaled = self.x_scaler.transform(x_arr)
        y_pred_scaled = self.model.predict(x_arr_scaled)
        y_pred_original = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        return y_pred_original.ravel()

def fit_gpr_2d(X, y):
    y_reshaped = y.reshape(-1, 1)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y_reshaped)

    kernel = C(1.0, (1e-3, 1e5)) * Matern(length_scale=[1.0, 1.0], length_scale_bounds=(0.5, 1e3), nu=1.5) \
             + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e+1))
             
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42, alpha=1e-5)
    
    try:
        gpr.fit(X_scaled, y_scaled)
    except Exception as e:
        logger.error(f"GPR fitting failed for data with shape {X.shape}: {e}")
        return None

    predictor_obj = GPRPredictor2D(model=gpr, x_scaler=x_scaler, y_scaler=y_scaler)
    
    model_info = {
        'name': 'GPR_2D',
        'predictor': predictor_obj,
        'params': gpr.kernel_.get_params()
    }
    
    return model_info
class CrossIntegralDifference2D(DistributionalDistanceCalculator):
    def __init__(self, numerical_features, categorical_feature, target_variable, n_classes, class_mapping: dict, min_samples=10, output_dir="results/fitted_curves_2D"):
        super().__init__(numerical_features, categorical_feature, target_variable, n_classes)
        self.output_dir = output_dir
        self.min_samples = min_samples
        self.inverse_class_mapping = {v: k for k, v in class_mapping.items()}
    
    def _spatial_representative_sample(self, df, n_samples, features=['Length', 'Width']):
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances_argmin_min
        """
        Seleziona n_samples che coprono in modo omogeneo lo spazio delle feature
        utilizzando i centroidi del K-Means.
        """
        if len(df) <= n_samples:
            return df

        # 1. Normalizziamo temporaneamente per il clustering (importante per distanze euclidee)
        X = df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 2. Troviamo N cluster nello spazio delle feature
        kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        centroids = kmeans.cluster_centers_

        # 3. Per ogni centroide, troviamo il punto reale più vicino
        # (così usiamo dati reali e non medie sintetiche)
        closest, _ = pairwise_distances_argmin_min(centroids, X_scaled)
        
        # Rimuoviamo duplicati se presenti
        indices = np.unique(closest)
        
        return df.iloc[indices]
                            

    def _train_function_per_class(self, train_df: pd.DataFrame):
        funcs, class_data = {}, {}
        
        # Definiamo un limite ragionevole per il GPR (es. 800 punti)
        MAX_GPR_SAMPLES = 800 

        for class_id in range(self.n_classes):
            class_name = self.inverse_class_mapping.get(class_id, f"Unknown_ID_{class_id}")
            df_class = train_df[train_df[self.categorical_feature] == class_id]
            
            if len(df_class) < self.min_samples:
                continue

            # --- LOGICA DI CAMPIONAMENTO SPAZIALE ---
            if len(df_class) > MAX_GPR_SAMPLES:
                logger.info(f"Applying spatial sampling for class '{class_name}' ({len(df_class)} -> {MAX_GPR_SAMPLES})")
                df_class = self._spatial_representative_sample(df_class, MAX_GPR_SAMPLES)
            # ----------------------------------------

            X_np = df_class[['Length', 'Width']].values 
            y_np = df_class[self.target_variable].values
            
            model_info = fit_gpr_2d(X_np, y_np)
            
            if model_info is not None:
                funcs[class_id] = model_info['predictor']
                class_data[class_id] = (X_np, y_np)
                self._save_fitted_surface(class_id, model_info, X_np, y_np)
                logger.info(f"Successfully fitted GPR for class '{class_name}' (ID: {class_id}).")
                
        return funcs, class_data
    
    def _save_fitted_surface(self, class_id: int, model_info: dict, X_data: np.ndarray, y_data: np.ndarray):
        os.makedirs(self.output_dir, exist_ok=True)
        class_name = self.inverse_class_mapping.get(class_id)
        safe_class_name = class_name.replace(' ', '_').replace('/', '_')
        
        predictor = model_info['predictor']

        l_min, l_max = X_data[:, 0].min(), X_data[:, 0].max()
        w_min, w_max = X_data[:, 1].min(), X_data[:, 1].max()
        
        l_grid = np.linspace(l_min, l_max, 30)
        w_grid = np.linspace(w_min, w_max, 30)
        L, W = np.meshgrid(l_grid, w_grid)
        
        grid_points = np.c_[L.ravel(), W.ravel()]
        T_pred = predictor(grid_points).reshape(L.shape)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(L, W, T_pred, cmap='viridis', alpha=0.7, edgecolor='none')
        ax.scatter(X_data[:, 0], X_data[:, 1], y_data, color='red', s=15, label='Original Data', depthshade=True)

        ax.set_title(f"GPR Surface Fit for Class: {class_name}")
        ax.set_xlabel('Length')
        ax.set_ylabel('Width')
        ax.set_zlabel('Tonnage')
        ax.legend()
        
        plot_filename = os.path.join(self.output_dir, f"{safe_class_name}_fit.png")
        plt.savefig(plot_filename)
        plt.close()

        model_filename = os.path.join(self.output_dir, f"{safe_class_name}_model.joblib")
        joblib.dump(model_info, model_filename)
        
        logger.info(f"Saved GPR fit for class '{class_name}' to '{self.output_dir}'")
    
    def _integrate_monte_carlo(self, func, l_start, l_end, w_start, w_end, num_samples=30000):
        l_samples = np.random.uniform(l_start, l_end, num_samples)
        w_samples = np.random.uniform(w_start, w_end, num_samples)
        
        values = func(w_samples, l_samples)
        
        mean_value = np.mean(values)
        area = (l_end - l_start) * (w_end - w_start)
        return mean_value * area
    
    def calculate(self, train_df: pd.DataFrame) -> np.ndarray:
        funcs, class_data = self._train_function_per_class(train_df)
        
        dist_matrix = np.full((self.n_classes, self.n_classes), -1.0)
        np.fill_diagonal(dist_matrix, 0)

        fitted_class_ids = set(funcs.keys())
        fitted_class_names = [self.inverse_class_mapping.get(i, i) for i in fitted_class_ids]
        logger.info(f"Fitted curves for classes: {fitted_class_names}")
        valid_integrals = []
        
        fitted_ids_list = sorted(list(fitted_class_ids))
        for i_idx, i in enumerate(fitted_ids_list):
            for j in fitted_ids_list[i_idx + 1:]:
                
                X_i, _ = class_data[i]
                X_j, _ = class_data[j]
                
                l_min_i, l_max_i = X_i[:, 0].min(), X_i[:, 0].max()
                w_min_i, w_max_i = X_i[:, 1].min(), X_i[:, 1].max()
            
                l_min_j, l_max_j = X_j[:, 0].min(), X_j[:, 0].max()
                w_min_j, w_max_j = X_j[:, 1].min(), X_j[:, 1].max()
                
                l_start = max(l_min_i, l_min_j)
                l_end   = min(l_max_i, l_max_j)
                w_start = max(w_min_i, w_min_j)
                w_end   = min(w_max_i, w_max_j)
                
                distance = -1.0
                if l_start < l_end and w_start < w_end:
                    def difference_integrand(w, l):
                        points = np.vstack([l, w]).T
                        predictions_i = funcs[i](points)
                        predictions_j = funcs[j](points)
                        return np.abs(predictions_i - predictions_j)
                    try:
                        integral_val = self._integrate_monte_carlo(
                            difference_integrand,
                            l_start, l_end,
                            w_start, w_end,
                            num_samples=60000  # Aumenta per più precisione, diminuisci per più velocità
                        )
                        #distance = integral_val
                        #valid_integrals.append(distance)
                        
                        area = (l_end - l_start) * (w_end - w_start)
                        if area > 1e-9:
                            distance = integral_val / area
                        else:
                            distance = -1.0
                    except Exception as e:
                        class_name_i = self.inverse_class_mapping.get(i, i)
                        class_name_j = self.inverse_class_mapping.get(j, j)
                        logger.warning(f"Integrazione numerica fallita per '{class_name_i}'-'{class_name_j}': {e}")
                dist_matrix[i, j] = dist_matrix[j, i] = distance
                
        if not valid_integrals:
            penalty = 1.0 
            logger.warning("Nessun integrale valido calcolato. Uso una penalità di default di 1.0.")
        else:
            max_dist = np.max(valid_integrals)
            penalty = max_dist * 1.1 if max_dist > 0 else 1.0
    
            dist_matrix[dist_matrix == -1.0] = penalty
        if hasattr(self, '_normalize_matrix') and callable(getattr(self, '_normalize_matrix', None)):
            #dist_matrix = self._normalize_matrix(dist_matrix)
            dist_matrix = self._normalize_matrix_sqrt(dist_matrix)
        return dist_matrix
    
#############################################################################################################
#
#   SOTTO HO PROVATO UNA NUOVA METRICA COSTRUITA SULLA MEDIA DI 3 DISTANZE:
##       - Distanza Funzionale (RDD)
##       - Distanza Rappresentazionale (basata su AUC di un classificatore)
##       - Distanza Distribuzionale (Wasserstein o Overlap)
#
#############################################################################################################
class RepresentationalDistanceCalculator(DistributionalDistanceCalculator):
    """
    Calculates distance based on the distinguishability of input features (X) between classes.
    Uses a binary classifier's AUC to measure separability.
    d = 2 * (AUC - 0.5)
    """
    def calculate(self, train_df: pd.DataFrame) -> np.ndarray:
        logger.info("Starting Representational distance matrix calculation (Classifier-based)...")
        dist_matrix = np.zeros((self.n_classes, self.n_classes))

        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                df_i = train_df[train_df[self.categorical_feature] == i]
                df_j = train_df[train_df[self.categorical_feature] == j]

                if df_i.empty or df_j.empty:
                    dist = 1.0
                else:
                    X_binary = pd.concat([df_i[self.numerical_features], df_j[self.numerical_features]], axis=0)
                    y_binary = np.array([0] * len(df_i) + [1] * len(df_j))
                    
                    clf = lgb.LGBMClassifier(
                        n_estimators=50,
                        num_leaves=10,
                        max_depth=4,
                        learning_rate=0.1,
                        random_state=42,
                        n_jobs=1
                    )
                    
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', clf)
                    ])

                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    auc_scores = []
                    for train_idx, test_idx in cv.split(X_binary, y_binary):
                        X_train_cv, X_test_cv = X_binary.iloc[train_idx], X_binary.iloc[test_idx]
                        y_train_cv, y_test_cv = y_binary[train_idx], y_binary[test_idx]
                        
                        pipeline.fit(X_train_cv, y_train_cv)
                        y_pred_proba = clf.predict_proba(X_test_cv)[:, 1]
                        
                        if len(np.unique(y_test_cv)) > 1:
                            auc_scores.append(roc_auc_score(y_test_cv, y_pred_proba))
                    
                    mean_auc = np.mean(auc_scores) if auc_scores else 0.5
                    
                    dist = 2 * (mean_auc - 0.5)
                
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        logger.info("Representational distance matrix calculated.")
        return dist_matrix

class FunctionalDistanceEnsembleCalculator(ResidualDistributionDistance):
    def calculate(self, train_df: pd.DataFrame) -> np.ndarray:
        logger.info("Starting Functional Ensemble distance matrix calculation...")

        logger.info("  Part 1: Calculating RDD with Polynomial model...")
        
        original_model_class = self.model_class
        original_model_params = self.model_params
        
        poly_calculator = ResidualDistributionDistance(
            self.numerical_features, self.categorical_feature, self.target_variable, self.n_classes
        )
        
        poly_calculator.model_class = Pipeline
        poly_calculator.model_params = {
            'steps': [
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('lin_reg', LinearRegression())
            ]
        }

        models_poly, data_poly = self._train_poly_models(poly_calculator, train_df)
        residuals_poly = poly_calculator._calculate_all_residuals(models_poly, data_poly)
        dist_matrix_poly = self._calculate_rdd_from_residuals(residuals_poly)

        logger.info("  Part 2: Calculating RDD with Gradient Boosting model...")
        
        models_gbt, data_gbt = self._train_models_per_class(train_df)
        residuals_gbt = self._calculate_all_residuals(models_gbt, data_gbt)
        dist_matrix_gbt = self._calculate_rdd_from_residuals(residuals_gbt)

        logger.info("  Part 3: Combining and normalizing distance matrices...")

        norm_dist_poly = self._normalize_matrix(dist_matrix_poly)
        norm_dist_gbt = self._normalize_matrix(dist_matrix_gbt)

        combined_matrix = 0.5 * norm_dist_poly + 0.5 * norm_dist_gbt

        final_matrix = self._normalize_matrix(combined_matrix)
        np.fill_diagonal(final_matrix, 0)
        
        logger.info("Functional Ensemble distance matrix calculated.")
        return final_matrix

    def _train_poly_models(self, calculator_instance, train_df):
        models, class_data = {}, {}
        min_samples_poly = PolynomialFeatures(degree=2).fit_transform(np.zeros((1, len(self.numerical_features)))).shape[1] + 1

        for i in range(self.n_classes):
            df_class = train_df[train_df[self.categorical_feature] == i]
            if len(df_class) >= min_samples_poly:
                X_class, y_class = df_class[self.numerical_features], df_class[self.target_variable]
                model = calculator_instance.model_class(**calculator_instance.model_params)
                model.fit(X_class, y_class)
                models[i] = model
                class_data[i] = (X_class, y_class)
        return models, class_data


    def _calculate_rdd_from_residuals(self, residual_distributions):
        dist_matrix = np.full((self.n_classes, self.n_classes), np.nan)
        
        np.fill_diagonal(dist_matrix, 0)

        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                if (i, i) in residual_distributions and (i, j) in residual_distributions and \
                   (j, j) in residual_distributions and (j, i) in residual_distributions:
                    
                    R_ii, R_ij = residual_distributions[(i, i)], residual_distributions[(i, j)]
                    R_jj, R_ji = residual_distributions[(j, j)], residual_distributions[(j, i)]
                    
                    if len(R_ii) > 0 and len(R_ij) > 0 and len(R_jj) > 0 and len(R_ji) > 0:
                        d_ij = wasserstein_distance(R_ii, R_ij)
                        d_ji = wasserstein_distance(R_jj, R_ji)
                        dist = (d_ij + d_ji) / 2.0
                        dist_matrix[i, j] = dist
                        dist_matrix[j, i] = dist
        return dist_matrix

    def _calculate_rdd_from_residuals(self, residual_distributions):
        """Helper to calculate the RDD matrix from a set of residuals."""
        dist_matrix = np.full((self.n_classes, self.n_classes), np.nan)
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                if (i, i) in residual_distributions and (i, j) in residual_distributions and \
                   (j, j) in residual_distributions and (j, i) in residual_distributions:
                    
                    R_ii, R_ij = residual_distributions[(i, i)], residual_distributions[(i, j)]
                    R_jj, R_ji = residual_distributions[(j, j)], residual_distributions[(j, i)]
                    
                    if len(R_ii) > 0 and len(R_ij) > 0 and len(R_jj) > 0 and len(R_ji) > 0:
                        d_ij = wasserstein_distance(R_ii, R_ij)
                        d_ji = wasserstein_distance(R_jj, R_ji)
                        dist = (d_ij + d_ji) / 2.0
                        dist_matrix[i, j] = dist
                        dist_matrix[j, i] = dist
        return dist_matrix
    
class TrinitarianDistanceCalculator(DistributionalDistanceCalculator):
    """
    Calculates a universal distance by combining three perspectives:
    1. d_dist (Distributional): Wasserstein distance on the target variable.
    2. d_repr (Representational): Classifier-based distance on the input features.
    3. d_func (Functional): Ensemble RDD-based distance on the X->y relationship.
    
    Combines them with specified weights.
    """
    def __init__(self, numerical_features, categorical_feature, target_variable, n_classes,
                 weights={'dist': 0.33, 'repr': 0.33, 'func': 0.34}):
        super().__init__(numerical_features, categorical_feature, target_variable, n_classes)
        self.weights = weights

    def calculate(self, train_df: pd.DataFrame) -> np.ndarray:
        logger.info("Starting Trinitarian distance matrix calculation...")
        
        common_params = {
            'numerical_features': self.numerical_features,
            'categorical_feature': self.categorical_feature,
            'target_variable': self.target_variable,
            'n_classes': self.n_classes
        }

        dist_calc = WassersteinDistanceCalculator(**common_params)
        d_dist = dist_calc.calculate(train_df)

        repr_calc = RepresentationalDistanceCalculator(**common_params)
        d_repr = repr_calc.calculate(train_df)

        func_calc = FunctionalDistanceEnsembleCalculator(**common_params)
        d_func = func_calc.calculate(train_df)

        w = self.weights
        combined_matrix = w['dist'] * d_dist + w['repr'] * d_repr + w['func'] * d_func
        
        final_matrix = self._normalize_matrix(combined_matrix)
        
        return final_matrix

#############################################################################################################
#
#   FACTORY FUNCTION TO GET THE DESIRED CALCULATOR
#
#############################################################################################################

def get_calculator(name: str, **kwargs):
    """Factory function to create the correct calculator instance."""
    calculators = {
        "wasserstein": WassersteinDistanceCalculator,
        "overlap": OverlapDistanceCalculator,
        "cped": CrossPredictionErrorDistance,
        "cped_knn": CrossPredictionErrorDistanceKNN,
        "rdd": ResidualDistributionDistance,
        "representational": RepresentationalDistanceCalculator,
        "functional_ensemble": FunctionalDistanceEnsembleCalculator,
        "trinitarian": TrinitarianDistanceCalculator,
        "cross_integral": CrossIntegralDifference,
        "cross_integral_2d": CrossIntegralDifference2D
    }
    calculator_class = calculators.get(name.lower())
    if not calculator_class:
        raise ValueError(f"Calculator '{name}' not found. Choose from: {list(calculators.keys())}")
    
    return calculator_class(**kwargs)