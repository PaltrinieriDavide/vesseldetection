import pandas as pd
import numpy as np
import logging
import argparse
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.utils import resample
from sklearn.base import BaseEstimator, RegressorMixin



import matplotlib.pyplot as plt
import seaborn as sns

from custom_knn_metrics import MixedMetric
import categorical_distances_calculators as categorical_distances_calculators

import visualization

#########################################################
#PROVA DI PLOT DELLE DISTRIBUZIONI DEL TARGET PER OGNI CLASSE

def analyze_and_plot_performance(y_test, y_pred, X_test, class_mapping, train_counts, metric_name):
    """
    Analyzes error metrics per class and generates professional visualization plots.
    """
    # Create an inverse mapping for labels
    inv_map = {v: k for k, v in class_mapping.items()}
    
    # Prepare results DataFrame
    results_df = pd.DataFrame({
        'True_Value': y_test.values if hasattr(y_test, 'values') else y_test,
        'Predicted_Value': y_pred,
        'Class_ID': X_test['Class_Name_Encoded'].values
    })
    results_df['Class_Name'] = results_df['Class_ID'].map(inv_map)
    results_df['Abs_Error'] = np.abs(results_df['True_Value'] - results_df['Predicted_Value'])
    results_df['APE'] = (results_df['Abs_Error'] / results_df['True_Value']) * 100

    # Aggregate metrics by class
    class_metrics = results_df.groupby('Class_Name').agg(
        Mean_APE=('APE', 'mean'),
        Median_APE=('APE', 'median'),
        MAE=('Abs_Error', 'mean'),
        Test_Samples=('Abs_Error', 'count')
    ).reset_index()

    # Integrate training representation data
    train_counts_df = train_counts.to_frame().reset_index()
    train_counts_df.columns = ['Class_Name', 'Train_Samples']
    class_metrics = class_metrics.merge(train_counts_df, on='Class_Name')

    print("\n" + "="*50)
    print(f"PERFORMANCE ANALYSIS PER CLASS (Metric: {metric_name})")
    print("="*50)
    print(class_metrics.sort_values(by='Mean_APE', ascending=False).to_string(index=False))

    # Create directories for results
    os.makedirs("results/plots", exist_ok=True)

    # PLOT 1: Error Distribution (Boxplot)
    plt.figure(figsize=(14, 7))
    sns.boxplot(data=results_df, x='Class_Name', y='APE', palette='viridis', hue='Class_Name', legend=False)
    plt.yscale('log')
    plt.title(f'Absolute Percentage Error Distribution by Class (Metric: {metric_name})', fontsize=14, pad=20)
    plt.xlabel('Vessel Class', fontsize=12)
    plt.ylabel('Absolute Percentage Error (%) - Log Scale', fontsize=12)
    plt.xticks(rotation=35, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"results/plots/error_distribution_{metric_name}.png", dpi=300)
    plt.show()

    # PLOT 2: Error vs. Representation (Dual Axis)
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax2 = ax1.twinx()
    
    sns.barplot(data=class_metrics, x='Class_Name', y='Train_Samples', ax=ax1, color='lightgrey', alpha=0.7)
    sns.lineplot(data=class_metrics, x='Class_Name', y='Mean_APE', ax=ax2, marker='o', color='red', linewidth=2.5, label='Mean APE %')
    
    ax1.set_xlabel('Vessel Class', fontsize=12)
    ax1.set_ylabel('Training Set Sample Count (Bars)', color='dimgrey', fontsize=12)
    ax2.set_ylabel('Mean Percentage Error % (Line)', color='red', fontsize=12)
    
    plt.title(f'Error Correlation vs. Training Data Representation ({metric_name})', fontsize=14, pad=20)
    ax1.tick_params(axis='x', rotation=35)
    ax1.grid(axis='y', linestyle=':', alpha=0.5)
    
    fig.tight_layout()
    plt.savefig(f"results/plots/error_vs_representation_{metric_name}.png", dpi=300)
    plt.show()

    return class_metrics

##########################################################








class CustomKNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_neighbors=5, weights='uniform', alpha=0.5,
                 cat_dist_matrix=None, cat_feature_index=None, X_num_for_scaling=None):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.alpha = alpha
        self.cat_dist_matrix = cat_dist_matrix
        self.cat_feature_index = cat_feature_index
        self.X_num_for_scaling = X_num_for_scaling

    def fit(self, X, y):
        custom_metric = MixedMetric(
            alpha=self.alpha,
            cat_dist_matrix=self.cat_dist_matrix,
            cat_feature_index=self.cat_feature_index,
            X_num=self.X_num_for_scaling
        )
        self.model_ = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=custom_metric
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        if not hasattr(self, 'model_'):
            raise RuntimeError("You must fit the model before predicting.")
        return self.model_.predict(X)


def create_balanced_subset(train_df: pd.DataFrame, categorical_feature: str, target_variable: str) -> pd.DataFrame:
    
    class_counts = train_df[categorical_feature].value_counts()
    min_class_size = class_counts.min()
        
    balanced_dfs = []
    
    for class_value in train_df[categorical_feature].unique():
        class_df = train_df[train_df[categorical_feature] == class_value]
        
        df_resampled = resample(class_df, 
                                replace=False,
                                n_samples=min_class_size, 
                                random_state=42)
        balanced_dfs.append(df_resampled)

    balanced_df = pd.concat(balanced_dfs)
    
    logging.info(f"Downsampling completed --> Shape: {balanced_df.shape}")
    logging.info(f"Balanced dataset distribution:\n{balanced_df[categorical_feature].value_counts()}")
    
    return balanced_df


def load_and_prepare_data(file_path: str):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Dataset '{file_path}' loaded. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Error: File '{file_path}' not found.")
        raise
    
    le = LabelEncoder()
    df['Class_Name_Encoded'] = le.fit_transform(df['Class_Name'])
    
    class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    
    logging.info(f"Class mapping: {class_mapping}")
    
    return df, class_mapping

def preprocessing_phase(df, class_mapping, metric):

    numerical_features = ['Length', 'Width']
    categorical_feature = 'Class_Name_Encoded'
    target_variable = 'Tonnage'
    
    df.drop(columns=['area'], errors='ignore', inplace=True)
    
    X = df[numerical_features + [categorical_feature]]
    y = df['Tonnage']
    
    print(f"Dataset shape: {df.head}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_df = X_train.join(y_train)
    
    balanced_train_subset = create_balanced_subset(train_df, categorical_feature, target_variable)
    
    #visualization.plot_target_distributions_by_class(train_df, categorical_feature, target_variable, class_mapping)
    
    logging.info(f"Calculating distance matrix using metric: {metric}")
    calculator_params = {
        "numerical_features": numerical_features,
        "categorical_feature": categorical_feature,
        "target_variable": target_variable,
        "n_classes": len(class_mapping)
    }
    if metric in ['cped']:
        poly_degree = 3
        calculator_params['degree'] = poly_degree
        
    if metric in ['cped_knn']:
        n_neighbors = 5
        calculator_params['n_neighbors'] = n_neighbors
        
    if metric in [ 'cross_integral', 'cross_integral_2d']:
        calculator_params['class_mapping'] = class_mapping

    calculator = categorical_distances_calculators.get_calculator(metric, **calculator_params)
    
    cat_dist_matrix = calculator.calculate(train_df) #balanced_train_subset
    
    visualization.create_heatmap(cat_dist_matrix, class_mapping, metric)
    
    logging.info(f"Calculated distance matrix:\n{np.round(cat_dist_matrix, 2)}")
    print(f"\nCalculated '{metric}' distance matrix:\n", np.round(cat_dist_matrix, 2))

    logging.info(f"Training Samples: {X_train.shape[0]} - Test Samples: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, numerical_features, categorical_feature, cat_dist_matrix

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset path')
    parser.add_argument(
        '--metric', type=str, default='cped',
        choices=['wasserstein', 'overlap', 'cped', 'cped_knn', 'rdd', 'trinitarian', 'cross_integral', 'cross_integral_2d'],
        help='Distance metric to use for the categorical feature.'
    )
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()

    # Setup logging
    os.makedirs(os.path.dirname('results/logs/knn_logFile.log'), exist_ok=True)
    logging.basicConfig(filename='results/logs/knn_logFile.log', filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Starting experiment with arguments: {args}")
    
    try:
        df, class_mapping = load_and_prepare_data(args.dataset)
    except FileNotFoundError:
        return    
    
    X_train, X_test, y_train, y_test, numerical_features, categorical_feature, cat_dist_matrix = preprocessing_phase(df, class_mapping, args.metric)
    
    cat_feature_index = X_train.columns.get_loc(categorical_feature)
        
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', 'passthrough', [categorical_feature]) 
        ],
        remainder='drop'
    )
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    num_feature_indices = slice(0, len(numerical_features))
    X_train_num_scaled = X_train_transformed[:, num_feature_indices]
        
    param_grid = {
        'model__n_neighbors': [5, 7],
        'model__weights': ['distance'],
        'model__alpha': np.linspace(0, 1, 5)
    }

    model_init_params = {
        'cat_dist_matrix': cat_dist_matrix,
        'cat_feature_index': cat_feature_index,
        'X_num_for_scaling': X_train_num_scaled
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', CustomKNNRegressor(**model_init_params))
    ])
    
    print(f"\n--- Optimizing KNN with '{args.metric}' metric ---")
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1, verbose=1)
    #grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring=lambda estimator, X_val, y_val: -pd.DataFrame({"ape": np.abs(np.asarray(y_val) - estimator.predict(X_val)) / np.maximum(np.abs(np.asarray(y_val)), 1e-9), "class": X_val["Class_Name_Encoded"].values}).groupby("class")["ape"].mean().mean(), n_jobs=-1, verbose=1)
    #grid_search = GridSearchCV(estimator=pipeline,param_grid=param_grid,cv=cv,scoring=lambda estimator, X_val, y_val: -np.average(np.abs(np.asarray(y_val) - estimator.predict(X_val)) / np.maximum(np.abs(np.asarray(y_val)), 1e-9),weights=1.0 / X_val["Class_Name_Encoded"].map(X_val["Class_Name_Encoded"].value_counts()).values),n_jobs=-1,verbose=1)
    grid_search.fit(X_train, y_train)
    
    
    best_alpha = grid_search.best_params_['model__alpha']

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    
    # --- AGGIUNGI DA QUI ---
    # Calcolo dei conteggi reali delle classi (usando i nomi originali)
    inv_map = {v: k for k, v in class_mapping.items()}
    train_counts = X_train[categorical_feature].map(inv_map).value_counts()
    
    # Chiamata alla funzione di analisi
    analyze_and_plot_performance(y_test, y_pred, X_test, class_mapping, train_counts, args.metric)
    # --- FINE AGGIUNTA --
    
    
    r2_test = r2_score(y_test, y_pred)
    mae_test = mean_absolute_error(y_test, y_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_pred) * 100

    results_summary = (
        f"\n--- Final Results ---\n"
        f"Best Parameters: {grid_search.best_params_}\n"
        f"Best MAPE (Cross-Validation): {abs(grid_search.best_score_) * 100:.2f}%\n"
        f"--- Performance on Test Set ---\n"
        f"R² Score: {r2_test:.4f}\n"
        f"Mean Absolute Error: {mae_test:.2f} tons\n"
        f"Mean Absolute Percentage Error: {mape_test:.2f}%"
    )
    
    print(results_summary)
    logging.info(results_summary)
        
    os.makedirs("models_result", exist_ok=True)
    
    artifacts_to_save = {
        'model': best_model,
        'class_mapping': class_mapping,
        'numerical_features': numerical_features,
        'categorical_feature_name': categorical_feature,
        'metric_name': args.metric,
        'alpha': best_alpha,
        'cat_dist_matrix': cat_dist_matrix
    }

    file_name = f"knn_model_{args.metric}.joblib"
    file_path = os.path.join("models_result", file_name)

    print(f"\nSaving model and artifacts to '{file_path}'...")
    joblib.dump(artifacts_to_save, file_path)
    

if __name__ == '__main__':
    main()
    
    
    '''
    KNN without categorical distance matrix: ovviamente alpha = 0
    --- Performance on Test Set ---
    R² Score: 0.9967
    Mean Absolute Error: 418.36 tons
    Mean Absolute Percentage Error: 23.61%
    
    
    --- Performance on Test Set ---
    R² Score: 0.0376
    Mean Absolute Error: 5753.93 tons
    Mean Absolute Percentage Error: 58.18%
    '''
