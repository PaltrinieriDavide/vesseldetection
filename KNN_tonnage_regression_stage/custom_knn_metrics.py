import numpy as np
from typing import Callable
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import euclidean, pdist


class MixedMetric:

    def __init__(self, alpha: float, cat_dist_matrix: np.ndarray, cat_feature_index: int, X_num: np.ndarray, numerical_norm_quantile=0.95):
        self.alpha = alpha
        self.cat_dist_matrix = cat_dist_matrix
        self.cat_feature_index = cat_feature_index
        
        #self.max_numerical_dist = np.max(pdist(X_num, metric='euclidean'))
        all_numerical_dists = pdist(X_num, metric='euclidean')
        if len(all_numerical_dists) > 0:
            self.numerical_dist_normalizer = np.quantile(all_numerical_dists, numerical_norm_quantile)
            if self.numerical_dist_normalizer < 1e-9:
                self.numerical_dist_normalizer = 1.0
        else:
            self.numerical_dist_normalizer = 1.0
        
    def __call__(self, u: np.ndarray, v: np.ndarray) -> float:
       
        cat_u = int(u[self.cat_feature_index])
        cat_v = int(v[self.cat_feature_index])
        
        num_u = np.delete(u, self.cat_feature_index)
        num_v = np.delete(v, self.cat_feature_index)
        
        numerical_dist = euclidean(num_u, num_v)
        norm_numerical_dist = np.clip(numerical_dist / self.numerical_dist_normalizer, 0, 1)
                
        categorical_dist = self.cat_dist_matrix[cat_u, cat_v]
        
        #print(f"numerical_dist: {(numerical_dist / self.max_numerical_dist)}, categorical_dist: {categorical_dist}, cat_u: {cat_u}, , cat_v: {cat_v}\n")
        
        total_distance = (1 - self.alpha) * norm_numerical_dist + self.alpha * categorical_dist
        
        return total_distance