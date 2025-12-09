import numpy as np
from scipy.special import softmax

from multidimensional_uncertainty.mdu.unc.general_metrics import MahalanobisDistance
from multidimensional_uncertainty.mdu.unc.entropic_ot import EntropicOTOrdering
from multidimensional_uncertainty.mdu.unc.constants import OTTarget, ScalingType, SamplingMethod


def uncertaintyOT(scores_train: np.ndarray, scores_test: np.ndarray, 
                  params: dict = None) -> np.ndarray:
    """    
    scores_train.shape = (n_1, d)
    scores_test.shape = (n_2, d)
    n_i -- number of samples
    d -- uncertainty vector size
    """
    if scores_test.ndim != 2 or scores_train.ndim != 2:
        raise ValueError('Check shape annotation.')
    if scores_train.shape[-1] != scores_test.shape[-1]:
        raise ValueError('Dimension number should be equal for train and test')
    
    if params is None:
        params = {
            'target': OTTarget.BETA,
            'sampling_method': SamplingMethod.GRID,
            'scaling_type': ScalingType.FEATURE_WISE,
            'grid_size': 5,
            'eps': 0.5,
            'target_params': {},
            'n_targets_multiplier': 1,
            'max_iters': 1000,
            'random_state': 42,
            'tol': 1e-6,
        }
    
    multi_dim_uncertainty = EntropicOTOrdering(**params)
    multi_dim_uncertainty.fit(scores_cal=scores_train)
    return multi_dim_uncertainty.predict(scores_test)


def mahalanobis_dist(X_train: np.ndarray, Y_train: np.ndarray, 
                     X_test: np.ndarray) -> np.ndarray:
    """
    Mahalanobis distance function

    X_train: array, shape (n_models, n_samples, n_features)
        Logits or embeddings from each ensemble member.
    Y_train: array, shape (n_samples,)
        True class labels for in-distribution data.
    X_test : array, shape (n_models, n_samples, n_features)
        Test logits or embeddings.
    """
    def check_x_size(x: np.ndarray):
        if x.ndim == 2: 
            x = np.expand_dims(x, axis=0)
        elif x.ndim < 2:
            raise ValueError('Wrong shape')
        return x
    X_train = check_x_size(X_train)
    X_test = check_x_size(X_test)
    
    mahalanobis = MahalanobisDistance()
    mahalanobis.fit(X_train, Y_train)
    return mahalanobis.predict(X_test)


def MSP(logits: np.ndarray) -> np.ndarray:
    """
    Input: logits (n_samples, feature_dim)
    Output: entropy (n_samples)
    """
    probs = softmax(logits, axis=-1)
    entropy = 1 - np.sum(probs * np.log(probs + 1e-8), axis=-1)
    return entropy


if __name__ == '__main__':
    dummy_test = np.random.randn(1000)

    print(uncertaintyOT(dummy_test, dummy_test))