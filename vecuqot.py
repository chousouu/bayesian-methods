import numpy as np

from multidimensional_uncertainty.mdu.unc.entropic_ot import EntropicOTOrdering
from multidimensional_uncertainty.mdu.unc.constants import OTTarget, ScalingType, SamplingMethod


def uncertaintyOT(scores_train: np.ndarray, scores_test: np.ndarray, params: dict = None):
    """    
    scores_train.shape = (n_1, d)
    scores_test.shape = (n_2, d)
    n_i -- number of samples
    d -- number of uncertainty measures
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


if __name__ == '__main__':
    dummy_test = np.random.randn(1000)

    print(uncertaintyOT(dummy_test, dummy_test))