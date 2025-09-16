from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import numpy as np
from functools import reduce
from typing import Optional
import numpy.typing as npt
from gpalexp.utils import prediction

def gpr_fit(gpr:GaussianProcessRegressor, 
            num_features: int, 
            data_record: npt.NDArray[np.floating]):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor object, got the type of {type(gpr).__name__}.")
    if not isinstance(num_features, int):
        raise TypeError(f"num_features should be an integer value, got the type of {type(num_features).__name__}.")
    if num_features<1:
        raise ValueError(f"num_features should be a positive integer, got {num_features}.")
    if not isinstance(data_record, np.ndarray):
        raise TypeError(f"data_record should be a numpy array.")
    if data_record.dtype!=np.floating:
        raise TypeError(f"data_record should have the float dtype, got the dtype of {data_record.dtype}.")
    if data_record.ndim!=2:
        raise ValueError(f"data_record should be a 2D numpy array, got {data_record.ndim} dimensions.")
    if data_record.shape[1]!=num_features+1:
         raise ValueError(f"data_record should be a 2D numpy array with {num_features+1} columns, got {data_record.shape[1]} columns.")
    
    fit_data_X=data_record[:, :-1]
    obs_data_Y=data_record[:, -1]
    gpr.fit(X=fit_data_X, y=obs_data_Y)
    log_marginal_likelihood=gpr.log_marginal_likelihood()
    return log_marginal_likelihood


def gpr_predict(gpr:GaussianProcessRegressor, 
                num_features:int, 
                predict_candidates_X: npt.NDArray[np.floating], 
                return_stdev:bool=False, 
                return_covar:bool=False):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor object, got the type of {type(gpr).__name__}.")
    if not isinstance(num_features, int):
        raise TypeError(f"num_features should be an integer value, got the type of {type(num_features).__name__}.")
    if num_features<1:
        raise ValueError(f"num_features should be a positive integer, got {num_features}.")
    if return_stdev and return_covar:
        raise ValueError(f"At most one of return_stdev and return_covar can be True.")
    if not isinstance(predict_candidates_X, np.ndarray):
        raise TypeError(f"predict_data_X should be a numpy array, got the type of {type(predict_candidates_X).__name__}.")
    if predict_candidates_X.dtype!=np.floating:
        raise TypeError(f"predict_data_X should have the float dtype.")
    if predict_candidates_X.ndim!=2:
        raise ValueError(f"predict_data_X should be a 2D numpy array, got {predict_candidates_X.ndim} dimensions.")
    if predict_candidates_X.shape[1]!=num_features:
         raise ValueError(f"predict_data_X should be a 2D numpy array with {num_features} columns, got {predict_candidates_X.shape[1]} columns.")
    
    posterior_mean=None
    posterior_stdev: Optional[np.ndarray] = None
    posterior_covariance: Optional[np.ndarray] = None
    if return_stdev and not return_covar:
        posterior_mean, posterior_stdev=gpr.predict(X=predict_candidates_X, 
                                                    return_std=return_stdev, 
                                                    return_cov=return_covar)
    if not return_stdev and return_covar:
        posterior_mean, posterior_covariance=gpr.predict(X=predict_candidates_X, 
                                                         return_std=return_stdev, 
                                                         return_cov=return_covar)
    if not return_stdev and not return_covar:
        posterior_mean = gpr.predict(X=predict_candidates_X, 
                                     return_std=return_stdev, 
                                     return_cov=return_covar)
    
    posterior_prediction_per_candidate=prediction(mean=posterior_mean, std=posterior_stdev, cov=posterior_covariance)
    return posterior_prediction_per_candidate


def next_design(posterior_prediction:prediction, 
                num_features: int, 
                predict_candidates_X:npt.NDArray[np.floating]):
    
    if not isinstance(posterior_prediction, prediction):
        raise TypeError(f"posterior_prediction should be the value returned from gpr_predict() function.")
    if not isinstance(num_features, int):
        raise TypeError(f"num_features should be an integer value, got the type of {type(num_features).__name__}.")
    if num_features<1:
        raise ValueError(f"num_features should be a positive integer, got {num_features}.")
    if not isinstance(predict_candidates_X, np.ndarray):
        raise TypeError(f"predict_candidates_X should be a numpy array, got the type of {type(predict_candidates_X).__name__}.")
    if predict_candidates_X.dtype!=np.floating:
        raise TypeError(f"predict_candidates_X should have the float dtype.")
    if predict_candidates_X.ndim!=2:
        raise ValueError(f"predict_candidates_X should be a 2D array, got {predict_candidates_X.ndim} dimensions.")
    if predict_candidates_X.shape[1]!=num_features:
        raise ValueError(f"predict_candidates_X should be a 2D numpy array with {num_features} columns, got {predict_candidates_X.shape[1]} columns.")    
    if posterior_prediction.mean is None:
        raise ValueError(f"The 'mean' field of posterior_predictions should not be None.")
    posterior_mean=posterior_prediction.mean
    if posterior_prediction.std is not None:
        posterior_stdev=posterior_prediction.std
    if posterior_prediction.cov is not None:
        posterior_covar=posterior_prediction.cov

    candidate_idx_with_max_stdev=np.argmax(posterior_stdev)
    next_stimulus_coordinate=predict_candidates_X[candidate_idx_with_max_stdev]
    next_feature_posterior_mean=posterior_mean[candidate_idx_with_max_stdev].item()
    next_feature_posterior_stdev=posterior_stdev[candidate_idx_with_max_stdev].item()
    return next_stimulus_coordinate.tolist(), next_feature_posterior_mean, next_feature_posterior_stdev
