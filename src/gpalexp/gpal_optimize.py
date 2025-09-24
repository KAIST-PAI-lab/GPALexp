from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from typing import Callable
import numpy.typing as npt
from typing import Optional
import inspect
from gpalexp.gpr_fit_predict import gpr_fit, gpr_predict, next_design



def gpal_optimize(gpr:GaussianProcessRegressor, 
                  num_features: int, 
                  data_record:npt.NDArray[np.floating],
                  stimulus_candidates:npt.NDArray[np.floating], 
                  stimulus_masking_function:Optional[Callable] = None, 
                  return_stdev: bool = True, 
                  return_covar:bool = False):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if not isinstance(num_features, int):
        raise TypeError(f"num_features should be an integer value, got the type of {type(num_features).__name__}.")
    if num_features<1:
        raise ValueError(f"num_features should be a positive integer, got {num_features}.")
    if not isinstance(data_record, np.ndarray):
        raise TypeError(f"data_record should be a numpy array, got the type of {type(data_record).__name__}.")
    if data_record.dtype!=np.floating:
        raise TypeError(f"data_record should have the float dtype, got the dtype of {data_record.dtype}.")
    if data_record.ndim!=2:
        raise ValueError(f"data_record should be a 2D array, got {data_record.ndim} dimensions.")
    if data_record.shape[1]!=num_features+1:
        raise ValueError(f"data_record should have {num_features+1} columns; got {data_record.shape[1]} columns.")
    if not isinstance(stimulus_candidates, np.ndarray):
        raise TypeError(f"stimulus_candidates should be a numpy array, got the type of {type(stimulus_candidates).__name__}.")
    if stimulus_candidates.dtype!=np.floating:
        raise TypeError(f"stimulus_candidates should have the float dtype, got the dtype of {stimulus_candidates.dtype}.")
    if stimulus_candidates.ndim>2 or stimulus_candidates.ndim==0:
        raise ValueError(f"stimulus_candidates should be a 2D array, got {stimulus_candidates.ndim} dimensions.")
    if stimulus_candidates.ndim>1 and stimulus_candidates.shape[1]!=num_features:
        raise ValueError(f"stimulus_candidates should have {num_features} columns, got {stimulus_candidates.shape[1]}.")
    if stimulus_masking_function is not None:
        if not callable(stimulus_masking_function):
            raise TypeError(f"stimulus_masking_function should be a callable function, got the type of {type(stimulus_masking_function).__name__}.")
        masking_function_params_num=len(inspect.signature(stimulus_masking_function).parameters)
        if masking_function_params_num != num_features:
            raise ValueError(f"stimulus_masking_function should have {num_features} parameters, got {masking_function_params_num} parameters.")
    if not isinstance(return_stdev, bool):
        raise TypeError(f"return_stdev should be a bool value, got the type of {type(return_stdev).__name__}.")
    if not isinstance(return_covar, bool):
        raise TypeError(f"return_covar should be a bool value, got the type of {type(return_covar).__name__}.")    

    if stimulus_candidates.ndim==1:
        stimulus_candidates=stimulus_candidates.reshape(-1,1)

    if stimulus_masking_function is None:
        predict_candidates_X=stimulus_candidates
    else:
        stimulus_candidates_T=stimulus_candidates.T
        stimulus_mask_binary=stimulus_masking_function(*stimulus_candidates_T) 
        predict_candidates_X=stimulus_candidates_T[:,stimulus_mask_binary].T

    lml=gpr_fit(gpr=gpr, num_features=num_features, data_record=data_record)
    posterior_prediction=gpr_predict(gpr, num_features=num_features, predict_candidates_X=predict_candidates_X,
                                    return_stdev=return_stdev, return_covar=return_covar)
    result, gp_mean, gp_std=next_design(posterior_prediction=posterior_prediction, num_features=num_features,
                                                            predict_candidates_X=predict_candidates_X)

    if num_features==1:
        result=result[0]
    return result, gp_mean, gp_std, lml
