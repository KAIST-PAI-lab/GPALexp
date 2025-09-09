from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from typing import Callable
import numpy.typing as npt
from typing import Optional
import inspect
from gpal.gpr_fit_predict import gpr_fit, gpr_predict, next_design



def gpal_optimize(gpr:GaussianProcessRegressor, 
                  num_DVs: int, 
                  data_record:npt.NDArray[np.floating],
                  design_candidates:npt.NDArray[np.floating], 
                  design_masking_function:Optional[Callable] = None, 
                  return_stdev: bool = True, 
                  return_covar:bool = False):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if not isinstance(num_DVs, int):
        raise TypeError(f"num_DVs should be an integer value, got the type of {type(num_DVs).__name__}.")
    if num_DVs<1:
        raise ValueError(f"num_DVs should be a positive integer, got {num_DVs}.")
    if not isinstance(data_record, np.ndarray):
        raise TypeError(f"data_record should be a numpy array, got the type of {type(data_record).__name__}.")
    if data_record.dtype!=np.floating:
        raise TypeError(f"data_record should have the float dtype, got the dtype of {data_record.dtype}.")
    if data_record.ndim!=2:
        raise ValueError(f"data_record should be a 2D array, got {data_record.ndim} dimensions.")
    if data_record.shape[1]!=num_DVs+1:
        raise ValueError(f"data_record should have {num_DVs+1} columns; got {data_record.shape[1]} columns.")
    if not isinstance(design_candidates, np.ndarray):
        raise TypeError(f"design_candidates should be a numpy array, got the type of {type(design_candidates).__name__}.")
    if design_candidates.dtype!=np.floating:
        raise TypeError(f"design_candidates should have the float dtype, got the dtype of {design_candidates.dtype}.")
    if design_candidates.ndim!=2:
        raise ValueError(f"design_candidates should be a 2D array, got {design_candidates.ndim} dimensions.")
    if design_candidates.shape[1]!=num_DVs:
        raise ValueError(f"design_candidates should have {num_DVs} columns, got {design_candidates.shape[1]}.")
    if design_masking_function is not None:
        if not callable(design_masking_function):
            raise TypeError(f"design_masking_function should be a callable function, got the type of {type(design_masking_function).__name__}.")
        masking_function_params_num=len(inspect.signature(design_masking_function).parameters)
        if masking_function_params_num != num_DVs:
            raise ValueError(f"design_masking_function should have {num_DVs} parameters, got {masking_function_params_num} parameters.")
    if not isinstance(return_stdev, bool):
        raise TypeError(f"return_stdev should be a bool value, got the type of {type(return_stdev).__name__}.")
    if not isinstance(return_covar, bool):
        raise TypeError(f"return_covar should be a bool value, got the type of {type(return_covar).__name__}.")    


    if design_masking_function is None:
        predict_candidates_X=design_candidates
    else:
        design_candidates_T=design_candidates.T
        design_mask_binary=design_masking_function(*design_candidates_T) 
        predict_candidates_X=design_candidates_T[:,design_mask_binary].T

    lml=gpr_fit(gpr=gpr, num_DVs=num_DVs, data_record=data_record)
    posterior_prediction=gpr_predict(gpr, num_DVs=num_DVs, predict_candidates_X=predict_candidates_X,
                                    return_stdev=return_stdev, return_covar=return_covar)
    result, gp_mean, gp_std=next_design(posterior_prediction=posterior_prediction, num_DVs=num_DVs,
                                                            predict_candidates_X=predict_candidates_X)

    if num_DVs==1:
        result=result[0]
    return result, gp_mean, gp_std, lml



'''
def gpal_optimize2D(gpr:GaussianProcessRegressor, dv1:npt.NDArray[np.float64], dv2:npt.NDArray[np.float64], 
                  est:npt.NDArray[np.float64], dv1Spec:list, dv2Spec:list, 
                  masking:Callable, r_s: bool = True, r_c:bool = False):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if not isinstance(dv1, np.ndarray):
        raise TypeError(f"dv1 should be a numpy array.")
    if dv1.dtype!=np.float64:
        raise TypeError(f"dv1 should have a dtype of np.float64.")
    if dv1.ndim!=2 or dv1.shape[1]!=1:
        raise ValueError(f"dv1 should be a 2D array with a single column, got the shape of {dv1.shape}.")
    N=dv1.shape[0]
    if not isinstance(dv2, np.ndarray):
        raise TypeError(f"dv2 should be a numpy array.")
    if dv2.dtype!=np.float64:
        raise TypeError(f"dv2 should have a dtype of np.float64.")
    if dv2.ndim!=2 or dv2.shape[1]!=1:
        raise ValueError(f"dv2 should be a 2D array with a single column, got the shape fo {dv2.shape}.")
    if dv2.shape[0]!=N:
        raise ValueError(f"dv2 should be of length {N}, got {dv2.shape[0]}.")
    if not isinstance(est, np.ndarray):
        raise TypeError(f"est should be a numpy array.")
    if est.dtype!=np.float64:
        raise TypeError(f"est should have a dtype of np.float64.")
    if est.ndim!=1:
        raise ValueError(f"ests should be a 1D array.")
    if est.shape[0]!=N:
        raise ValueError(f"ests should be of length {N}, got {est.shape[0]}.")
    if not isinstance(dv1Spec, list):
        raise TypeError(f"dv1Spec should be a list.")
    if len(dv1Spec)!=3:
        raise ValueError(f"dv1Spec should have a length of 3, got {len(dv1Spec)}.")
    if not isinstance(dv1Spec[2], int):
        raise TypeError(f"The last element of dv1Spec should be an integer value.")
    if not isinstance(dv2Spec, list):
        raise TypeError(f"dv2Spec should be a list.")
    if len(dv2Spec)!=3:
        raise ValueError(f"dv2Spec should have a length of 3, got {len(dv2Spec)}.")
    if not isinstance(dv2Spec[2], int):
        raise TypeError(f"The last element of dv2Spec should be an integer value.")
    if not callable(masking):
        raise TypeError(f"masking should be a callable function.")
    if not isinstance(r_s, bool):
        raise TypeError(f"r_s should be a bool value, got {type(r_s).__name__}.")
    if not isinstance(r_c, bool):
        raise TypeError(f"r_c should be a bool value, got {type(r_c).__name__}.")    
    
    X_fit=np.column_stack([dv1, dv2])
    y_fit=est

    dv1_cands=np.linspace(dv1Spec[0], dv1Spec[1], dv1Spec[2])
    dv2_cands=np.linspace(dv2Spec[0], dv2Spec[1], dv2Spec[2])

    dv1_grid, dv2_grid=np.meshgrid(dv1_cands, dv2_cands, indexing='ij')
    dv1_grid=dv1_grid.ravel()
    dv2_grid=dv2_grid.ravel()

    mask=masking(dv1_grid, dv2_grid)  # Masking the 2D coordinates with a lambda equation
    dvs_grid=np.column_stack([dv1_grid, dv2_grid])
    X_pred=dvs_grid[mask,:]
    lml=gprFit2D(gpr, X_fit, y_fit)
    pred=gprPredict2D(gpr, X_pred, r_s, r_c)
    nextX, pMean, pStd=nextDesign2D(pred, X_pred)


    return nextX, pMean, pStd, lml




def gpal_optimize1D(gpr:GaussianProcessRegressor, dv1:npt.NDArray[np.float64], 
                  est:npt.NDArray[np.float64], dv1Spec:list,
                  masking:Callable, r_s: bool = True, r_c:bool = False):
    if not isinstance(gpr, GaussianProcessRegressor):
        raise TypeError(f"gpr should be a GaussianProcessRegressor instance, got {type(gpr).__name__}.")
    if not isinstance(dv1, np.ndarray):
        raise TypeError(f"dv1 should be a numpy array.")
    if dv1.dtype!=np.float64:
        raise TypeError(f"dv1 should have a dtype of np.float64.")
    if dv1.ndim!=2 or dv1.shape[1]!=1:
        raise ValueError(f"dv1 should be a 2D numpy array with a single column, got the shape of {dv1.shape}.")
    N=dv1.shape[0]
    if not isinstance(est, np.ndarray):
        raise TypeError(f"est should be a numpy array.")
    if est.dtype!=np.float64:
        raise TypeError(f"est should have a dtype of np.float64.")
    if est.ndim!=1:
        raise ValueError(f"ests should be a 1D array.")
    if est.shape[0]!=N:
        raise ValueError(f"ests should be of length {N}, got {est.shape[0]}.")
    if not isinstance(dv1Spec, list):
        raise TypeError(f"dv1Spec should be a list.")
    if len(dv1Spec)!=3:
        raise ValueError(f"dv1Spec should have a length of 3, got {len(dv1Spec)}.")
    if not isinstance(dv1Spec[2], int):
        raise TypeError(f"The last element of dv1Spec should be an integer value.")
    if not callable(masking):
        raise TypeError(f"masking should be a callable function.")
    if not isinstance(r_s, bool):
        raise TypeError(f"r_s should be a bool value, got {type(r_s).__name__}.")
    if not isinstance(r_c, bool):
        raise TypeError(f"r_c should be a bool value, got {type(r_c).__name__}.")    
    
    X_fit=dv1
    y_fit=est

    dv1_grid=np.linspace(dv1Spec[0], dv1Spec[1], dv1Spec[2])

    mask=masking(dv1_grid)  # Masking the 1D coordinates with a lambda equation
    X_pred=np.expand_dims(dv1_grid[mask], -1)

    lml=gprFit1D(gpr, X_fit, y_fit)
    pred=gprPredict1D(gpr, X_pred, r_s, r_c)
    nextX, pMean, pStd=nextDesign1D(pred, X_pred)


    return nextX, pMean, pStd, lml

'''


