from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import numpy.typing as npt
from functools import wraps
from typing import NamedTuple, Optional, Union, Tuple, Callable, List
from scipy.stats import norm


BoundsType=Union[Tuple[float, float], str]
ScalesType=Union[npt.NDArray[np.float64], float]
npFuncType=Callable[[npt.NDArray, npt.NDArray], npt.NDArray]
MetricsType=Union[str, npFuncType]
kernelTypeDic={0:'ConstantKernel', 
                1:'DotProduct', 
                2:'ExpSineSquared', 
                3:'Exponentiation',
                4:'Matern', 
                5:'PairwiseKernel', 
                6:'RBF', 
                7:'RationalQuadratic', 
                8:'WhiteKernel'}

class prediction(NamedTuple):
    mean: np.ndarray
    std: Optional[np.ndarray] = None
    cov: Optional[np.ndarray] = None

def check_BoundsType_arguments(arg:BoundsType, tuple_elem_type=float, bounds_min=0.0, allowed_strs=["fixed"]):
    if isinstance(arg, tuple):
        if len(arg)!=2:
            raise ValueError(f"arg should be of length 2, got the length of {len(value)}.")
        if not (isinstance(arg[0], tuple_elem_type) and isinstance(arg[1], tuple_elem_type)):
            raise TypeError(f"arg should contain elements of type {tuple_elem_type}.")
        if not (arg[0]>=bounds_min and arg[1]>=bounds_min):
            raise ValueError(f"arg should contain elements not smaller than {bounds_min}.")
        
    elif isinstance(arg, str):
        if arg not in allowed_strs:
            raise ValueError(f"The allowed string values for arg are {allowed_strs}, got {arg}.")
        
    else:
        raise TypeError(f"arg should be either a tuple or a string, got the type of {type(value).__name__}.")
        
def check_ConstantKernel_arguments(arg1:float=1.0, arg2:BoundsType=(1e-5, 1e5)):
    if not isinstance(arg1, float):
        raise TypeError(f"arg1 should be a float value, got the type of {type(arg1).__name__}.")

    check_BoundsType_arguments(arg2)

    return {'constant_value':arg1, 'constant_value_bounds':arg2}


def check_DotProduct_arguments(arg1:float=1.0, arg2:BoundsType=(1e-5, 1e5)):
    if not isinstance(arg1, float):
        raise TypeError(f"arg1 should be a float value, got the type of {type(arg1).__name__}.")
    if not arg1>=0:
        raise ValueError(f"arg1 should be a non-negative float value, got {arg1}.")

    check_BoundsType_arguments(arg2)
    
    return {'sigma_0':arg1, 'sigma_0_bounds':arg2}


def check_ExpSineSquared_arguments(arg1:float=1.0, arg2:float=1.0, arg3:BoundsType=(1e-5, 1e5), arg4:BoundsType=(1e-5, 1e5)):
    if not isinstance(arg1, float):
        raise TypeError(f"arg1 should be a float value, got the type of {type(arg1).__name__}.")
    if not arg1>0:
        raise ValueError(f"arg1 should be a positive value, got {arg1}.")
    
    if not isinstance(arg2, float):
        raise TypeError(f"arg2 should be a float value, got the type of {type(arg1).__name__}.")
    if not arg2>0:
        raise ValueError(f"arg2 should be a positive value, got {arg2}.")
    
    check_BoundsType_arguments(arg3)
    check_BoundsType_arguments(arg4)

    return {'length_scale':arg1, 'periodicity':arg2, 'length_scale_bounds':arg3, 'periodicity_bounds':arg4}


def check_Exponentiation_arguments(arg1:Kernel, arg2:float):
    if not isinstance(arg1, Kernel):
        raise TypeError(f"arg1 should be a valid kernel object, got the type of {type(arg1).__name__}.")
    if not isinstance(arg2, float):
        raise TypeError(f"arg2 should be a float value, got the type of {arg2}.")
    
    return {'kernel':arg1, 'exponent':arg2}


def check_Matern_arguments(arg1:ScalesType=1.0, arg2:BoundsType=(1e-5, 1e5), arg3:float=1.5):
    if not (isinstance(arg1, np.ndarray) or isinstance(arg1, float)):
        raise TypeError(f"arg1 should be either a 1D numpy array or a float value, got the type of {type(arg1).__name__}.")
    elif isinstance(arg1, np.ndarray):
        if arg1.ndim!=1:
            raise ValueError(f"arg1 array should be a 1D numpy array, got {arg1.ndim} dimensions.")
        if arg1.dtype!=np.float64:
            raise TypeError(f"The dtype of arg1 array should be np.float64, got the dtype of {arg1.dtype}.")
        if np.min(arg1) <=0:
            raise ValueError(f"arg1 array should only contain positive elements, got {np.min(arg1)}.")
    else:
        if arg1<=0:
            raise ValueError(f"arg1 should be a positive value, got {arg1}.")

    check_BoundsType_arguments(arg2)

    if not isinstance(arg3, float):
        raise TypeError(f"arg3 should be a float value, got the type of {type(arg3).__name__}.")
    if arg3<=0:
        raise ValueError(f"arg3 should be a positive value, got {arg3}.")
    

    return {'length_scale':arg1, 'length_scale_bounds':arg2, 'nu': arg3}            

'''
def check_PairwiseKernel_argument_3(arg3:npFuncType):
    @wraps(arg3)
    def check_pairwise_arguments(data1:npt.NDArray, data2:Optional[npt.NDArray]):
        if not isinstance(data1, np.ndarray):
            raise TypeError(f"The first argument of arg3 function should be a numpy array, got the type of {type(data1).__name__}.")
        if data1.ndim!=2:
            raise ValueError(f"The first argument of arg3 function should be a 2D numpy array, got {data1.ndim} dimensions.")
        if data2 is not None:
            if not isinstance(data2, np.ndarray):
                raise TypeError(f"The second argument of arg3 function should be a numpy array, got the type of {type(data2).__name__}.")
            if data2.ndim!=2:
                raise ValueError(f"The second argument of arg3 function should be a 2D numpy array, got {data2.ndim} dimensions.")
            if data1.shape[1]!=data2.shape[1]:
                raise ValueError(f"The two arguments of arg3 functon should have equal number of columns, got {data1.shape[1]} and {data2.shape[1]}.")
            result=arg3(data1, data2)
        else:
            result=arg3(data1, data1)
        if not isinstance(result, np.ndarray):
            raise TypeError(f"The return value of arg3 function should be a numpy array.")
        if result.ndim!=2 or result.shape[0]!=result.shape[1]:
            raise ValueError(f"The return value of the arg3 callable should be a 2D square numpy array, got the shape of {result.shape}.")
        if not np.allclose(result, result.T, atol=1e-10):
            raise ValueError(f"The return value of the arg3 callable should be symmetric.")
        try:
            np.linalg.cholesky(result+1e-10*np.eye(result.shape[0]))
        except np.linalg.LinAlgError:
            raise ValueError("The return value of arg3 callable should be a positive semi-definite kernel matrix.")
        return None
    return checker

'''

## arg4: dict[parameter : argument]
## param_type_dic: dict[parameter : type]
## param_default_dic: dict[parameter : defaults]
def check_PairwiseKernel_argument_4(arg4:dict, param_type_dic:dict, param_default_dic:dict):
    ## Empty dictionary - for additive_chi2 kernel.
    if param_type_dic=={}:
        return {}
    
    all_params_list=list(param_type_dic.keys())
    all_params_list.sort()
    all_params_set=set(all_params_list)

    all_input_params_list=list(arg4.keys())
    input_param_indices_set=set([all_params_list.index(param) for param in all_input_params_list])
    
    if not input_param_indices_set.issubset(all_params_set):
        raise ValueError(f'''The input dictionary arg4 should contain keys included in the following list: {all_params_list}.\n 
                             Note that not all available keys need to be present.''')
    
    for input_param in all_input_params_list:
        input_argument=arg4[input_param] # Argument of interest
        expected_type=param_type_dic[input_param] # Expected type of aoi value
        if not isinstance(input_argument, expected_type):
            raise TypeError(f"The arg4[{input_param}] should be of type {expected_type}, got the type of {type(input_argument).__name__}")
    
    for param in all_params_list:
        if param not in all_input_params_list:
            arg4[param]=param_default_dic[param]

    return arg4

def check_PairwiseKernel_arguments(arg1:float=1.0, arg2:BoundsType=(1e-5, 1e5), arg3:MetricsType="linear", arg4:Optional[dict]=None):
    if not isinstance(arg1, float):
        raise TypeError(f"arg1 should be a float value, got the type of {type(arg1).__name__}.")
    else:
        if arg1<=0:
            raise ValueError(f"arg1 should be a positive value, got {arg1}.")
    check_BoundsType_arguments(arg2)

    arg3Cands=["linear", "additive_chi2", "chi2", "poly", "polynomial", "rbf", "laplacian", "sigmoid", "cosine", "precomputed"]
    

    linear_kwargs_type={'dense_output': bool}
    additive_chi2_kwargs_type={}
    chi2_kwargs_type={'gamma': float}
    poly_kwargs_type={'degree': float, 'gamma': Optional[float], 'coef0': float}
    rbf_kwargs_type={'gamma': Optional[float]}
    laplacian_kwargs_type={'gamma': Optional[float]}
    sigmoid_kwargs_type={'gamma': Optional[float], 'coef0': float}
    cosine_kwargs_type={'dense_output': bool}

    linear_kwargs={'dense_output': True}
    additive_chi2_kwargs={}
    chi2_kwargs={'gamma': 1.0}
    poly_kwargs={'degree': 3.0, 'gamma': None, 'coef0': 1.0}
    rbf_kwargs={'gamma': None}
    laplacian_kwargs={'gamma': None}
    sigmoid_kwargs={'gamma': None, 'coef0': 1.0}
    cosine_kwargs={'dense_output': True}
    
    if not (isinstance(arg3, str) or isinstance(arg3, Callable)):
        raise TypeError(f"arg3 should be a Callable or a string, got the type of {type(arg3).__name__}")

    if isinstance(arg3, str):
        if arg3 not in arg3Cands:
            raise ValueError(f"arg3 string value should be one of the following values: {arg3Cands}")
        ## For the case wehre arg3=="precomputed",
        ## sklearn.metrics.pairwise automatically checks
        ## whether the input array is symmetric
        ## and therefore able to be regarded as a kernel matrix.
        ## So whatever we pass does not matter, as long as the types match.
        if arg3=="precomputed":
            return {'gamma':arg1, 'gamma_bounds':arg2, 'metric':arg3, 'pairwise_kernel_args':None}
    ## For the case where arg3 is a Callable object
    ## there are no restrictions on the dimension of input array.
    '''
    if isinstance(arg3, Callable):
        pairwiseArg3(arg3)
    '''

    if arg4 is not None:
        if not isinstance(arg4, dict):
            raise TypeError(f"arg4 should be a dictionary or None, got the type of {type(arg4).__name__}")
        else:
            if arg3=="linear":
                type_dic=linear_kwargs_type
                default_dic=linear_kwargs
            elif arg3=="additive_chi2":
                type_dic=additive_chi2_kwargs_type
                default_dic=additive_chi2_kwargs
            elif arg3=="chi2":
                type_dic=chi2_kwargs_type
                default_dic=chi2_kwargs
            elif arg3=="poly" or arg3=="polynomial":
                type_dic=poly_kwargs_type
                default_dic=poly_kwargs
            elif arg3=="rbf":
                type_dic=rbf_kwargs_type
                default_dic=rbf_kwargs
            elif arg3=="laplacian":
                type_dic=laplacian_kwargs_type
                default_dic=laplacian_kwargs
            elif arg3=="sigmoid":
                type_dic=sigmoid_kwargs_type
                default_dic=sigmoid_kwargs
            elif arg3=="cosine":
                type_dic=cosine_kwargs_type
                default_dic=cosine_kwargs
            else:
                raise ValueError(f"arg3 string value should be one of the following options: {arg3Cands}")
            
            kwargs=check_PairwiseKernel_argument_4(arg4=arg4, 
                                                   param_type_dic=type_dic, 
                                                   param_default_dic=default_dic)

            if arg3=="laplacian":
                if "gamma" in list(kwargs.keys()):
                    gamma=kwargs['gamma']
                    if gamma <=0:
                        raise ValueError(f"arg4['gamma'] should be a positive value, got {gamma}.")
    
    return {'gamma':arg1, 'gamma_bounds':arg2, 'metric':arg3, 'pairwise_kernel_kwargs':kwargs}
    

def check_RBF_arguments(arg1:ScalesType=1.0, arg2:BoundsType=(1e-05, 1e5)):
    if not (isinstance(arg1, np.ndarray) or isinstance(arg1, float)):
        raise TypeError(f"arg1 should be a 1D numpy array or a float value, got the type of{type(arg1).__name__}.")
    if isinstance(arg1, np.ndarray):
        if arg1.ndim!=1:
            raise ValueError(f"arg1 array should be 1D, got {arg1.ndim} dimensions.")
        if arg1.dtype!=np.float64:
            raise TypeError(f"The dtype of arg1 should be np.float64, got {arg1.dtype}.")
        if np.min(arg1)<=0:
            raise ValueError(f"arg1 array should only contain positive elements, got {np.min(arg1)}.")
    else:
        if arg1<=0:
            raise ValueError(f"arg1 should be a positive float value, got {arg1}.")
    check_BoundsType_arguments(arg2)
    
    return {'length_scale':arg1, 'length_scale_bounds':arg2}                


def check_RationalQuadratic_arguments(arg1:float=1.0, arg2:float=1.0, arg3:BoundsType=(1e-5,1e5), arg4:BoundsType=(1e-5,1e5)):
    if not isinstance(arg1, float):
        raise TypeError(f"arg1 should be a float value, got the type of {type(arg1).__name__}.")
    else:
        if arg1<=0:
            raise ValueError(f"arg1 should be a positive value, got {arg1}.")
    
    if not isinstance(arg2, float):
        raise TypeError(f"arg2 should be a float value, got the type of {type(arg2).__name__}.")
    else:
        if arg2<=0:
            raise ValueError(f"arg2 should be a positive value, got {arg2}.")
    
    check_BoundsType_arguments(arg3)
    check_BoundsType_arguments(arg4)

    return {'length_scale':arg1, 'alpha':arg2, 'length_scale_bounds':arg3, 'alpha_bounds':arg4}


def check_WhiteKernel_arguments(arg1:float=1.0, arg2:BoundsType=(1e-5,1e5)):
    if not isinstance(arg1, float):
        raise TypeError(f"arg1 should be a float value, got the type of {type(arg1).__name__}.")
    else:
        if arg1<=0:
            raise ValueError(f"arg1 should be a positive value, got {arg1}.")

    check_BoundsType_arguments(arg2)

    return {'noise_level':arg1, 'noise_level_bounds':arg2}


    
def argsConstructor(kernel_type_list:List, kernel_arguments_list:List[List]):
    if not isinstance(kernel_type_list, list):
        raise TypeError(f"kernel_type_list should be a list, got the type of {type(kernel_type_list).__name__}.")
    if not all(isinstance(kt, int|str) for kt in kernel_type_list):
        typ=kernel_type_list[[isinstance(kt, int|str) for kt in kernel_type_list].index(False)]
        raise TypeError(f"kernel_type_list should contain either integer or tring values, got the type of {type(typ).__name__ }.")
    num_kernels=len(kernel_type_list)
    if not isinstance(kernel_arguments_list, list):
        raise TypeError(f"kernel_arguments_list should be a list, got the type of {type(kernel_arguments_list).__name__}.")
    if not all(isinstance(kal, list) for kal in kernel_arguments_list):
        typ=kernel_arguments_list[[isinstance(kal, list) for kal in kernel_arguments_list].index(False)]
        raise TypeError(f"kernel_arguments_list should contain list elements, got the type of {type(typ).__name__ }.")
    if len(kernel_arguments_list)!=num_kernels:
        raise ValueError(f"kernel_arguments_list should be of length {num_kernels}, got {len([kernel_arguments_list])}.")
    
    kernel_type_index_list=[i for i in range(9)]
    kernel_type_names_list=["ConstantKernel", "DotProduct", "ExpSineSquared", "Exponentiation",
                            "Matern", "PairwiseKernel", "RBF", "RationalQuadratic", "WhiteKernel"]
    kernel_type_options_list=kernel_type_index_list+kernel_type_names_list
    kti_masks=[kt not in kernel_type_options_list for kt in kernel_type_list]
    if any(kti_masks):
        raise ValueError(f"The kernel type should be the following: \n An integer index within (0,9) or the name of the valid kernel types.")
    
    kernel_types=[kernelTypeDic[kt] for kt in kernel_type_list if isinstance(kt, int)]
    kernel_arguments_dict_list=[]

    for kernel_index, kernel_type in enumerate(kernel_types):
        args=kernel_arguments_list[kernel_index]
        if kernel_type=="ConstantKernel":
            args_dict=check_ConstantKernel_arguments(*args)
        elif kernel_type=="DotProduct":
            args_dict=check_DotProduct_arguments(*args)
        elif kernel_type=="ExpSineSquared":
            args_dict=check_ExpSineSquared_arguments(*args)
        elif kernel_type=="Exponentiation":
            args_dict=check_Exponentiation_arguments(*args)
        elif kernel_type=="Matern":
            args_dict=check_Matern_arguments(*args)
        elif kernel_type=="PairwiseKernel":
            args_dict=check_PairwiseKernel_arguments(*args)
        elif kernel_type=="RBF":
            args_dict=check_RBF_arguments(*args)
        elif kernel_type=="RationalQuadratic":
            args_dict=check_RationalQuadratic_arguments(*args)
        elif kernel_type=="WhiteKernel":
            args_dict=check_WhiteKernel_arguments(*args)
        else:
            raise ValueError(f"The following kernel type is not valid: {kernel_type}.")
        kernel_arguments_dict_list.append(args_dict)
    
    return kernel_types, kernel_arguments_dict_list

def sequence_with_interval(start_val: int|float, end_val: int|float, interval:int|float):
    if not isinstance(start_val, float|int):
        raise TypeError(f"start_val should be a float or an integer value, got the type of {type(start_val).__name__}.")
    if not isinstance(end_val, float|int):
        raise TypeError(f"end_val should be a float or an integer value, got the type of {type(end_val).__name__}.")
    if not isinstance(interval, int|float):
        raise TypeError(f"interval should be a float or an integer value, got the type of {type(interval).__name__}.")
    if interval==0:
        raise ValueError(f"The specified interval is zero; this will result in an infinite sequence.")
    if (end_val - start_val)*interval<0:
        raise ValueError(f"Wrong signs; the sequence starts from {start_val} to {end_val}, with an interval of {interval}.")

    if isinstance(start_val, int):
        start_val=float(start_val)
    if isinstance(end_val, int):
        end_val=float(end_val)
    if isinstance(interval, int):
        interval=float(interval)
    return np.linspace(start_val, end_val, np.floor((end_val-start_val)/interval).astype(int)+1)

def grid_with_sequences(*sequences):
    for seq in sequences:
        if not isinstance(seq, np.ndarray):
            raise TypeError(f"Every argument should be a numpy array, got the type of {type(seq).__name__}.")
        if seq.ndim!=2:
            raise ValueError(f"Every argument should be a 2D array, got the dimension of {seq.ndim}.")
        if seq.shape[1]!=1:
            raise ValueError(f"Every argument should have a single column, got {seq.shape[1]} columns.")
            
    seq_2D= (lambda *args: np.concat(*args, axis=0))(*sequences)
    coords_per_axis=list(np.meshgrid(*seq_2D, indexing='ij'))
    coords_grid=np.stack(coords_per_axis, -1)
    return coords_grid




        







