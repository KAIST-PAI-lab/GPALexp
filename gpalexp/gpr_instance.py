from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, ConstantKernel, DotProduct
from sklearn.gaussian_process.kernels import ExpSineSquared, Exponentiation, Matern
from sklearn.gaussian_process.kernels import PairwiseKernel, RBF, RationalQuadratic, WhiteKernel
from sklearn.gaussian_process.kernels import Product, Sum
import numpy as np
from functools import reduce
from typing import Optional
from typing import Union
import numpy.typing as npt
import re


def GPRInstance(kernel_types:list[str], 
                kernel_arguments:list[dict], 
                combine_format:str, 
                alpha:Union[float, npt.NDArray[np.floating]]=1e-10, 
                normalize_y:bool=True, 
                n_restarts_optimizer:int=50, 
                random_state:Optional[Union[int, np.random.RandomState]]=None):
    if not isinstance(combine_format, str):
        raise TypeError(f"combine_format should be a string, got {type(format).__name__}.")
    if not isinstance(alpha, Union[float, np.ndarray]):
        raise TypeError(f"alpha should be float or numpy array, got {type(alpha).__name__}.")
    if isinstance(alpha, float):
        if alpha<=0:
            raise ValueError(f"alpha should be a positive value, got {alpha}.")
    if isinstance(alpha, np.ndarray):
        if alpha.dtype!=np.floating:
            raise TypeError(f"alpha should have the float dtype.")
        if not np.all(alpha>0):
            raise ValueError(f"alpha should only contain positive elements.")
    if not isinstance(normalize_y, bool):
        raise TypeError(f"normalize_y should be a bool value, got {type(normalize_y).__name__}.")
    if not isinstance(n_restarts_optimizer, int):
        raise TypeError(f"n_restarts_optimizer should be an integer value, got {type(n_restarts_optimizer).__name__}.")
    if n_restarts_optimizer<0:
        raise ValueError(f"n_restarts_optimizer should be non-negative, got {n_restarts_optimizer}.")
    if random_state is not None:
        if not (isinstance(random_state, int) or isinstance(random_state, np.random.RandomState)):
            raise TypeError(f"random_state should be None, an int value, or a RandomState object. Got {type(random_state).__name__}")


    kernel_builder=KernelBuilder(num_kernels=len(kernel_types), 
                           kernel_types_strs_list=kernel_types, 
                           kernel_arguments_dics_list=kernel_arguments, 
                           combine_format_str=combine_format)
    created_kernel=kernel_builder.create_compound_kernel()
    gpr=GaussianProcessRegressor(kernel=created_kernel, alpha=alpha, normalize_y=normalize_y, 
                                 n_restarts_optimizer=n_restarts_optimizer, random_state=random_state)
    return created_kernel, gpr


## Adding and multiplying kernel instances are processed after creating all individual kernel instances.
class KernelBuilder():
    def __init__(self, 
                 num_kernels:int, 
                 kernel_types_strs_list:list[str], 
                 kernel_arguments_dics_list:list[dict], 
                 combine_format_str:str):
        if not isinstance(num_kernels, int):
            raise TypeError(f"num_kernels should be an int value, got the type of {type(num_kernels).__name__}.")
        if num_kernels<=0:
            raise ValueError(f"num_kernels should be a positive integer, got {num_kernels}.")
        if not isinstance(kernel_types_strs_list, list):
            raise TypeError(f"kernel_types_str_list should be a list, got the type of {type(kernel_types_strs_list).__name__}.")
        if not all(isinstance(kernel, str) for kernel in kernel_types_strs_list):
            typ=kernel_types_strs_list[[isinstance(kernel, str) for kernel in kernel_types_strs_list].index(False)]
            raise TypeError(f"kernel_types_str_list should contain string elements, got a {type(typ).__name__ } type element.")
        if len(kernel_types_strs_list)!=num_kernels:
            raise ValueError(f"kernel_types_str_list should be of length {num_kernels}, got {len(kernel_types_strs_list)}.")
        if not isinstance(kernel_arguments_dics_list, list):
            raise TypeError(f"kernel_arguments_dics_list should be a list, got the type of {type(kernel_arguments_dics_list).__name__}.")
        if not all(isinstance(kad, dict) for kad in kernel_arguments_dics_list):
            typ=kernel_arguments_dics_list[[isinstance(kad, dict) for kad in kernel_arguments_dics_list].index(False)]
            raise TypeError(f"kernel_arguments_dics_list should contain dictionary elements, got a {type(typ).__name__} type element.")
        if len(kernel_arguments_dics_list)!=num_kernels:
            raise ValueError(f"kernel_arguments_dics_list should be of length {num_kernels}, got {len(kernel_arguments_dics_list)}.")
        if not isinstance(combine_format_str, str):
            raise TypeError(f"combine_format_str should be a string, got the type of {type(combine_format_str).__name__}.")
        
        split_format_elements=re.split(r'(\+|\*)', combine_format_str)
        if len(split_format_elements) != 2*num_kernels-1:
            raise ValueError(f"combine_format_str should indicate a valid combination of kernel objects.")
        for i in range(0, (len(split_format_elements)+1)//2):
            if int(split_format_elements[2*i][1:])!=i+1:
                raise ValueError(f"The {i}-th operand of combine_format_str should be k{i+1}, not {split_format_elements[2*i]}.")
            
        split_format_numbers=re.findall(r'\d+', combine_format_str)
        kernel_symbol_idxs=list(map(int, split_format_numbers))
        if max(kernel_symbol_idxs)!=num_kernels:
            raise ValueError(f"combine_format_str should include {num_kernels} kernel symbols (from k1 to k{num_kernels}), got a symbol k{max(kernel_symbol_idxs)}.")
        
        split_format_unalloweds=re.findall(r'[^a-zA-z0-9+*]', combine_format_str)
        if split_format_unalloweds:
            raise ValueError(f"combine_format_str should only include alphabet characters, numbers, +, and *, got {split_format_unalloweds}.")
        

        self.num_kernels=num_kernels
        self.kernel_types_strs_list=kernel_types_strs_list
        self.kernel_arguments_dics_list=kernel_arguments_dics_list
        self.split_format_elements=split_format_elements

        self.basicTypes=['ConstantKernel', 'DotProduct', 'ExpSineSquared', 'Exponentiation',
                        'Matern', 'PairwiseKernel', 'RBF', 'RationalQuadratic', 'WhiteKernel']
        self.final_kernel: Optional[Kernel] = None
        self.individual_kernel_objects=[]
    
    def create_compound_kernel(self):
        if self.num_kernels==1:
            final_kernel=self.create_individual_kernel(kernel_type_str=self.kernel_types_strs_list[0], 
                                                       kernel_args_dict=self.kernel_arguments_dics_list[0])    
            self.final_kernel=final_kernel
        else:
            for kernel_index in range(self.num_kernels):
                kernel_object=self.create_individual_kernel(kernel_type_str=self.kernel_types_strs_list[kernel_index], 
                                                            kernel_args_dict=self.kernel_arguments_dics_list[kernel_index])
                self.individual_kernel_objects.append(kernel_object)

            for elem_index, format_elem in enumerate(self.split_format_elements):
                if format_elem=="*":  # * is always at the odd index
                    operand_first=self.individual_kernel_objects[(elem_index-1)//2]
                    operand_second=self.individual_kernel_objects[(elem_index+1)//2]
                    kernel_product=operand_first*operand_second
                    self.individual_kernel_objects.insert((elem_index-1)//2, kernel_product)
                    self.individual_kernel_objects.remove(operand_first)
                    self.individual_kernel_objects.remove(operand_second)
                else:
                    continue
            final_compound_kernel=reduce(lambda x,y:x+y, self.individual_kernel_objects)
            self.final_kernel=final_compound_kernel
        return self.final_kernel
    
    def create_individual_kernel(self, 
                                 kernel_type_str:str, 
                                 kernel_args_dict:dict):
                                 
        def check_params_match(kernel:Kernel, given_args_dict:dict):
            kernel_params=kernel.get_params().keys()
            if not set(given_args_dict.keys()).issubset(set(kernel_params)):
                raise ValueError(f"The set of parameters provided ({set(given_args_dict.keys())}) \nis not the part of the set of required parameters: \n{set(kernel_params)}.")
            return True

        if not isinstance(kernel_type_str, str):
            raise TypeError(f"kernel_type_str should be a string value, got the type of {type(kernel_type_str).__name__}.")
        if not isinstance(kernel_args_dict, dict):
            raise TypeError(f"kernel_args_dict should be a dictionary, got the type of {type(kernel_args_dict).__name__}.")
        
        kernel_type_index=0
        kernel_object=None
        try:
            kernel_type_index=self.basicTypes.index(kernel_type_str)
        except ValueError:
            raise ValueError(f"The given kernel type '{kernel_type_str}' is not a valid kernel type.")
        
        if kernel_type_index==0:
            kernel_object=ConstantKernel()
            if check_params_match(kernel=kernel_object, given_args_dict=kernel_args_dict):
                kernel_object.set_params(**kernel_args_dict)

        elif kernel_type_index==1:
            kernel_object=DotProduct()
            if check_params_match(kernel=kernel_object, given_args_dict=kernel_args_dict):
                kernel_object.set_params(**kernel_args_dict)
        
        elif kernel_type_index==2:
            kernel_object=ExpSineSquared()
            if check_params_match(kernel=kernel_object, given_args_dict=kernel_args_dict):
                kernel_object.set_params(**kernel_args_dict)

        elif kernel_type_index==3:
            if len(kernel_args_dict.keys())!=2:
                raise ValueError(f"The Exponentiation kernel takes 2 arguments, got {len(kernel_args_dict.keys())} arguments.")
            if not isinstance(kernel_args_dict['kernel'], Kernel):
                raise TypeError(f'''The 'kernel' argument of the Exponentiation kernel should be a valid kernel instance, 
                                got the type of {type(kernel_args_dict['kernel']).__name__}.''')
            if not isinstance(kernel_args_dict['exponent'], float):
                raise TypeError(f'''The 'exponent' argument of the Exponentiation kernel should be a float value, 
                                got the type of {type(kernel_args_dict['exponent']).__name__}.''')
            kernel_object=Exponentiation(**kernel_args_dict)

        elif kernel_type_index==4:
            kernel_object=Matern()
            if check_params_match(kernel=kernel_object, given_args_dict=kernel_args_dict):
                kernel_object.set_params(**kernel_args_dict)
        
        elif kernel_type_index==5:
            kernel_object=PairwiseKernel()
            if check_params_match(kernel=kernel_object, given_args_dict=kernel_args_dict):
                kernel_object.set_params(**kernel_args_dict)

        elif kernel_type_index==6:
            kernel_object=RBF()
            if check_params_match(kernel=kernel_object, given_args_dict=kernel_args_dict):
                kernel_object.set_params(**kernel_args_dict)

        elif kernel_type_index==7:
            kernel_object=RationalQuadratic()
            if check_params_match(kernel=kernel_object, given_args_dict=kernel_args_dict):
                kernel_object.set_params(**kernel_args_dict)

        elif kernel_type_index==8:
            kernel_object=WhiteKernel()
            if check_params_match(kernel=kernel_object, given_args_dict=kernel_args_dict):
                kernel_object.set_params(**kernel_args_dict)
        
        else:
            raise ValueError(f"The type '{kernel_type_str}' is not a valid kernel type.")

        return kernel_object
