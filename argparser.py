import argparse

def argparser():
    parser = argparse.ArgumentParser(description='Run the Number Line Estimation Experiment.')
    parser.add_argument('--num_trials', type=int, default=20, help='Number of trials in the experiment.')
    parser.add_argument('--seed', default=None, help='Random seed for reproducibility.')
    parser.add_argument('--num_FS', type=int, default=1, help='The number of features stimuli to be optimized.')
    
    ## Arguments related to gpr_instance.py
    parser.add_argument('--kernel_types', type=list, default=[0,6,8], help='The indices (or names) of the kernels to be combined.')
    parser.add_argument('--kernel_arguments', type=list, default=[[1.0], [1.0], [0.01]], help='The values to be fed to create each kernel objects.')
    parser.add_argument('--combine_format', type=str, default="k1*k2+k3", help='A string representing how the kernels should be combined.')
    parser.add_argument('--normalize_y', default=True, help='A binary mask indicating whether to normalize obs_data_Y while fitting.')
    parser.add_argument('--n_restarts_optimizer', default=0, help='The number of restarts of the optimizer to find the optimal kernel parameters.')
    parser.add_argument('--gpr_random_state', default=None, help='A parameter determining random number generation in initializing the centers.')
    
    ## Arguments related to gpr_optimize.py
    parser.add_argument('--return_std', default=True, help='A binary mask indicating whether to return standard deviation of posterior distribution at each query value.')
    parser.add_argument('--return_cov', default=False, help='A binary mask indicating whether to return covaraince matrix of posterior distribution at each query value.')

    parser.add_argument('--save_results_dir', default='results', help='A directory to store the task results in .csv format.')
    parser.add_argument('--save_models_dir', default='models', help='A directory to store the trained Gaussian process regressor model.')
    parser.add_argument('--save_figures_dir', default='figures', help='A directory to store the figures for analyzing the results.')



    
    
    ## return args
    return parser.parse_args()