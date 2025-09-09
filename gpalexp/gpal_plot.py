#%%
from typing import Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.metrics import mean_squared_error

## This function plots a 2-dimensional uncertainty plot.
## This 'uncertainty plot' visualizes the experimental data as a scatterplot.
## Then it visualizes the posterior mean calculated for each design candidate,
## and the uncertainty range determined by the associated posterior standard deviations.

## Parameter Descriptions
## fig_size: The size of the figure. Must be a tuple holding integer values.
## fit_data_X: The numpy array recording the provided design variables for each given experimental trial. 
## obs_data_Y: The numpy array recording the subject responses for each given experimental trial.
## predict_candidates_X: The numpy array containing the design candidates for GPAL optimization.
## post_mean: The numpy array holding the posterior mean values for each design candidate (in predict_candidate_X).
## post_stdev: The numpy array holding the posterior standard deviation value for each design candidate (in predict_candidate_X).
## x_label: The text label associated with the x-axis (design candidates).
## y_label: The text label associated with the y-axis (subject responses).
## title: The title of the figure.
## sigma_coef: The coefficient determining the uncertainty range. Must be a positive float value.

## NOTE: The uncertainty range is defined as the following:
##       [post_mean - sigma_coef * post_stdev, post_mean + sigma_coef * post_stdev]
## NOTE: The post_mean and post_stdev must be the posterior statistics 
##       obtained by gpr.predict(predict_candidates_X) after gpr.fit(fit_data_X, obs_data_Y).
##       In other words, post_mean and post_stdev must hold posterior statistics
##       for GPAL optimization of the upcoming (next) trial.
##       Then GPAL will select the design candidate with the largest posterior standard deviation,
##       which is equivalent to the largest uncertainty range.

## Return Value Descriptions
## figure: The resulting figure.
## ax: A plot visualized in the resulting figure.

def plot_GPAL_uncertainty(fig_size:Tuple[int, int], 
                          fit_data_X:npt.NDArray, 
                          obs_data_Y:npt.NDArray, 
                          predict_candidates_X:npt.NDArray, 
                          post_mean:npt.NDArray, 
                          post_stdev:npt.NDArray, 
                          x_label:str, 
                          y_label:str, 
                          title:str, 
                          sigma_coef:float=1.0):
    if not isinstance(fig_size, Tuple):
        raise TypeError(f"fig_size should be a tuple, got the type of {type(fig_size).__name__}.")
    if any([not isinstance(fs, int) for fs in fig_size]):
        raise ValueError(f"fig_size should contain integer elements.")
    if len(fig_size)!=2:
        raise ValueError(f"figsize should be of length 2, got {len(fig_size)}.")
    if not isinstance(fit_data_X, np.ndarray):
        raise TypeError(f"fit_data_X should be a numpy array, got the type of {type(fit_data_X).__name__}.")
    if fit_data_X.ndim!=2:
        raise ValueError(f"fit_data_X should be a 2D array, got {fit_data_X.ndim} dimensions.")
    if not isinstance(predict_candidates_X, np.ndarray):
        raise TypeError(f"predict_candidates_X should be a numpy array, got the type of {type(predict_candidates_X).__name__}.")
    if predict_candidates_X.ndim!=1:
        raise ValueError(f"predict_candidates_X should be a 1D array, got {predict_candidates_X.ndim} dimensions.")
    if not isinstance(obs_data_Y, np.ndarray):
        raise TypeError(f"obs_data_Y should be a numpy array, got the type of {type(obs_data_Y).__name__}.")
    if obs_data_Y.ndim!=1:
        raise ValueError(f"obs_data_Y should be a 1D array, got {obs_data_Y.ndim} dimensions.")
    if not isinstance(post_mean, np.ndarray):
        raise TypeError(f"post_mean should be a numpy array, got the type of {type(post_mean).__name__}.")
    if post_mean.ndim!=1:
        raise ValueError(f"post_mean should be a 1D array, got {post_mean.ndim} dimensions.")
    if not isinstance(post_stdev, np.ndarray):
        raise TypeError(f"post_stdev should be a numpy array, got {type(post_stdev).__name__}.")
    if post_stdev.ndim!=1:
        raise ValueError(f"post_stdev should be a 1D array, got {post_stdev.ndim} dimensions.")
    if fit_data_X.shape[0]!=obs_data_Y.shape[0]:
        raise ValueError(f"fit_data_X and obs_data_Y should have equal number of data, got {fit_data_X.shape[0]} and {obs_data_Y.shape[0]}.")
    if predict_candidates_X.shape[0]!=post_mean.shape[0]:
        raise ValueError(f"predict_candidates_X and post_mean should have equal number of data, got {predict_candidates_X.shape[0]} and {post_mean.shape[0]}.")
    if predict_candidates_X.shape[0]!=post_stdev.shape[0]:
        raise ValueError(f"predict_candidates_X and post_stdev should have equal number of data, got {predict_candidates_X.shape[0]} and {post_stdev.shape[0]}.")
    if not isinstance(sigma_coef, float):
        raise TypeError(f"sigma_coef should be a float value, got the type of {type(sigma_coef).__name__}.")
    if sigma_coef<0:
        raise ValueError(f"sigma_coef should be non-negative, got {sigma_coef}.")
    if not isinstance(x_label, str):
        raise TypeError(f"xlabel should be a string value, got the type of {type(x_label).__name__}.")
    if not isinstance(y_label, str):
        raise TypeError(f"ylabel should be a string value, got the type of {type(y_label).__name__}.")
    if not isinstance(title, str):
        raise TypeError(f"title should be a string value, got the type of {type(title).__name__}.")
    
    figure=plt.figure(figsize=fig_size)
    ax=figure.add_subplot(1,1,1)
    
    ## Plots the experiment data as a scatterplot.
    ax.scatter(fit_data_X.ravel(), obs_data_Y, c='black', label='Data')
    ## Plots the posterior mean values associated with every design candidate.
    ax.plot(predict_candidates_X, post_mean, label="Prediction", linewidth=2.5, color='black')
    ## Plots the uncertainty range with semi-transparent color.
    ax.fill_between(predict_candidates_X, post_mean-sigma_coef*post_stdev, 
                     post_mean+sigma_coef*post_stdev, alpha=0.3, label='Uncertainty')
    
    ## Setting the title and labels.
    ax.set_title(title, fontsize=28)
    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(y_label, fontsize=24)
    
    return figure, ax




## This function plots the uncertainty plot at two consecutive trials.
## Since GPAL optimizes the experiment design adaptively, 
## we can visualize how the uncertainty plot changes after a single iteration of GPAL.
## This figure visualizes the uncertainty plot at a certain 'target' trial (on the right subplot) 
## and at the 'previous' trial (on the left subplot).
## The uncertainty is calculated based on the GPAL optimization conducted up to the (target/previous) trial.
## We can directly observe how the uncertainty interval shrinks at the GPAL-selected design point.
## Moreover, a dotted vertical line indicates the design candidate value
## whose associated posterior standard deviation is the largest.
## We can see the design coordinate of the new experiment design (a red dot in the target trial)
## exactly corresponds to the position of the vertical line in the previous trial.

## Parameter descriptions
## fig_size: The size of the figure. Must be a tuple holding integer values.
## font_size: The font size of the text in the figure. Must be positive.
## fit_data_X: The numpy array recording the provided design variables, up to the target trial
## obs_data_Y: The numpy array recording the subject responses, up to the target trial
## predict_candidates_X: The numpy array containing the design candidates for GPAL optimization.
## post_mean_previous: The numpy array holding the posterior mean value for each design candidate, up to the previous trial
## post_stdev_previous: The numpy array holding the posterior standard deviation value for each design candidate, up to the previous trial.
## post_mean_target: The numpy array holding the posterior mean values, up to the target trial.
## post_stdev_target: The numpy array holding the posterior standard deviation values, up to the target trial.
## title: The title of the whole figure.
## title_previous: The title of the left figure (GPAL up to the previous trial)
## title_target: The title of the right figure (GPAL up to the target trial)
## max_stdev_design: The design coordinate with maximum posterior standard deviation, in the previous trial.
## sigma_coef: The coefficient determining the uncertainty range. Must be a positive float value.

## Return Value Specifications
## figure: The whole resulting figure.
## ax1: A plot visualized in the left subplot of the figure.
## ax2: A plot visualized in the right subplot of the figure.

def plot_GPAL_compare_uncertainty(fig_size:Tuple[int, int], 
                                  font_size:int, 
                                  fit_data_X:npt.NDArray, 
                                  obs_data_Y:npt.NDArray,  
                                  predict_candidates_X:npt.NDArray, 
                                  post_mean_previous:npt.NDArray, 
                                  post_stdev_previous:npt.NDArray, 
                                  post_mean_target:npt.NDArray, 
                                  post_stdev_target:npt.NDArray,
                                  xlabel:str,
                                  ylabel:str,
                                  title:str, 
                                  title_previous:str, 
                                  title_target:str, 
                                  max_stdev_design:float, 
                                  sigma_coef:float=1.0):
    
    if not isinstance(fig_size, tuple):
        raise TypeError(f"fig_size should be a tuple, got the type of {type(fig_size).__name__}")
    if any([not isinstance(fs, int) for fs in fig_size]):
        raise ValueError(f"fig_size should contain integer elements.")
    if len(fig_size)!=2:
        raise ValueError(f"fig_size should be of length 2, got {len(fig_size)}.")
    if not isinstance(font_size, int):
        raise TypeError(f"font_size should be an integer value, got the type of {type(font_size).__name__}.")
    if font_size<=0:
        raise ValueError(f"font_size should be a positive value, got {font_size}.")
    if not isinstance(fit_data_X, np.ndarray):
        raise TypeError(f"fit_data_X should be a numpy array, got the type of {type(fit_data_X).__name__}.")
    if fit_data_X.ndim!=2:
        raise ValueError(f"fit_data_X should be a 2D array, got {fit_data_X.ndim} dimensions.")
    if not isinstance(predict_candidates_X, np.ndarray):
        raise TypeError(f"predict_candidates_X should be a numpy array, got the type of {type(predict_candidates_X).__name__}.")
    if predict_candidates_X.ndim!=1:
        raise ValueError(f"predict_candidates_X should be a 1D array, got {predict_candidates_X.ndim} dimensions.")
    if not isinstance(obs_data_Y, np.ndarray):
        raise TypeError(f"obs_data_Y should be a numpy array, got the type of {type(obs_data_Y).__name__}.")
    if obs_data_Y.ndim!=1:
        raise ValueError(f"obs_data_Y should be a 1D array, got {obs_data_Y.ndim} dimensions.")
    if not isinstance(post_mean_previous, np.ndarray):
        raise TypeError(f"post_mean_previous should be a numpy array, got the type of {type(post_mean_previous).__name__}.")
    if post_mean_previous.ndim!=1:
        raise ValueError(f"post_mean_previous should be a 1D array, got {post_mean_previous.ndim} dimensions.")
    if not isinstance(post_stdev_previous, np.ndarray):
        raise TypeError(f"post_stdev_previous should be a numpy array, got the type of {type(post_stdev_previous).__name__}.")
    if post_stdev_previous.ndim!=1:
        raise ValueError(f"post_stdev_previous should be a 1D array, got {post_stdev_previous.ndim} dimensions.")
    if not isinstance(post_mean_target, np.ndarray):
        raise TypeError(f"post_mean_target should be a numpy array, got the type of {type(post_mean_target).__name__}.")
    if post_mean_target.ndim!=1:
        raise ValueError(f"post_mean_target should be a 1D array, got {post_mean_target.ndim} dimensions.")
    if not isinstance(post_stdev_target, np.ndarray):
        raise TypeError(f"post_stdev_target should be a numpy array, got the type of {type(post_stdev_target).__name__}.")
    if post_stdev_target.ndim!=1:
        raise ValueError(f"post_stdev_target should be a 1D array, got {post_stdev_target.ndim} dimensions.")
    if fit_data_X.shape[0]!=obs_data_Y.shape[0]:
        raise ValueError(f"fit_data_X and obs_data_Y should have equal number of data, got {fit_data_X.shape[0]} and {obs_data_Y.shape[0]}.")
    if predict_candidates_X.shape[0]!=post_mean_previous.shape[0]:
        raise ValueError(f"predict_candidates_X and post_mean_previous should have equal number of data, got {predict_candidates_X.shape[0]} and {post_mean.shape[0]}.")
    if predict_candidates_X.shape[0]!=post_stdev_previous.shape[0]:
        raise ValueError(f"predict_candidates_X and post_stdev_previous should have equal number of data, got {predict_candidates_X.shape[0]} and {post_stdev.shape[0]}.")
    if predict_candidates_X.shape[0]!=post_mean_target.shape[0]:
        raise ValueError(f"predict_candidates_X and post_mean_target should have equal number of data, got {predict_candidates_X.shape[0]} and {post_mean_after.shape[0]}.")
    if predict_candidates_X.shape[0]!=post_stdev_target.shape[0]:
        raise ValueError(f"predict_candidates_X and post_stdev_target should have equal number of data, got {predict_candidates_X.shape[0]} and {post_stdev_after.shape[0]}.")
    if not isinstance(max_stdev_design, float):
        raise TypeError(f"max_stdev_design should be a float value, got the type of {type(max_stdev_design).__name__}.")
    if not isinstance(sigma_coef, float):
        raise TypeError(f"sigma_coef should be a float value, got the type of {type(sigma_coef).__name__}")
    if sigma_coef<0:
        raise ValueError(f"sigma_coef should be non-negative, got {sigma_coef}.")
    if not isinstance(xlabel, str):
        raise TypeError(f"xlabel should be a string value, got the type of {type(xlabel).__name__}.")
    if not isinstance(ylabel, str):
        raise TypeError(f"ylabel should be a string value, got the type of {type(ylabel).__name__}.")
    if not isinstance(title, str):
        raise TypeError(f"title should be a string value, got the type of {type(title).__name__}.")
    if not isinstance(title_previous, str):
        raise TypeError(f"title_previous should be a string value, got the type of {type(title_previoius).__name__}.")
    if not isinstance(title_target, str):
        raise TypeError(f"title_target should be a string value, got the type of {type(title_target).__name__}.")
    
    ## Creating a figure with two subplots.
    figure, (ax1, ax2)=plt.subplots(2,1, figsize=fig_size)
    
    ## Left subplot
    ## Plotting the experiment data up to the previous trial.
    ax1.scatter(fit_data_X[:-1], obs_data_Y[:-1], c='black', label='Data')
    ## Plotting the posterior mean, calculated with so-far obtained experiment data.
    ax1.plot(predict_candidates_X, post_mean_previous, label="Prediction", linewidth=2.5, color='black')
    ## Plotting the uncertainty range for each design candidate.
    ax1.fill_between(predict_candidates_X.ravel(), post_mean_previous-sigma_coef*post_stdev_previous,
                     post_mean_previous+sigma_coef*post_stdev_previous, alpha=0.3, label='Uncertainty')
    ## Plotting a dotted vertical line, at the design candidate associated with maximum posterior standard deviation.
    ax1.axvline(x=fit_data_X[-1], color='green', linestyle='--', linewidth=3)
    ax1.set_xlabel(xlabel, fontsize=16)
    ax1.set_ylabel(ylabel, fontsize=16)
    ## Setting the title for the left subplot.
    ax1.set_title(title_previous, fontsize=font_size)
    

    ## Right subplot
    ## Plotting the experiment data up to the target trial.
    ax2.scatter(fit_data_X, obs_data_Y, c='black')
    ## Plotting the posterior mean, calculated with so-far obtained experiment data.
    ax2.plot(predict_candidates_X, post_mean_target, linewidth=2.5, color='black')
    ## Plotting the uncertainty range for each design candidate.
    ax2.fill_between(predict_candidates_X.ravel(), post_mean_target-sigma_coef*post_stdev_target, 
                     post_mean_target+sigma_coef*post_stdev_target, alpha=0.3)
    ## Plotting an experiment data for the newly selected experimental design.
    ax2.scatter(fit_data_X[-1], obs_data_Y[-1], c='red')
    ## Plotting a dotted vertical line, at the design candidate associated with maximums posterior standard deviation.
    ax2.axvline(x=max_stdev_design, color='green', linestyle='--', linewidth=3)
    ax2.set_xlabel(xlabel, fontsize=16)
    ax2.set_ylabel(ylabel, fontsize=16)
    ## Setting the title for the right subplot.
    ax2.set_title(title_target, fontsize=font_size)

    ## Setting the title for the whole figure.
    figure.suptitle(title, fontsize=font_size)
    plt.tight_layout()
    
    return figure, ax1, ax2



## This function plots the design selection frequencies of 1D GPAL as a 2D histogram.
## GPAL selects the optimal design value among design candidates in an adaptive manner.
## We can observe the distribution of those 'selected' design values (of the design variable),
## therefore examine features of the function of our interest.
## As "1D" in the function name implies, this functions plots a 2D histogram,
## where the x-axis denotes the design values and the y-axis indicates the frequencies.


## Parameter Descriptions.
## fig_size: The size of the figure. Must be a tuple holding integer values.
## num_data: The number of selected design values (i.e. the optimal design candidates).  
## design_var: A numpy array holding all selected design values.
## bins: The number of equal-length bins dividing the range of selected design values.
## ranges: A tuple indicating the total range of the selected design values.
## x_label: The text label associated with the x-axis (design candidates).
## y_label: The text label associated with the y-axis (subject responses).
## title: The title of the figure.
## mode: A string determining the mode of the histogram. Must be either 'sum' or 'average'.
##       Setting it to 'average' will normalize the histogram values.

## NOTE: The ranges parameter will be automatically set to the folloiwng, if not specified explicitly.
##       ranges = (np.min(design_var), np.max(design_var))

## Return Value Specifications
## figure: The whole resulting figure.
## ax: A plot visualized in the figure.

def plot_frequency_histogram_1D(fig_size:Tuple[int, int], 
                                num_data:int, 
                                design_var:npt.NDArray, 
                                bins:int, 
                                ranges:Optional[Tuple[float, float]], 
                                x_label:str, 
                                y_label:str, 
                                title:str, 
                                mode:str="sum"):
    if not isinstance(fig_size, tuple):
        raise TypeError(f"fig_size should be a tuple, got the type of {type(fig_size).__name__}.")
    if any([not isinstance(fs, int) for fs in fig_size]):
        raise TypeError(f"fig_size should have integer elements.")
    if len(fig_size)!=2:
        raise ValueError(f"fig_size should be of length 2, got {len(fig_size)}.")
    if not isinstance(num_data, int):
        raise TypeError(f"num_data should be an integer value, got the type of {type(num_data).__name__}.")
    if num_data<1:
        raise ValueError(f"num_data should be a positive integer, got {num_data}.")
    if not isinstance(design_var, np.ndarray):
        raise TypeError(f"design_var should be a numpy array, got the type of {type(design_var).__name__}.")
    if design_var.ndim!=1:
        raise ValueError(f"dv1 should be a 1D array, got {design_var.ndim} dimensions.")
    if not isinstance(bins, int):
        raise TypeError(f"bins should be an integer value, got the type of {type(bins).__name__}.")
    if ranges is not None:
        if not isinstance(ranges, tuple):
            raise TypeError(f"ranges should be a tuple or None, got the type of {type(ranges).__name__}.")
        if not all([isinstance(r, float) for r in ranges]):
            raise TypeError(f"ranges should contain float elements.")
        if len(ranges)!=2:
            raise ValueError(f"ranges should be of length 2, got {len(ranges)}.")
    if not isinstance(mode, str):
        raise TypeError(f"mode should be a string value, got the type of {type(mode).__name__}.")
    if mode not in ["average", "sum"]:
        raise ValueError(f"mode should be either 'average' or 'sum', got {mode}.")
    if not isinstance(x_label, str):
        raise TypeError(f"x_label should be a string value, got the type of {type(x_label).__name__}.")
    if not isinstance(y_label, str):
        raise TypeError(f"y_label should be a string value, got the type of {type(y_label).__name__}.")
    if not isinstance(title, str):
        raise TypeError(f"title should be a string value, got the type of {type(title).__name__}.")

    ## Drawing a figure
    figure=plt.figure(figsize=fig_size)
    ax=figure.add_subplot(1,1,1)

    ## Creating a histogram with np.nistogram()
    ## hist: The values of the resulting histogram.
    ## dv1_pos: The values at the edge of each bins. 
    hist, dv1_pos=np.histogram(design_var, bins=bins, range=ranges)
    if mode=='average':
        hist=hist/num_data

    ## Determining the width of each bar in the figure.
    bin_width = (ranges[1] - ranges[0]) / bins if ranges else dv1_pos[1] - dv1_pos[0]
    ## Determining the position (x-coordinate) of the center of each bar.
    bin_centers = dv1_pos[:-1] + bin_width / 2
    
    ## Drawing a bar plot with the obtained histogram values
    ax.bar(
        x=bin_centers,
        height=hist.ravel(),
        width=bin_width,
        bottom=0,
        align='center',
        color='skyblue',       
        alpha=0.6,             
        edgecolor='black',     
        linewidth=0.8          
    )

    ## Setting the labels and the title.
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    return figure, ax



## This function plots the design selection frequencies of 2D GPAL as a 3D histogram.
## GPAL selects the optimal design value among design candidates in an adaptive manner.
## We can observe the distribution of those 'selected' design values (of the design variable),
## therefore examine features of the function of our interest.
## As "2D" in the function name implies, this functions plots a 3DD histogram,
## where the (x,y) coordinate denotes the design values and the z-axis indicates the frequencies.


## Parameter Descriptions.
## fig_size: The size of the figure. Must be a tuple holding integer values.
## num_data: The number of selected design values (i.e. the optimal design candidates).  
## design_var: A numpy array holding all selected design (coordinate) values. 
##             Each row corresponds to a single design. Must contain 2 columns (considering 2D GPAL).
## bins: A list holding the number of equal-length bins for the x,y axis. 
##       Each element corresponds to the number of bins for each axis. Must be of length 2.
## ranges: A list of 2 lists, indicating the total range of the selected design values.
##         The first list elements specifies the range of x-coordinate values of the selected designs.
##         The second list element specifies the range of y-coordinate values of the selected designs.
## x_label: The text label associated with the x-axis (first design variable)
## y_label: The text label associated with the y-axis (second design variable)
## z_label: The text label associated with the z-axis.
## title: The title of the figure.
## mode: A string determining the mode of the histogram. Must be either 'sum' or 'average'.
##       Setting it to 'average' will normalize the histogram values.


## NOTE: The ranges parameter will be automatically set to the folloiwng, if not specified explicitly.
##       ranges = [[np.min(design_var[:,0]), np.max(design_var[:,0])],
##                 [np.min(design_var[:,1]), np.max(design_var[:,1])]]

## Return Value Specifications
## figure: The whole resulting figure.
## ax: A plot visualized in the figure.
def plot_frequency_histogram_2D(fig_size:Tuple[int, int], 
                                num_data:int, 
                                design_vars:npt.NDArray, 
                                bins:list[int], 
                                ranges:Optional[list[list[float]]], 
                                xlabel:str, 
                                ylabel:str, 
                                zlabel:str, 
                                title:str, 
                                mode:str='sum'):
    
    if not isinstance(fig_size, tuple):
        raise TypeError(f"fig_size should be a tuple, got the type of {type(fig_size).__name__}.")
    if any([not isinstance(fs, int) for fs in fig_size]):
        raise ValueError(f"fig_size should contain integer elements.")
    if len(fig_size)!=2:
        raise ValueError(f"fig_size should be of length 2, got {len(fig_size)}.")
    if not isinstance(num_data, int):
        raise TypeError(f"num_data should be an integer value, got the type of {type(num_data).__name__}.")
    if not isinstance(design_vars, np.ndarray):
        raise TypeError(f"design_vars should be a numpy array, got the type of {type(design_vars).__name__}.")
    if design_vars.ndim!=2:
        raise ValueError(f"design_vars should be a 2D array, got {design_vars.ndim} dimensions.")
    if design_vars.shape[1]!=2:
        raise ValueError(f"design_vars should have two columns, got {design_vars.shape[1]} columns.")
    if not isinstance(bins, list):
        raise TypeError(f"bins should be a list, got the type of {type(bins).__name__}.")
    if not all([isinstance(b, int) for b in bins]):
        raise TypeError(f"bins should contain integer values.")
    if len(bins)!=2:
        raise ValueError(f"bins should be of length 2, got {len(bins)}.")
    if ranges is not None:
        if not isinstance(ranges, list):
            raise TypeError(f"ranges should be a list, got the type of {type(ranges).__name__}.")
        if not all([isinstance(r, list) for r in ranges]):
            raise TypeError(f"ranges should contain list elements.")
        if not all([isinstance(r1, float) for r1 in ranges[0]]):
            raise TypeError(f"ranges[0] should contain float type elements.")
        if not all([isinstance(r2, float) for r2 in ranges[1]]):
            raise TypeError(f"ranges[1] should contain float type elements.")
        if len(ranges)!=2:
            raise ValueError(f"ranges should contain two list elements, got {len(ranges)}.")
        if len(ranges[0])!=2:
            raise ValueError(f"ranges[0] should contain two float elements, got {len(ranges[0])}.")
        if len(ranges[1])!=2:
            raise ValueError(f"ranges[1] should contain two float elements, got {len(ranges[1])}.")
    
    if not isinstance(xlabel, str):
        raise TypeError(f"xlabel should be a string value, got the type of {type(xlabel).__name__}.")
    if not isinstance(ylabel, str):
        raise TypeError(f"ylabel should be a string value, got the type of {type(ylabel).__name__}.")
    if not isinstance(zlabel, str):
        raise TypeError(f"zlabel should be a string value, got the type of {type(zlabel).__name__}.")
    if not isinstance(title, str):
        raise TypeError(f"title should be a string value, got the type of {type(title).__name__}.")
    if not isinstance(mode, str):
        raise TypeError(f"mode should be a string value, got the type of {type(mode).__name__}.")
    if mode not in ["average", "sum"]:
        raise ValueError(f"mode should be either 'average' or 'sum', got {mode}.")

       
    figure=plt.figure()
    ax=figure.add_subplot(projection='3d')
    hist, dv1_edge, dv2_edge=np.histogram2d(design_vars[:,0], design_vars[:,1], bins=bins, range=ranges)
    if mode=="average":
        hist=hist/num_data

    dv1_pos, dv2_pos=np.meshgrid(dv1_edge[:-1], dv2_edge[:-1], indexing="ij")
    dv1_pos=dv1_pos.ravel()
    dv2_pos=dv2_pos.ravel()
    hist_pos=0

    bin_width=(ranges[0][1]-ranges[0][0])/bins[0] if ranges else dv1_edge[1] - dv1_edge[0]
    bin_depth=(ranges[1][1]-ranges[1][0])/bins[1] if ranges else dv2_edge[1] - dv1_edge[0]
    h=hist.ravel()
    
    bin_centers_dv1 = dv1_pos[:-1] + bin_width/2
    bin_centers_dv2 = dv2_pos[:-1] + bin_depth/2
    
    
    ax.bar3d(x=bin_centers_dv1, y=bin_centers_dv2, z=hist_pos, 
             dx=bin_width, dy=bin_depth, dz=h, zsort='average')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    return figure, ax

def plot_convergence(gp_regressor, x_range, y_range, x_data_points, y_data_points, figure_size=(12, 4), function_colors=["lightgreen", "lightblue", "mediumpurple", "black"]):
    """
        Visualize Gaussian Process (GP) regression convergence.

        This function fits a GaussianProcessRegressor incrementally as more data points are added,
        plots the GP mean function at selected quantiles of the training process, and shows
        the convergence of the mean squared error (MSE) between each trial and the final trial.

        Arguments:
            gp_regressor (GaussianProcessRegressor): 
                An initialized scikit-learn GaussianProcessRegressor object.
                Raises TypeError if the object is not of this type.
            x_range (array-like): 
                Full range of possible X values.
            y_range (array-like): 
                Full range of possible Y values.
            x_data_points (array-like): 
                Observed X data points.
            y_data_points (array-like): 
                Observed Y data points.
            figure_size (tuple, optional): 
                Size of the figure as (width, height). Defaults to (12, 4).
            function_colors (list of strings, optional): 
                Colors for visualizing GP mean functions at different quantiles. 
                Defaults to ["lightgreen", "lightblue", "mediumpurple", "black"].

        Returns:
            tuple:
                - figure (matplotlib.figure.Figure): The overall figure object.
                - axes (numpy.ndarray of matplotlib.axes.Axes): Array containing the two subplot axes.

        Raises:
            TypeError: If `gp_regressor` is not an instance of 
                    sklearn.gaussian_process.GaussianProcessRegressor.
            ValueError: If the lengths of `x_data_points` and `y_data_points` do not match
    """

    if not isinstance(gp_regressor, GaussianProcessRegressor):
        raise TypeError(
            f"gp_regressor must be sklearn.gaussian_process.GaussianProcessRegressor, "
            f"got {type(gp_regressor).__name__}."
        )
    
    if len(x_data_points) != len(y_data_points):
        raise ValueError(
            f"x_data_points and y_data_points must have the same length, "
            f"got {len(x_data_points)} and {len(y_data_points)} for x and y respectively."
        )

    num_data_points = len(x_data_points)
        
    figure, axes = plt.subplots(1, 2, figsize=figure_size)
    gp_mean_function_list = []

    quantiles_visualize = [0.25, 0.5, 0.75, 1]
    n_trials_visualize = []
    for quantile in quantiles_visualize:
        n_trials_visualize.append(int(num_data_points * quantile))

    # Generate plot 1: GP mean functions visualized
    x_range_reshaped_for_gpr = np.array(x_range).reshape(-1, 1)
    quantile_count = 0
    for i in range(1, num_data_points+1):
        x_data_points_current_trial = x_data_points[:i]
        x_data_points_reshaped_for_gpr = np.array(x_data_points_current_trial).reshape(-1, 1)

        y_data_points_current_trial = y_data_points[:i]
        y_data_points_reshaped_for_gpr = np.array(y_data_points_current_trial).reshape(-1, 1)
        
        gp_regressor.fit(x_data_points_reshaped_for_gpr, y_data_points_reshaped_for_gpr)

        gp_mean_function = gp_regressor.predict(x_range_reshaped_for_gpr)

        gp_mean_function_list.append(gp_mean_function)
        
        if i in n_trials_visualize[:-1]:
            current_quantile = int(quantiles_visualize[quantile_count] * 100)
            axes[0].plot(x_range, gp_mean_function, color=function_colors[quantile_count], linewidth=2, label=f"Trial #{i} ({current_quantile}%)")
            quantile_count += 1
        elif i == n_trials_visualize[-1]:
            axes[0].plot(x_range, gp_mean_function, color=function_colors[-1], linewidth=2.5, label="Final Trial")


    # Set plot margins for better visibility
    ymin, ymax = min(y_range), max(y_range)
    span = ymax - ymin
    margin = 0.1 * span
    axes[0].set_ylim(ymin - margin, ymax + margin)
    axes[0].set_title("GP mean functions visualized")
    axes[0].legend()

    # Data points scatter
    axes[0].scatter(x_data_points, y_data_points, c="darkgrey", edgecolor="white", zorder=3, s=30)

    # Generate plot 2: MSE value between each trial and the final trial
    mse_values = []
    print(num_data_points)
    print(len(gp_mean_function_list))
    for i in range(num_data_points):
        mse = mean_squared_error(gp_mean_function_list[i], gp_mean_function_list[-1])
        mse_values.append(mse)

    trials = np.arange(1, len(mse_values) + 1) 
    axes[1].plot(trials, mse_values, marker='o', linewidth=2)
    axes[1].xaxis.set_major_locator(MultipleLocator(1))
    axes[1].set_xlim(0.5, len(mse_values)+0.5)
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Distance to the final function")
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.suptitle("Convergence Plot", fontsize=16, y=1.03)

    return figure, axes