#%%

## Importing required Python packages
import os
import random
import warnings
from typing import Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from mpl_toolkits.mplot3d.axes3d import Axes3D
## Importing key functions from the psychopy package.
## Our number-line task file is implemented based on the psychopy package.
from psychopy import event, logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error

## Importing key functions from our gpal package.
## These three functions must be utilized to conduct GPAL properly.
from gpal.gpal_optimize import gpal_optimize
from gpal.gpr_instance import GPRInstance
from gpal.utils import argsConstructor, sequence_with_interval


def plot_GP(gp_regressor: GaussianProcessRegressor, dataframe: pd.DataFrame, sigma_coefficient = 1, x_range = (), y_range=(), column_names_specified:list=[], trial_numbers_specified=()):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(dataframe)}")

    # Get the datapoints from the dataframe
    if column_names_specified:
        print(f"Dataframe column names {list(dataframe.columns)}")
        n_names_specified = len(column_names_specified)

        if n_names_specified == 2:
            x_column_name, y_column_name = column_names_specified

            # If the specified column namaes can't be searched on the dataframe, raise ValueError
            missing_columns = [col for col in column_names_specified if col not in dataframe.columns]
            if missing_columns:
                raise ValueError(f"Specified column names {missing_columns} not found in the dataframe")
            
            x_data_points = dataframe[x_column_name].tolist()
            y_data_points = dataframe[y_column_name].tolist()

        elif n_names_specified == 3:
            x_column_name, y_column_name, z_column_name = column_names_specified

            # If the specified column namaes can't be searched on the dataframe, raise ValueError
            missing_columns = [col for col in column_names_specified if col not in dataframe.columns]
            if missing_columns:
                raise ValueError(f"Specified column names {missing_columns} not found in the dataframe")
            
            x_data_points = dataframe[x_column_name].tolist()
            y_data_points = dataframe[y_column_name].tolist()
            z_data_points = dataframe[z_column_name].tolist()

        else:
            raise ValueError(f"The number of the specified columns should be either 2 or 3")
    else:
        x_data_points = dataframe.iloc[:, 0].tolist()
        y_data_points = dataframe.iloc[:, 1].tolist()

    # generate subplots as many as the trial numbers
    if len(trial_numbers_specified) == 1 or not trial_numbers_specified:
        number_subplots = 1
    else:
        number_subplots = len(trial_numbers_specified)

    figure, axes = plt.subplots(number_subplots, 1, figsize=(6, 5 * number_subplots))

    for i in range(number_subplots):
        if len(trial_numbers_specified) == 1:
            trial_number = trial_numbers_specified[0]
            x_data_points_specified = x_data_points[:trial_number]
            print(x_data_points_specified)
            y_data_points_specified = y_data_points[:trial_number]

            x_data_points_reshaped = np.asarray(x_data_points_specified, dtype=float).reshape(-1, 1)

            gp_regressor.fit(x_data_points_reshaped, y_data_points_specified)
            
            if x_range:
                x_values_predict = np.linspace(x_range[0], 
                                    x_range[1], 
                                    100).reshape(-1, 1)
            else:
                x_values_predict = np.linspace(min(x_data_points_specified), 
                                    max(x_data_points_specified), 
                                    100).reshape(-1, 1)

            post_mean, post_stdev = gp_regressor.predict(x_values_predict, return_std=True)

            ## Plots the experiment data as a scatterplot.
            axes.scatter(x_data_points_specified, y_data_points_specified, c='black', label='Data Points')

            ## Plots the posterior mean values associated with every design candidate.
            axes.plot(x_values_predict, post_mean, label="Prediction", linewidth=2.5, color='black')

            max_stdev_design = x_values_predict[np.argmax(post_stdev)]
        
            axes.axvline(x=max_stdev_design, color='green', linestyle='--', linewidth=3)
            
            ## Plots the uncertainty range with semi-transparent color.
            axes.fill_between(x_values_predict.ravel(), post_mean-sigma_coefficient*post_stdev, 
                            post_mean+sigma_coefficient*post_stdev, alpha=0.3, label='Uncertainty')
            
            axes.annotate(
                f"{int(max_stdev_design)}",
                xy=(max_stdev_design, 0),             
                xycoords=("data", "axes fraction"),
                xytext=(0, -17.5),                   
                textcoords="offset points",
                ha="center", va="top",
                fontsize=12, color="green",
                fontweight="bold"
            )

            axes.set_title(f"Trial #{trial_number}")

            if y_range:
                axes.set_ylim(y_range[0], y_range[1])

            axes.legend()

        elif len(trial_numbers_specified) >= 2:
            # datapoints
            trial_number = trial_numbers_specified[i]
            x_data_points_specified = x_data_points[:trial_number]
            y_data_points_specified = y_data_points[:trial_number]
            print(f"Current X data points: {x_data_points_specified}")
            print(f"Current Y data points: {y_data_points_specified}")

            x_data_points_reshaped = np.asarray(x_data_points_specified, dtype=float).reshape(-1, 1)
            gp_regressor.fit(x_data_points_reshaped, y_data_points_specified)
            
            if not x_range:
                x_values_predict = np.linspace(min(x_data_points_specified), 
                                    max(x_data_points_specified), 
                                    100).reshape(-1, 1)
            else:
                x_values_predict = np.linspace(x_range[0], 
                                    x_range[1], 
                                    100).reshape(-1, 1)
                
            post_mean, post_stdev = gp_regressor.predict(x_values_predict, return_std=True)

            axes[i].scatter(x_data_points_specified, y_data_points_specified, c='black', label='Data Points')

            axes[i].plot(x_values_predict.ravel(), post_mean, label="Prediction", linewidth=2.5, color='black')
            
            axes[i].fill_between(x_values_predict.ravel(), post_mean-sigma_coefficient*post_stdev, 
                            post_mean+sigma_coefficient*post_stdev, alpha=0.3, label='Uncertainty')

            max_stdev_design = x_values_predict[np.argmax(post_stdev)]
        
            axes[i].axvline(x=max_stdev_design, color='green', linestyle='--', linewidth=3)
            axes[i].annotate(
                f"{int(max_stdev_design)}",
                xy=(max_stdev_design, 0),             
                xycoords=("data", "axes fraction"),
                xytext=(0, -17.5),                   
                textcoords="offset points",
                ha="center", va="top",
                fontsize=12, color="green",
                fontweight="bold"    
            )
            
            axes[i].set_title(f"Trial #{trial_number}")

            axes[i].legend()
            

            if y_range:
                axes[i].set_ylim(y_range[0], y_range[1])
            
        else:
            x_data_points_reshaped = np.asarray(x_data_points, dtype=float).reshape(-1, 1)
            gp_regressor.fit(x_data_points_reshaped, y_data_points)
            
            if not x_range:
                x_values_predict = np.linspace(min(x_data_points), 
                                    max(x_data_points), 
                                    100).reshape(-1, 1)
            else:
                x_values_predict = np.linspace(x_range[0], 
                                    x_range[1], 
                                    100).reshape(-1, 1)

            post_mean, post_stdev = gp_regressor.predict(x_values_predict, return_std=True)

            ## Plots the experiment data as a scatterplot.
            axes.scatter(x_data_points, y_data_points, c='black', label='Data Points')

            ## Plots the posterior mean values associated with every design candidate.
            axes.plot(x_values_predict, post_mean, label="Prediction", linewidth=2.5, color='black')
            
            ## Plots the uncertainty range with semi-transparent color.
            axes.fill_between(x_values_predict.ravel(), post_mean-sigma_coefficient*post_stdev, 
                            post_mean+sigma_coefficient*post_stdev, alpha=0.3, label='Uncertainty')

            axes.legend()

            if y_range:
                axes.set_ylim(y_range[0], y_range[1])

    return figure, axes, gp_regressor



    # TODO 2D experiment
    # TODO add plotting configs


def plot_selection_frequency(
    dataframe: pd.DataFrame,
    bins: int = 10,
    x_range: Tuple[float, float] = (),
    column_names_specified: Tuple[str, ...] = ()
):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(dataframe)}")

    if column_names_specified:
        print(f"Dataframe column names {list(dataframe.columns)}")
        n_names_specified = len(column_names_specified)

        if n_names_specified == 2:
            x_column_name, y_column_name = column_names_specified

            missing_columns = [col for col in column_names_specified if col not in dataframe.columns]
            if missing_columns:
                raise ValueError(f"Specified column names {missing_columns} not found in the dataframe")
            
            x_data_points = dataframe[x_column_name].tolist()
            y_data_points = dataframe[y_column_name].tolist()

        elif n_names_specified == 3:
            x_column_name, y_column_name, z_column_name = column_names_specified

            missing_columns = [col for col in column_names_specified if col not in dataframe.columns]
            if missing_columns:
                raise ValueError(f"Specified column names {missing_columns} not found in the dataframe")
            
            x_data_points = dataframe[x_column_name].tolist()
            y_data_points = dataframe[y_column_name].tolist()
            z_data_points = dataframe[z_column_name].tolist()

        else:
            raise ValueError(f"The number of the specified columns should be either 2 or 3")
    else:
        x_data_points = dataframe.iloc[:, 0].tolist()
        y_data_points = dataframe.iloc[:, 1].tolist()

    fig_size = (10, 5)
    figure=plt.figure(figsize=fig_size)
    ax= figure.add_subplot(1,1,1)
    hist, dv1_pos=np.histogram(x_data_points, bins=bins)

    hist=hist/len(x_data_points)    ## Determining the width of each bar in the figure.

    bin_width = (x_range[1] - x_range[0]) / bins if x_range else dv1_pos[1] - dv1_pos[0]

    bin_centers = dv1_pos[:-1] + bin_width / 2
    
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

    ax.set_xlabel("Numbers")
    ax.set_ylabel("Selection Frequency")
    ax.set_title("Design Selection Histogram")
    
    return figure, ax


def plot_convergence(gp_regressor: GaussianProcessRegressor, dataframe: pd.DataFrame, x_range = (), y_range=(), column_names_specified:tuple=(), figure_size=(12, 4), function_colors=["lightgreen", "lightblue", "mediumpurple", "black"]):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(dataframe)}")

    # Get the datapoints from the df
    if column_names_specified:
        n_names_specified = len(column_names_specified)

        if n_names_specified == 2:
            x_column_name, y_column_name = column_names_specified

            # If the specified column names can't be searched on the dataframe, raise ValueError
            missing_columns = [col for col in column_names_specified if col not in dataframe.columns]
            if missing_columns:
                raise ValueError(f"Specified column names {missing_columns} not found in the dataframe")
            
            x_data_points = dataframe[x_column_name].tolist()
            y_data_points = dataframe[y_column_name].tolist()

        elif n_names_specified == 3:
            x_column_name, y_column_name, z_column_name = column_names_specified

            # If the specified column namaes can't be searched on the dataframe, raise ValueError
            missing_columns = [col for col in column_names_specified if col not in dataframe.columns]
            if missing_columns:
                raise ValueError(f"Specified column names {missing_columns} not found in the dataframe")
            
            x_data_points = dataframe[x_column_name].tolist()
            y_data_points = dataframe[y_column_name].tolist()
            z_data_points = dataframe[z_column_name].tolist()

        else:
            raise ValueError(f"The number of the specified columns should be either 2 or 3")
        
    else:
        x_data_points = dataframe.iloc[:, 0].tolist()
        y_data_points = dataframe.iloc[:, 1].tolist()

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
    if not x_range:
        x_range = np.linspace(min(x_data_points), 
                            max(x_data_points), 100)
        x_range_reshaped_for_gpr = x_range.reshape(-1, 1)
    else:
        x_range_reshaped_for_gpr = np.linspace(x_range[0], 
                            x_range[1], 
                            100).reshape(-1, 1)
            
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
            axes[0].plot(x_range_reshaped_for_gpr.ravel(), gp_mean_function, color=function_colors[quantile_count], linewidth=2, label=f"Trial #{i} ({current_quantile}%)")
            quantile_count += 1
        elif i == n_trials_visualize[-1]:
            axes[0].plot(x_range_reshaped_for_gpr.ravel(), gp_mean_function, color=function_colors[-1], linewidth=2.5, label="Final Trial")


    # Set plot margins for better visibility
    ymin, ymax = min(y_data_points), max(y_data_points)
    span = ymax - ymin
    margin = 0.1 * span
    axes[0].set_ylim(ymin - margin, ymax + margin)
    axes[0].set_title("GP mean functions visualized")
    if y_range:
        axes[0].set_ylim(y_range[0], y_range[1])
    axes[0].legend()

    # Data points scatter
    axes[0].scatter(x_data_points, y_data_points, c="black", edgecolor="white", zorder=3, s=30)

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

    return figure, axes, mse_values


def plot_convergence_and_frequency(gp_regressor: GaussianProcessRegressor, dataframe: pd.DataFrame, bins = 10, x_range_arg = (), y_range=(), column_names_specified:tuple=(), figure_size=(10, 20), function_colors=["lightgreen", "lightblue", "mediumpurple", "black"]):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(dataframe)}")

    # Get the datapoints from the df
    if column_names_specified:
        n_names_specified = len(column_names_specified)

        if n_names_specified == 2:
            x_column_name, y_column_name = column_names_specified

            # If the specified column names can't be searched on the dataframe, raise ValueError
            missing_columns = [col for col in column_names_specified if col not in dataframe.columns]
            if missing_columns:
                raise ValueError(f"Specified column names {missing_columns} not found in the dataframe")
            
            x_data_points = dataframe[x_column_name].tolist()
            y_data_points = dataframe[y_column_name].tolist()

        elif n_names_specified == 3:
            x_column_name, y_column_name, z_column_name = column_names_specified

            # If the specified column namaes can't be searched on the dataframe, raise ValueError
            missing_columns = [col for col in column_names_specified if col not in dataframe.columns]
            if missing_columns:
                raise ValueError(f"Specified column names {missing_columns} not found in the dataframe")
            
            x_data_points = dataframe[x_column_name].tolist()
            y_data_points = dataframe[y_column_name].tolist()
            z_data_points = dataframe[z_column_name].tolist()

        else:
            raise ValueError(f"The number of the specified columns should be either 2 or 3")
        
    else:
        x_data_points = dataframe.iloc[:, 0].tolist()
        y_data_points = dataframe.iloc[:, 1].tolist()

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
    
    figure, axes = plt.subplots(3, 1, figsize=figure_size)
    gp_mean_function_list = []

    quantiles_visualize = [0.25, 0.5, 0.75, 1]
    n_trials_visualize = []
    for quantile in quantiles_visualize:
        n_trials_visualize.append(int(num_data_points * quantile))

    # Generate plot 1: GP mean functions visualized
    if x_range_arg:
        x_range_reshaped_for_gpr = np.linspace(x_range_arg[0], 
                            x_range_arg[1], 
                            100).reshape(-1, 1)
    else:
        x_range = np.linspace(min(x_data_points), 
                            max(x_data_points), 100)
        x_range_reshaped_for_gpr = x_range.reshape(-1, 1)
            
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
            axes[0].plot(x_range_reshaped_for_gpr.ravel(), gp_mean_function, color=function_colors[quantile_count], linewidth=2, label=f"Trial #{i} ({current_quantile}%)")
            quantile_count += 1
        elif i == n_trials_visualize[-1]:
            axes[0].plot(x_range_reshaped_for_gpr.ravel(), gp_mean_function, color=function_colors[-1], linewidth=2.5, label="Final Trial")


    # Set plot margins for better visibility
    ymin, ymax = min(y_data_points), max(y_data_points)
    span = ymax - ymin
    margin = 0.1 * span
    axes[0].set_ylim(ymin - margin, ymax + margin)
    axes[0].set_title("Figure A: GP mean functions visualized")
    if y_range:
        axes[0].set_ylim(y_range[0], y_range[1])
    axes[0].legend()

    # Data points scatter
    axes[0].scatter(x_data_points, y_data_points, c="black", edgecolor="white", zorder=3, s=30)

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
    axes[1].set_title("Figure B: Distance to the final function")
    axes[1].grid(True, alpha=0.3)


    # frequency histogram

    hist, dv1_pos=np.histogram(x_data_points, bins=bins)

    hist=hist/len(x_data_points) 

    bin_width = (x_range_arg[1] - x_range_arg[0]) / bins if x_range_arg else dv1_pos[1] - dv1_pos[0]

    bin_centers = dv1_pos[:-1] + bin_width / 2
    
    axes[2].bar(
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

    axes[2].set_xlabel("Numbers")
    axes[2].set_ylabel("Selection Frequency")
    axes[2].set_title("Figure C: Design Selection Histogram")
    
    return figure, axes
