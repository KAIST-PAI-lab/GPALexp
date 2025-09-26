#%%
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error



def plot_GP(gp_regressor: GaussianProcessRegressor, 
            dataframe: pd.DataFrame, 
            x_range: Optional[Tuple[float, float]] = None, 
            y_range: Optional[Tuple[float, float]] = None,
            x_num: int = 100, 
            column_names_specified:Optional[list[str]] = None, 
            trial_numbers_specified:Optional[list[int]] = None,
            figure_size: Tuple[int, int] = (6,4),
            sigma_coefficient: float = 1.0,
            font_size_title = 20,
            font_size_axis_label = 15,
            x_label_name = "Stimulus",
            y_label_name = "Response"):
    
    if not isinstance(gp_regressor, GaussianProcessRegressor):
        raise TypeError(f"gp_regressor should be a GaussianProcessRegressor object, got the type of {type(gp_regressor).__name__}.")
    if not isinstance(dataframe, pd.DataFrame):                 
        raise TypeError(f"dataframe should be a Pandas DataFrame, got the type of {type(dataframe).__name__}.")
    if x_range is not None:
        if not isinstance(x_range, Tuple):
            raise TypeError(f"x_range should be a Tuple, got the type of {type(x_range).__name__}.")
        else:
            if len(x_range)!=2:
                raise ValueError(f"x_range should have 2 elements, got {len(x_range)} elements.")
        if not all([isinstance(r, float) for r in x_range]):
            raise TypeError(f"x_range should only contain float values.")
        if x_range[0] >= x_range[1]:
            raise ValueError(f"x_range[1] should be larger than x_range[0].")
    if y_range is not None:
        if not isinstance(y_range, Tuple):
            raise TypeError(f"y_range should be a Tuple, got the type of {type(y_range).__name__}.")
        else:
            if len(y_range)!=2:
                raise ValueError(f"y_range should have 2 elements, got {len(y_range)} elements.")
        if not all([isinstance(r, float) for r in y_range]):
            raise TypeError(f"y_range should only contain float values.")
        if y_range[0] >= y_range[1]:
            raise ValueError(f"y_range[1] should be larger than y_range[0].")
    if not isinstance(x_num, int):
        raise TypeError(f"x_num should be an integer value, got the type of {type(x_num).__name__}.")    
    if not x_num>0:
        raise ValueError(f"x_num should be a positive integer, got {x_num}.")
    if column_names_specified is not None:
        if not isinstance(column_names_specified, list):
            raise TypeError(f"column_names_specified should be a list, got the type of {type(column_names_specified).__name__}.")
        if not all([isinstance(cn, str) for cn in column_names_specified]):
            raise TypeError(f"column_names_specified should only contain string elements.")
        if len(column_names_specified)<2 or len(column_names_specified)>3:
            raise ValueError(f"column_names_specified should be of length 2 or 3, got the length of {len(column_names_specified)}.")
    if trial_numbers_specified is not None:
        if not isinstance(trial_numbers_specified, list):
            raise TypeError(f"trial_numbers_specified should be a list, got the type of {type(trial_numbers_specified).__name__}.")
        if len(trial_numbers_specified)<1:
            raise ValueError(f"trial_numbers_specified should not be an empty list.")
        if not all([isinstance(tn, int) for tn in trial_numbers_specified]):
            raise TypeError(f"trial_numbers_specified should only contain string elements.") 
        if not all([tn <= dataframe.shape[0] for tn in trial_numbers_specified]):
            raise ValueError(f"trial_numbers_specified should not contain values larger than {dataframe.shape[0]}.")
    if not isinstance(figure_size, Tuple):
        raise TypeError(f"figure_size should be a Tuple, got the type of {type(figure_size).__name__}.")
    if len(figure_size)!=2:
        raise ValueError(f"figure_size should have 2 elements, got {len(figure_size)} elements.")
    if not all([isinstance(fs, int) for fs in figure_size]):
        raise TypeError(f"figure_size should only contain int values.")
    if not isinstance(sigma_coefficient, float):
        raise TypeError(f"sigma_coefficient should be a float value, got the type of {type(sigma_coefficient).__name__}.")       


    # Get the datapoints from the dataframe
    if column_names_specified is not None:
        #print(f"The columns included in dataframe: {list(dataframe.columns)}.")
        n_names_specified = len(column_names_specified)

        if n_names_specified == 2:
            x_column_name, y_column_name = column_names_specified

            # If the specified column namaes can't be searched on the dataframe, raise ValueError
            missing_columns = [col for col in column_names_specified if col not in dataframe.columns]
            if len(missing_columns)>0:
                raise ValueError(f"The following columns cannot be found in the dataframe: {missing_columns}.")
            
            x_data_points = dataframe[x_column_name].to_numpy()
            y_data_points = dataframe[y_column_name].to_numpy()

        ## 3D Uncertainty plot - not implemented yet
        elif n_names_specified == 3:
            x_column_name, y_column_name, z_column_name = column_names_specified

            # If the specified column namaes can't be searched on the dataframe, raise ValueError
            missing_columns = [col for col in column_names_specified if col not in dataframe.columns]
            if len(missing_columns)>0:
                raise ValueError(f"The following columns cannot be found in the dataframe: {missing_columns}.")
            
            x_data_points = dataframe[x_column_name].to_numpy()
            y_data_points = dataframe[y_column_name].to_numpy()
            z_data_points = dataframe[z_column_name].to_numpy()

    else:
        x_data_points = dataframe.iloc[:, 0].to_numpy()
        y_data_points = dataframe.iloc[:, 1].to_numpy()

    # generate subplots as many as the trial numbers
    if trial_numbers_specified is None:
        number_subplots = 1
    else:
        number_subplots = len(trial_numbers_specified)
        figure_size = (6, 5*number_subplots)

    figure, axes = plt.subplots(number_subplots, 1, figsize=figure_size)

    if trial_numbers_specified is not None:
        if len(trial_numbers_specified) == 1:
            trial_number = trial_numbers_specified[0]
            x_data_points_specified = x_data_points[:trial_number]

            y_data_points_specified = y_data_points[:trial_number]

            x_data_points_reshaped = x_data_points_specified.reshape(-1, 1)

            gp_regressor.fit(x_data_points_reshaped, y_data_points_specified)
            
            if x_range is not None:
                x_values_predict = np.linspace(x_range[0], 
                                    x_range[1], 
                                    x_num).reshape(-1, 1)
            else:
                x_values_predict = np.linspace(min(x_data_points_specified), 
                                    max(x_data_points_specified), 
                                    x_num).reshape(-1, 1)

            post_mean, post_stdev = gp_regressor.predict(x_values_predict, return_std=True)

            ## Plots the experiment data as a scatterplot.
            axes.scatter(x_data_points_specified, y_data_points_specified, c='black', label='Data Points')

            ## Plots the posterior mean values associated with every design candidate.
            axes.plot(x_values_predict, post_mean, label="Prediction", linewidth=2.5, color='black')

            max_stdev_design = x_values_predict[np.argmax(post_stdev)].item()
        
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

            axes.set_title(f"Trial #{trial_number}", fontsize = font_size_title)
            axes.set_xlabel(f"{x_label_name}", fontsize = font_size_axis_label) 
            axes.set_ylabel(f"{y_label_name}", fontsize = font_size_axis_label) 

            if y_range:
                axes.set_ylim(y_range[0], y_range[1])

            axes.legend()
        
        
        elif len(trial_numbers_specified) >= 2:
            for i in range(number_subplots):
                # datapoints
                trial_number = trial_numbers_specified[i]

                x_data_points_specified = x_data_points[:trial_number]
                y_data_points_specified = y_data_points[:trial_number]

                x_data_points_reshaped = x_data_points_specified.reshape(-1, 1)
                gp_regressor.fit(x_data_points_reshaped, y_data_points_specified)

                if x_range is not None:                    
                    x_values_predict = np.linspace(x_range[0], 
                                        x_range[1], 
                                        x_num).reshape(-1, 1)
                else:
                    x_values_predict = np.linspace(min(x_data_points_specified), 
                                        max(x_data_points_specified), 
                                        x_num).reshape(-1, 1)

                    
                post_mean, post_stdev = gp_regressor.predict(x_values_predict, return_std=True)

                axes[i].scatter(x_data_points_specified, y_data_points_specified, c='black', label='Data Points')

                axes[i].plot(x_values_predict.ravel(), post_mean, label="Prediction", linewidth=2.5, color='black')
                
                axes[i].fill_between(x_values_predict.ravel(), post_mean-sigma_coefficient*post_stdev, 
                                post_mean+sigma_coefficient*post_stdev, alpha=0.3, label='Uncertainty')

                max_stdev_design = x_values_predict[np.argmax(post_stdev)].item()


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
                
                axes[i].set_title(f"Trial #{trial_number}", fontsize = font_size_title)
                axes[i].set_xlabel(f"{x_label_name}", fontsize = font_size_axis_label) 
                axes[i].set_ylabel(f"{y_label_name}", fontsize = font_size_axis_label) 

                axes[i].legend()
                
                if y_range:
                    axes[i].set_ylim(y_range[0], y_range[1])

            
    else:
        x_data_points_specified = x_data_points
        x_data_points_reshaped = x_data_points_specified.reshape(-1, 1)
        gp_regressor.fit(x_data_points_reshaped, y_data_points)
        
        if x_range is not None:
            x_values_predict = np.linspace(x_range[0], 
                                x_range[1], 
                                x_num).reshape(-1, 1)

        else:
            x_values_predict = np.linspace(min(x_data_points), 
                                max(x_data_points), 
                                x_num).reshape(-1, 1)

        post_mean, post_stdev = gp_regressor.predict(x_values_predict, return_std=True)

        ## Plots the experiment data as a scatterplot.
        axes.scatter(x_data_points, y_data_points, c='black', label='Data Points')

        ## Plots the posterior mean values associated with every design candidate.
        axes.plot(x_values_predict.ravel(), post_mean, label="Prediction", linewidth=2.5, color='black')
        
        ## Plots the uncertainty range with semi-transparent color.
        axes.fill_between(x_values_predict.ravel(), post_mean-sigma_coefficient*post_stdev, 
                        post_mean+sigma_coefficient*post_stdev, alpha=0.3, label='Uncertainty')

        axes.set_xlabel(f"{x_label_name}", fontsize = font_size_axis_label) 
        axes.set_ylabel(f"{y_label_name}", fontsize = font_size_axis_label) 

        axes.legend()

        if y_range:
            axes.set_ylim(y_range[0], y_range[1])

    figure.tight_layout()
    return figure, axes



    # TODO 2D experiment
    # TODO add plotting configs


def plot_selection_frequency(
    dataframe: pd.DataFrame,
    bins: int = 10,
    val_range: Tuple[float, float] = (0.0, 500.0),
    column_names_specified: Optional[str] = None,
    figure_size: Tuple[int, int] = (6, 4), 
    mode: str = 'sum',
    font_size_title = 20,
    font_size_axis_label = 15,

):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"dataframe should be a Pandas DataFrame, got the type of {type(dataframe).__name__}.")
    if not isinstance(bins, int):
        raise TypeError(f"bins should be an integer value, got the type of {type(bins).__name__}.")
    if not bins>0:
        raise ValueError(f"bins should be a positive integer, got {bins}.")
    if not isinstance(val_range, Tuple):
        raise TypeError(f"val_range should be a Tuple, got the type of {type(val_range).__name__}.")
    else:
        if len(val_range)!=2:
            raise ValueError(f"val_range should have 2 elements, got {len(val_range)} elements.")
    if not all([isinstance(r, float) for r in val_range]):
        raise TypeError(f"val_range should only contain float values.")
    if val_range[0] >= val_range[1]:
        raise ValueError(f"val_range[1] should be larger than val_range[0].")
    if column_names_specified is not None:
        if not isinstance(column_names_specified, str):
            raise TypeError(f"column_names_specified should be a string value, got the type of {type(column_names_specified).__name__}.")
    if not isinstance(figure_size, Tuple):
        raise TypeError(f"figure_size should be a Tuple, got the type of {type(figure_size).__name__}.")
    if len(figure_size)!=2:
        raise ValueError(f"figure_size should have 2 elements, got {len(figure_size)} elements.")
    if not all([isinstance(fs, int) for fs in figure_size]):
        raise TypeError(f"figure_size should only contain int values.")
    if not isinstance(mode, str):
        raise TypeError(f"mode should be a string value, got the type of {type(mode).__name__}.")
    if mode not in ['sum', 'average']:
        raise ValueError(f"mode should be either 'sum' or 'average', got {mode}.")

    if column_names_specified is not None:
        #print(f"The columns included in dataframe: {list(dataframe.columns)}.")

        target_data_points = dataframe[column_names_specified].to_numpy()
        
        '''
        n_names_specified = len(column_names_specified)

        if n_names_specified == 2:
            x_column_name, y_column_name = column_names_specified

            missing_columns = [col for col in column_names_specified if col not in dataframe.columns]
            if len(missing_columns)>0:
                raise ValueError(f"The following columns cannot be found in the dataframe: {missing_columns}.")
            
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
        '''
    else:
        target_data_points = dataframe.iloc[:, 0].to_numpy()
        '''
        y_data_points = dataframe.iloc[:, 1].tolist()
        '''
    
    figure=plt.figure(figsize=figure_size)
    ax= figure.add_subplot(1,1,1)

    mask_range = lambda v: v >= val_range[0] and v <= val_range[1]
    mask_binary = np.array([mask_range(val) for val in target_data_points])
    target_data_points=target_data_points[mask_binary]

    hist, dv1_pos=np.histogram(target_data_points, bins=bins, range=val_range)
    #print(dv1_pos)
    if mode=='average':
        hist=hist/len(target_data_points)    ## Determining the width of each bar in the figure.

    if val_range is not None:
        bin_width = (val_range[1] - val_range[0])/bins
    else:
        bin_width = (dv1_pos[-1] - dv1_pos[0])/bins

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
    if mode=='sum':
    	ax.yaxis.set_major_locator(MultipleLocator(1))

    ax.set_xlabel("Stimulus Range", fontsize = font_size_axis_label)
    if mode == "sum":
        ax.set_ylabel("Selection Frequency", fontsize = font_size_axis_label)
    elif mode == "average":
        ax.set_ylabel("Selection Ratio", fontsize = font_size_axis_label)
    ax.set_title("Stimulus Selection Histogram", fontsize = font_size_title)
    figure.tight_layout()

    return figure, ax


def plot_convergence(gp_regressor: GaussianProcessRegressor, 
                     dataframe: pd.DataFrame, 
                     x_range: Optional[Tuple[float, float]] = None, 
                     y_range: Optional[Tuple[float, float]] = None, 
                     x_num: int = 100,
                     column_names_specified: Optional[list[str]] = None, 
                     figure_size: Tuple[int, int]=(12, 4), 
                     function_colors: list[str]=["lightgreen", "lightblue", "mediumpurple", "black"],
                     font_size_title = 20,
                     font_size_axis_label = 15):
    
    if not isinstance(gp_regressor, GaussianProcessRegressor):
        raise TypeError(f"gp_regressor should be a GaussianProcessRegressor object, got the type of {type(gp_regressor).__name__}.")
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"dataframe should be a Pandas DataFrame, got the type of {type(dataframe).__name__}.")
    if x_range is not None:
        if not isinstance(x_range, Tuple):
            raise TypeError(f"x_range should be a Tuple, got the type of {type(x_range).__name__}.")
        else:
            if len(x_range)!=2:
                raise ValueError(f"x_range should have 2 elements, got {len(x_range)} elements.")
        if not all([isinstance(r, float) for r in x_range]):
            raise TypeError(f"x_range should only contain float values.")
        if x_range[0] >= x_range[1]:
            raise ValueError(f"x_range[1] should be larger than x_range[0].")
    if y_range is not None:
        if not isinstance(y_range, Tuple):
            raise TypeError(f"y_range should be a Tuple, got the type of {type(y_range).__name__}.")
        else:
            if len(y_range)!=2:
                raise ValueError(f"y_range should have 2 elements, got {len(y_range)} elements.")
        if not all([isinstance(r, float) for r in y_range]):
            raise TypeError(f"y_range should only contain float values.")
        if y_range[0] >= y_range[1]:
            raise ValueError(f"y_range[1] should be larger than y_range[0].")
    if not isinstance(x_num, int):
        raise TypeError(f"x_num should be an integer value, got the type of {type(x_num).__name__}.")
    if not x_num>0:
        raise ValueError(f"x_num should be a positive integer, got {x_num}.")
    
    if not isinstance(figure_size, Tuple):
        raise TypeError(f"figure_size should be a Tuple, got the type of {type(figure_size).__name__}.")
    if len(figure_size)!=2:
        raise ValueError(f"figure_size should have 2 elements, got {len(figure_size)} elements.")
    if not all([isinstance(fs, int) for fs in figure_size]):
        raise TypeError(f"figure_size should only contain int values.")
     
    if column_names_specified is not None:
        if not isinstance(column_names_specified, list):
            raise TypeError(f"column_names_specified should be a list, got the type of {type(column_names_specified).__name__}.")
        if not all([isinstance(cn, str) for cn in column_names_specified]):
            raise TypeError(f"column_names_specified should only contain string elements.")
        if len(column_names_specified)<2 or len(column_names_specified)>3:
            raise ValueError(f"column_names_specified should be of length 2 or 3, got the length of {len(column_names_specified)}.")


    if column_names_specified is not None:
        n_names_specified = len(column_names_specified)

        if n_names_specified == 2:
            x_column_name, y_column_name = column_names_specified

            missing_columns = [col for col in column_names_specified if col not in dataframe.columns]
            if len(missing_columns)>0:
                raise ValueError(f"The following columns cannot be found in the dataframe: {missing_columns}.")
            
            x_data_points = dataframe[x_column_name].to_numpy()
            y_data_points = dataframe[y_column_name].to_numpy()

        elif n_names_specified == 3:
            x_column_name, y_column_name, z_column_name = column_names_specified

            # If the specified column namaes can't be searched on the dataframe, raise ValueError
            missing_columns = [col for col in column_names_specified if col not in dataframe.columns]
            if len(missing_columns)>0:
                raise ValueError(f"The following columns cannot be found in the dataframe: {missing_columns}.")
            
            x_data_points = dataframe[x_column_name].to_numpy()
            y_data_points = dataframe[y_column_name].to_numpy()
            z_data_points = dataframe[z_column_name].to_numpy()
        
    else:
        x_data_points = dataframe.iloc[:, 0].to_numpy()
        y_data_points = dataframe.iloc[:, 1].to_numpy()

    
    num_data_points = len(x_data_points)
    
    figure, axes = plt.subplots(1, 2, figsize=figure_size)
    gp_mean_function_list = []

    quantiles_visualize = [0.25, 0.5, 0.75, 1]
    n_trials_visualize = []
    for quantile in quantiles_visualize:
        n_trials_visualize.append(int(num_data_points * quantile))

    # Generate plot 1: GP mean functions visualized
    if x_range is not None:
        x_values_predict = np.linspace(x_range[0], x_range[1], x_num)
    else:
        x_values_predict = np.linspace(min(x_data_points), max(x_data_points), x_num)
    x_range_reshaped_for_gpr = x_values_predict.reshape(-1,1)
    
            
    quantile_count = 0
    for i in range(1, num_data_points+1):
        x_data_points_current_trial = x_data_points[:i].reshape(-1,1)

        y_data_points_current_trial = y_data_points[:i]
        
        gp_regressor.fit(x_data_points_current_trial, y_data_points_current_trial)

        gp_mean_function = gp_regressor.predict(x_range_reshaped_for_gpr)

        gp_mean_function_list.append(gp_mean_function)

        if i in n_trials_visualize[:-1]:
            current_quantile = int(quantiles_visualize[quantile_count] * 100)
            axes[1].plot(x_range_reshaped_for_gpr.ravel(), gp_mean_function, color=function_colors[quantile_count], linewidth=2, label=f"Trial #{i} ({current_quantile}%)")
            quantile_count += 1
        elif i == n_trials_visualize[-1]:
            axes[1].plot(x_range_reshaped_for_gpr.ravel(), gp_mean_function, color=function_colors[-1], linewidth=2.5, label="Final Trial")


    # Set plot margins for better visibility
    if y_range:
        ymin, ymax = y_range[0], y_range[1]
    else:
        ymin, ymax = min(y_data_points), max(y_data_points)
    span = ymax - ymin
    margin = 0.1 * span
    axes[1].set_ylim(ymin - margin, ymax + margin)
    axes[1].set_title("GP Mean Functions", fontsize=font_size_title)
    axes[1].set_xlabel("Stimulus", fontsize = font_size_axis_label)
    axes[1].set_ylabel("Response", fontsize = font_size_axis_label)
    axes[1].legend()

    # Data points scatter
    axes[1].scatter(x_data_points, y_data_points, c="black", edgecolor="white", zorder=3, s=30)

    # Generate plot 2: MSE value between each trial and the final trial
    mse_values = []
    #print(num_data_points)
    #print(len(gp_mean_function_list))
    for i in range(num_data_points):
        mse = mean_squared_error(gp_mean_function_list[i], gp_mean_function_list[-1])
        mse_values.append(mse)

    trials = np.arange(1, len(mse_values) + 1) 
    axes[0].plot(trials, mse_values, marker='o', linewidth=2)
    axes[0].xaxis.set_major_locator(MultipleLocator(1))
    axes[0].set_xlim(0.5, len(mse_values)+0.5)
    axes[0].set_xlabel("Trial", fontsize=font_size_axis_label)
    axes[0].set_ylabel("MSE", fontsize=font_size_axis_label)
    axes[0].set_title("Distance to the Final Function", fontsize=font_size_title)
    axes[0].grid(True, alpha=0.3)

    figure.tight_layout()

    return figure, axes, mse_values