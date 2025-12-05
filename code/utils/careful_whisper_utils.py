import os, sys
import glob

import pandas as pd
import numpy as np

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns

from itertools import product

############################################
#########$$ Analysis functions #############
############################################

def find_all_model_comparisons(df, main_models, comparison_model, kind='cubic', group=False, stabilization_method=None):
    """
    Find equivalent points between a set of models (main_models) and a target comparison
    model (comparison_model). 
    
    Parameters:
    df: DataFrame with columns ['model_name', 'subset', 'accuracy']
    
    Returns:
    tuple: (DataFrame with comparison results, Dictionary of model curves and equivalent points)
    """
    # Get unique models
    models = df['model_name'].unique()
    
    # Store all comparison results and model data
    all_comparisons = []
    curves = {}

    # Compare each pair of models
    for main, comparison in product(main_models, comparison_model):
        # Get data for both models
        main_data = df[df['model_name'] == main].sort_values('subset')
        comparison_data = df[df['model_name'] == comparison].sort_values('subset')

        print (f"Comparing {main} and {comparison}")
        
        # Create interpolation functions for model1
        try:

            curves[main] = {}

            if group:
                grouped = main_data.groupby(['subset'])[['accuracy', 'perplexity']].mean().reset_index()
            else:
                grouped = main_data.copy()

            if kind in ['linear', 'quadratic', 'cubic']:
                accuracy_curve = interp1d(grouped['accuracy'], grouped['subset'], 
                                        kind=kind, fill_value='extrapolate')

                perplexity_curve = interp1d(grouped['perplexity'], grouped['subset'], 
                            kind=kind, fill_value='extrapolate')
            else:
                fitted = [fit_curves(grouped[metric], grouped['subset'], verbose=True, stabilization_method=stabilization_method) for metric in ['accuracy', 'perplexity']]
                accuracy_curve, perplexity_curve = [fit['function'] for fit in fitted]

                curves[main].update({
                    'accuracy_info': fitted[0],
                    'perplexity_info': fitted[1]
                })

            print (f"\n\nComparison for {main} and {comparison}")
            print (f"Best curve accuracy: {fitted[0]['type']}, R2 = {fitted[0]['r2']}")
            print (f"Best curve perplexity: {fitted[1]['type']}, R2 = {fitted[1]['r2']}")

            curves[main].update({
                'accuracy_curve': accuracy_curve,
                'perplexity_curve': perplexity_curve,
            })
            
            # Find equivalent points
            for i, row in comparison_data.iterrows():
                # Find the comparison accuracy / subset
                comparison_accuracy = row['accuracy']
                comparison_perplexity = row['perplexity']
                comparison_subset = row['subset']
                batch = row['batch_number'] if 'batch_number' in row else None

                current_filter = np.logical_and(
                    main_data['subset'] == comparison_subset, # Filter by current subset
                    main_data['batch_number'] == batch if batch is not None else True # Filter by batch if exists
                )

                current_data = main_data[current_filter]

                if len(current_data) != 1:
                    print (f"No data found: subset {comparison_subset}, batch {batch}", flush=True)
                    continue
                
                main_accuracy, main_perplexity = current_data[['accuracy', 'perplexity']].iloc[0]
                
                try:
                    if group:
                        df_group = comparison_data.groupby(['subset'])[['accuracy', 'perplexity']].mean().reset_index()
                        comparison_accuracy, comparison_perplexity = df_group.loc[df_group['subset'] == comparison_subset, ['accuracy', 'perplexity']].values.T.squeeze().tolist()

                    equivalence_accuracy_point = accuracy_curve(comparison_accuracy)  # Get the equivalent subset for model1
                    equivalence_perplexity_point = perplexity_curve(comparison_perplexity)  # Get the equivalent subset for model1

                    # Append the comparison data
                    all_comparisons.append({
                        'main_model': main,
                        'batch_number': batch,
                        'comparison_model': comparison,
                        'main_accuracy': main_accuracy,
                        'main_perplexity': main_perplexity,
                        'comparison_accuracy': comparison_accuracy,
                        'comparison_perplexity': comparison_perplexity,
                        'true_subset': comparison_subset,
                        'equivalence_accuracy_point': equivalence_accuracy_point,
                        'equivalence_perplexity_point': equivalence_perplexity_point,
                        'subset_accuracy_ratio':  equivalence_accuracy_point / comparison_subset,
                        'subset_perplexity_ratio':  equivalence_perplexity_point / comparison_subset,
                        'pair': f'{main} vs {comparison}'
                    })
                except ValueError:
                    continue
        except ValueError:
            print(f"Warning: Could not create interpolation for {main_model} vs {comparison_model}")
            continue
    # Return both the comparison DataFrame and the results dictionary

    comparison_data = comparison_data.rename(columns={
        'model_name': 'main_model', 
        'accuracy': 'main_accuracy',
        'perplexity': 'main_perplexity',
        'subset': 'true_subset',
    })

    copy_keys = ['main_accuracy', 'main_perplexity', 'true_subset']
    save_keys = ['comparison_accuracy', 'comparison_perpleixty', 'equivalence_accuracy_point']

    comparison_data[save_keys] = comparison_data[copy_keys]
    comparison_data['equivalence_accuracy_point'] = comparison_data['true_subset']

    df_comparisons = pd.DataFrame(all_comparisons)
    df_comparisons = pd.concat([df_comparisons, comparison_data]).reset_index(drop=True)
    return df_comparisons, curves

def fit_curves(x, y, verbose=False, stabilization_method=None):
    """
    Fits multiple types of curves to the data with robust stabilization
    
    Parameters:
    x: array-like, independent variable
    y: array-like, dependent variable
    stabilization_method: str, method to handle outliers 
        Options: 'truncate', 'winsorize', 'weighted', 'huber', 'bisquare'
    
    Returns:
    dict with best fitting function type, parameters, R-squared value,
    and a lambda function for predictions
    """
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c
        
    def power_law(x, a, b, c):
        return a * (x + b)**(-c)

    # Robust curve fitting methods
    def robust_curve_fit(x, y, func, p0, method='huber'):
        """
        Robust curve fitting with multiple stabilization methods
        """
        def huber_weight(residuals, k=1.9):
            # Improved Huber weights to reduce impact of outliers
            scaled_residuals = residuals / (1.4826 * np.median(np.abs(residuals)))
            weights = np.where(np.abs(scaled_residuals) <= k, 1.0, k / np.abs(scaled_residuals))
            return weights
        
        def bisquare_weight(residuals, k=4.685):
            # Improved Tukey's bisquare (biweight) weighting
            scaled_residuals = residuals / (1.4826 * np.median(np.abs(residuals)))
            
            # Soft transition to prevent zero weights
            weights = np.where(
                np.abs(scaled_residuals) <= k, 
                (1 - (scaled_residuals / k)**2)**2, 
                1e-10  # Small non-zero weight instead of zero
            )
            return weights
        
        if method in ['truncate', 'winsorize', 'weighted']:
            # Existing methods remain the same as in the previous implementation
            if method == 'truncate':
                # Remove extreme outliers using IQR method
                Q1 = np.percentile(y, 25)
                Q3 = np.percentile(y, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                mask = (y >= lower_bound) & (y <= upper_bound)
                x_filtered = x[mask]
                y_filtered = y[mask]
                
                # Fit to filtered data
                params, covariance = curve_fit(func, x_filtered, y_filtered, p0=p0, maxfev=10000)
                
            elif method == 'winsorize':
                # Cap extreme values
                y_winsorized = stats.mstats.winsorize(y, limits=[0.05, 0.05])
                params, covariance = curve_fit(func, x, y_winsorized, p0=p0, maxfev=10000)
            
            elif method == 'weighted':
                # Weighted least squares
                # Initial fit to get initial residuals
                initial_params, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
                initial_pred = func(x, *initial_params)
                residuals = y - initial_pred
                
                # Compute weights based on residuals
                weights = 1 / (np.abs(residuals) + 1e-10)
                params, covariance = curve_fit(
                    func, x, y, 
                    p0=initial_params, 
                    sigma=1/weights, 
                    absolute_sigma=True,
                    maxfev=10000
                )
        
        elif method in ['huber', 'bisquare']:
            # Robust iterative weighting methods
            weight_func = huber_weight if method == 'huber' else bisquare_weight
            
            # Initial fit
            initial_params, _ = curve_fit(func, x, y, p0=p0, maxfev=10000)
            initial_pred = func(x, *initial_params)
            residuals = y - initial_pred
            
            # Iterative reweighting
            for _ in range(3):  # Typically 3 iterations are sufficient
                weights = weight_func(residuals)
                
                # Ensure no zero weights to prevent divide by zero
                weights = np.maximum(weights, 1e-10)
                
                params, covariance = curve_fit(
                    func, x, y, 
                    p0=initial_params, 
                    sigma=1/weights, 
                    absolute_sigma=True,
                    maxfev=10000
                )
                initial_params = params
                initial_pred = func(x, *params)
                residuals = y - initial_pred
        
        else:
            print (f"Fitting with no robust method")
            params, covariance = curve_fit(func, x, y, p0=p0, maxfev=10000)
        
        return params, covariance

    # Rest of the function remains the same as in the previous implementation
    # (function definitions for exp_decay, power_law, and the rest of fit_curves)
    # ... [previous implementation continues]

    functions = {
        'exponential_decay': (exp_decay, [0.5, 0.02, 0.2]),
        'power_law': (power_law, [0.5, 0.01, 0.5]),
    }
    
    results = {}
    
    # Try fitting each function with robust methods
    for name, (func, p0) in functions.items():
        try:
            # Robust curve fitting
            params, covariance = robust_curve_fit(x, y, func, p0, method=stabilization_method)
                
            # Calculate predictions and R-squared
            pred = func(x, *params)
            r2 = 1 - np.sum((y - pred)**2) / np.sum((y - np.mean(y))**2)
            
            # Calculate parameter uncertainties
            param_uncertainties = np.sqrt(np.diag(covariance))
            
            results[name] = {
                'type': name,
                'params': params,
                'param_uncertainties': param_uncertainties,
                'r2': r2,
                'function': lambda x, f=func, p=params: f(x, *p),
                'rmse': np.sqrt(np.mean((y - pred)**2)),
                'stabilization_method': stabilization_method
            }

            # Optional verbose output
            if verbose:
                print(f"{name.capitalize()} Robust Fit ({stabilization_method}):")
                print("  Parameters:")
                for i, (param, uncertainty) in enumerate(zip(params, param_uncertainties)):
                    print(f"    p{i}: {param:.4f} ± {uncertainty:.4f}")
                print(f"  R-squared: {r2:.4f}")
                print(f"  RMSE: {results[name]['rmse']:.4f}\n")
            
        except (RuntimeError, ValueError) as e:
            print(f"Could not fit {name} with {stabilization_method} method: {str(e)}")
            continue
    
    if not results:
        print("Error: Could not fit any function to the data.")
        return None
    
    # Find the best fit based on R-squared value
    best_fit = max(results.items(), key=lambda x: x[1]['r2'])
    
    # Add comparison of all fits to the results
    all_fits_comparison = {name: {'r2': info['r2'], 'rmse': info['rmse']} 
                          for name, info in results.items()}
    
    return {
        **best_fit[1],
        'all_fits': all_fits_comparison
    }

def plot_fit(x, y, fit_result, extrapolate_to=100):
    """
    Plots original data with the fitted curve and extrapolation
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Original Data', color='blue', alpha=0.5)
    
    # Generate points for the fitted curve, including extrapolation
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = fit_result['function'](x_fit)
    
    plt.plot(x_fit, y_fit, 'r-', label=f'Fitted {fit_result["type"]} (R² = {fit_result["r2"]:.3f})')
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data with Fitted Curve and Extrapolation')
    plt.legend()
    plt.show()

def plot_all_comparisons(comparison_df, comparison_model, x_axis='true_subset', palette='RdBu_r', metrics=['accuracy', 'perplexity'], visualize_equivalence=False, remove_outliers=False):
    """
    Create a seaborn plot showing all model comparisons, including equivalent points and subset ratios.
    """
    
    # Create figure with two subplots
    fig, axes = plt.subplots(len(metrics), 2, figsize=(7, 3*len(metrics)))
    axes = axes.flatten()

    # Plot the first subplot: Accuracy curves with equivalent points
    counter = 0

    if x_axis == 'true_subset':
        xlim = [0, 110]
        xticks = np.linspace(0, 100, 5)  # 5 ticks evenly spaced from 0 to 9
    elif x_axis == 'hours':
        max_hours = max(comparison_df[x_axis])
        xlim = [0, 1.1*max_hours]
        xticks = np.linspace(0, 800, 5)  # 5 ticks evenly spaced from 0 to 9

    # cm.get_cmap(palette, )

    if remove_outliers:
        comparison_df = remove_df_outliers(comparison_df, 'subset_accuracy_ratio')

    # Go through each metric
    for metric in metrics:
        # Plot equivalent points if requested (e.g., drawing lines between curves)
        if visualize_equivalence:
            for _, df in comparison_df.groupby('main_model'):
                    for i, row in df.iterrows():
                        axes[counter].plot([row[x_axis], row[f'equivalence_{metric}_point']], 
                                [row[f'comparison_{metric}'], row[f'comparison_{metric}']], 
                                '--', color='k', alpha=0.75) #cmap(i)

        # Plot accuracy
        sns.lineplot(
            comparison_df,
            x=x_axis, 
            y=f"main_{metric}", 
            hue='main_model',
            palette=palette, 
            linewidth=1.5, 
            ax=axes[counter],
            legend=False
        )
        
        axes[counter].set_xlabel('Number of samples')
        axes[counter].set_ylabel(f'{metric.capitalize()}')
        axes[counter].set_title(f'Model {metric} curves with equivalence points')

        if metric == 'accuracy':
            axes[counter].set_ylim([0.15, 0.35])
        elif metric == 'perplexity':
            axes[counter].set_ylim([0, 200])

        axes[counter].set_xlim(xlim)
        axes[counter].set_xticks(xticks)

        counter += 1

        curve_fit_dfs = []
        columns = [x_axis, f'subset_{metric}_ratio']

        for i, df in comparison_df.groupby('pair'):

            x, y = df[columns].values.T
            curves = fit_curves(x,y)

            # Generate points for the fitted curve, including extrapolation
            x_fit = np.linspace(min(x), max(x), 200)
            y_fit = curves['function'](x_fit)

            fits = np.stack([x_fit, y_fit]).T
            df = pd.DataFrame(fits, columns=columns)
            df['pair'] = i

            curve_fit_dfs.append(df)

        curve_fit_df = pd.concat(curve_fit_dfs).reset_index(drop=True)

        ratio_df = comparison_df.groupby(['pair', x_axis])[f'subset_{metric}_ratio'] \
            .mean() \
            .reset_index() \

        sns.scatterplot(
            data=ratio_df,
            x=x_axis,
            y=f'subset_{metric}_ratio',
            palette=palette,
            hue='pair',
            ax=axes[counter],
            legend=False,
            s=20
            # style='pair',
        )

        # comparison_df['subset_ratio'] = comparison_df['subset_ratio'] * 100
        # Plot the second subplot: Subset ratio comparison
        sns.lineplot(
            data=curve_fit_df,
            x=x_axis,
            y=f'subset_{metric}_ratio',
            palette=palette,
            hue='pair',
            style='pair',
            linewidth=1.5,
            ax=axes[counter]
        )
        
        # Add a horizontal line at y=14.
        # axes[counter].axhline(y=1.0, color='k', linestyle='-', linewidth=1.5, zorder=0)
        axes[counter].set_ylim([0, 1.1])
        axes[counter].set_xlim(xlim)

        axes[counter].set_xticks(xticks)
        # axes[counter].set_xlim([0, 100])

        # change to percentage ticks
        ticks = axes[counter].get_yticks()
        axes[counter].set_yticklabels(['{:.0%}'.format(x) for x in ticks])
        
        axes[counter].set_xlabel('Samples of text-only data')
        axes[counter].set_ylabel(f'Percent less data')
        axes[counter].set_title(f'Amount of data required for equivalent {metric}')
        axes[counter].legend(title='Model Pairs', bbox_to_anchor=(1.05, 1), loc='upper left')

        counter += 1
        print (counter)

    # Adjust layout to prevent overlap of subplots
    plt.tight_layout()
    sns.despine()

    return plt.gcf()

def remove_df_outliers(df, column, std=1.5):
    """
    Remove outliers from a dataframe based on IQR of a specific column.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    column (str): Name of the column to check for outliers
    
    Returns:
    pandas.DataFrame: Dataframe with outliers removed
    """
    # Check if column exists in dataframe
    if column not in df.columns:
        raise ValueError("Column not found in dataframe.")
    
    # Check if column is numeric
    if not np.issubdtype(df[column].dtype, np.number):
        raise ValueError("Selected column is not numeric.")
    
    # Store initial number of rows
    initial_rows = len(df)
    
    # Calculate IQR bounds
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - std * IQR
    upper_bound = Q3 + std * IQR
    
    # Filter rows based on bounds
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Calculate and print number of rows removed
    removed_rows = initial_rows - len(df_clean)
    print(f"Number of rows removed due to outliers in column {column}: {removed_rows}")
    
    return df_clean