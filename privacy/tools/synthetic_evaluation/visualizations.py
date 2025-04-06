import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import math

from pandas.api.types import is_numeric_dtype
from tools.eda import compute_cramer_v_contingency_matrix

DEFAULT_BLUE = '#1f77b4'
DEFAULT_ORANGE = '#ff7f0e'

# General function for plotting numerical vs categorical features (Box, Violin, etc.)
def plot_categorical_vs_numerical(df, df_synth=None, numerical_features=None, categorical_features=None, plot_type='box'):
    n_cols = 2
    n_rows = math.ceil(len(categorical_features) * len(numerical_features) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
    axes = axes.flatten()
    plot_count = 0
    name_fig_type = plot_type

    for cat_feature in categorical_features:
        for num_feature in numerical_features:
            if plot_type == 'box':
                sns.boxplot(x=df[cat_feature], y=df[num_feature], ax=axes[plot_count], color='skyblue', boxprops=dict(alpha=.5))
                if df_synth is not None:
                    sns.boxplot(x=df_synth[cat_feature], y=df_synth[num_feature], ax=axes[plot_count], color='lightcoral', boxprops=dict(alpha=.5))
            elif plot_type == 'violin':
                sns.violinplot(x=df[cat_feature], y=df[num_feature], ax=axes[plot_count], color='skyblue', alpha=.5)
                if df_synth is not None:
                    sns.violinplot(x=df_synth[cat_feature], y=df_synth[num_feature], ax=axes[plot_count], color='lightcoral', alpha=.5)
                
                    
            axes[plot_count].set_title(f"{cat_feature} vs {num_feature}")
            axes[plot_count].tick_params(axis='x', labelrotation=45)
            plot_count += 1

    # Remove empty axes
    for j in range(plot_count, len(axes)):
        fig.delaxes(axes[j])
        
    # Adding a global legend
    legend_handles = []
    if df_synth is not None:
        legend_handles = [plt.Line2D([0], [0], color='skyblue', lw=4, label='Original'),
                          plt.Line2D([0], [0], color='lightcoral', lw=4, label='Synthetic')]
    else:
        legend_handles = [plt.Line2D([0], [0], color='skyblue', lw=4, label='Original')]

    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.15, 1), title="Dataset")

    
    plt.tight_layout()
    #plt.savefig(name_fig_type+"_plot_level1_great.png", bbox_extra_artists=(legend_handles), bbox_inches='tight')
    plt.show()


def distribution_plots(df, features, df_synth=None, generator_name='Generator', plot_type='hist', n_cols=3):
    print(f"Comparing {plot_type.capitalize()} Plots:")
    n_rows = math.ceil(len(features) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    # fig.subplots_adjust(top=10)
    fig.suptitle(f"Features distribution comparison between Original and {generator_name} datasets", y=0.999)

    axes = axes.flatten()

    for i, feature in enumerate(features):
        if plot_type == 'hist':
            if is_numeric_dtype(df[feature]):
                # define bins beforehand for consistent bins across histograms
                bins = np.histogram_bin_edges(df[feature], bins='auto')
                sns.histplot(df[feature], kde=True, ax=axes[i], color=DEFAULT_BLUE, label='Original', stat="density", bins=bins)
                if df_synth is not None:
                    sns.histplot(df_synth[feature], kde=True, ax=axes[i], color=DEFAULT_ORANGE, label=generator_name, stat="density", bins=bins)
            else:
                if df_synth is None:
                    sns.countplot(data=df, x=feature, ax=axes[i], color=DEFAULT_BLUE, label='Original')
                else:
                    sub_df_ori = df[[feature]].copy()
                    sub_df_ori['type']="Original"
                    sub_df_synth = df_synth[[feature]].copy()
                    sub_df_synth['type']=generator_name

                    sns.countplot(pd.concat([sub_df_ori, sub_df_synth], axis=0), x=feature, hue='type', alpha=0.8, ax=axes[i])
        elif plot_type == 'box':
            sns.boxplot(data=df, y=feature, ax=axes[i], color=DEFAULT_BLUE, boxprops=dict(alpha=.5))
            if df_synth is not None:
                sns.boxplot(data=df_synth, y=feature, ax=axes[i], color=DEFAULT_ORANGE, boxprops=dict(alpha=.5))

        elif plot_type == 'violin':
            sns.violinplot(data=df, y=feature, ax=axes[i], color=DEFAULT_BLUE, alpha=.5)
            if df_synth is not None:
                sns.violinplot(data=df_synth, y=feature, ax=axes[i], color=DEFAULT_ORANGE, alpha=.5)

        axes[i].set_title(f"Distribution of {feature}")
        axes[i].legend().remove()

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
                             
                             
     # Adding a global legend
    legend_handles = []
    if df_synth is not None:
        legend_handles = [plt.Line2D([0], [0], color=DEFAULT_BLUE, lw=4, label='Original'),
                          plt.Line2D([0], [0], color=DEFAULT_ORANGE, lw=4, label=generator_name)]
    else:
        legend_handles = [plt.Line2D([0], [0], color=DEFAULT_BLUE, lw=4, label='Original')]

    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.15, 1), title="Dataset")
    plt.tight_layout()
    plt.show()

# Pair plot 
def pair_plot_numerical(df, numerical_features, df_synth=None):
    print("Pair Plot for Numerical Features:")
    
    df_c = df.copy()

    if df_synth is not None:
        df_synth_c = df_synth.copy()
        
        df_c['Dataset'] = 'Original'
        df_synth_c['Dataset'] = 'Synthetic'
        
        features = list(numerical_features) + ['Dataset']
        combined_df = pd.concat([df_c[features], df_synth_c[features]])
        
        # Create the pair plot with 'Dataset' as the hue to differentiate
        sns.pairplot(combined_df, hue='Dataset', diag_kind='kde', palette={'Original': DEFAULT_BLUE, 'Synthetic': DEFAULT_ORANGE}, plot_kws=dict(alpha=0.3))
    else:
        # If no synthetic data is provided, just plot the original dataset
        sns.pairplot(df[numerical_features], diag_kind='kde')
        
    plt.tight_layout()
    #plt.savefig("pair_plot_level1_great.png")
    plt.show()


# General t-SNE function
def tsne_plot(df, df_synth=None, numerical_features=None, n_components=2, perplexity=30, random_state=42):
    print("t-SNE plot:")
    
    # Combine datasets if needed
    if df_synth is not None:
        df_combined = pd.concat([df[numerical_features], df_synth[numerical_features]], axis=0)
        labels = ['Original'] * len(df) + ['Synthetic'] * len(df_synth)
    else:
        df_combined = df[numerical_features]
        labels = ['Original'] * len(df)
    
    # Scale data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_combined)

    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    tsne_results = tsne.fit_transform(df_scaled)

    tsne_df = pd.DataFrame(tsne_results, columns=[f'TSNE{i+1}' for i in range(n_components)])
    tsne_df['Dataset'] = labels
    
    if n_components == 2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='TSNE1', y='TSNE2', hue='Dataset', data=tsne_df, alpha=.5)
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], tsne_df['TSNE3'], c=tsne_df['Dataset'], alpha=.5)
    
    plt.title(f"{n_components}D t-SNE Plot")
    plt.show()
    
    
# To plot more complexe bias
def plot_bias(df, x, y, bias_col, alpha=0.7, figsize=(10, 10), n_legend_col=1, legend_shift=2.0):    
    fig, axs = plt.subplots(3, 2, figsize=figsize, sharey='row')

    sns.scatterplot(df, x=x, y=y, hue=bias_col, ax=axs[0, 0], alpha=alpha)
    sns.histplot(df, x=x, hue=bias_col, alpha=alpha, kde=True, ax=axs[1, 0] , legend=False)
    sns.histplot(df, x=x, alpha=alpha, kde=True, ax=axs[1, 1])
    sns.histplot(df, x=y, hue=bias_col, alpha=alpha, kde=True, ax=axs[2, 0], legend=False)
    sns.histplot(df, x=y, alpha=alpha, kde=True, ax=axs[2, 1])
    
    axs[0, 0].legend(loc="upper right", ncol=n_legend_col, bbox_to_anchor=(legend_shift, 1.07))
    fig.delaxes(axs[0, 1])
    
    #plt.savefig("plot_bias_level1_ori.png")
    plt.show()


# Main Function to run the analyses
def run_analysis(df, df_synth=None):
    print("Running analysis...")

    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns
    all_features = list(numerical_features) + list(categorical_features)
    
    #print(f"Numerical Features: {numerical_features}")
    #print(f"Categorical Features: {categorical_features}")
    
    # Comparative Plots
    distribution_plots(df, all_features, df_synth)
    if len(numerical_features)>0:
        distribution_plots(df, all_features, df_synth, plot_type="box")
        distribution_plots(df, all_features, df_synth, plot_type="violin")
    
        # Pair plot
        pair_plot_numerical(df, numerical_features, df_synth)
    
        # Box and Violin Plots
        plot_categorical_vs_numerical(df, df_synth, numerical_features, categorical_features, plot_type='box')
        plot_categorical_vs_numerical(df, df_synth, numerical_features, categorical_features, plot_type='violin')
    
    # t-SNE Plot
    #tsne_plot(df, df_synth, numerical_features)

    print("Analysis completed!")

# scatter plot of cantons
def plot_swiss_map(df, cantons='all', n_cols=3, legend_shift=2.0, alpha=0.8, n_legend_col=3):
    cantons_list = df['canton'].unique()
    ch_fr_cantons = ["JU", "GE", "VS", "VD", "FR", "BE", "NE"]
    ch_de_ti_cantons = [canton for canton in df.canton.unique() if canton not in ch_fr_cantons] + ["BE"]
    
    if cantons == 'ch_fr':
        df_plot = df[df.canton.isin(ch_fr_cantons)]  
    elif cantons== 'ch_de':
        df_plot = df[df.canton.isin(ch_de_ti_cantons)]  
    else:
        df_plot = df

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    sns.scatterplot(df, x='lon', y='lat', hue='canton', ax=axs[0], alpha=alpha)

    axs[0].legend(loc="upper right", ncol=n_legend_col, bbox_to_anchor=(legend_shift, 1.07))
    fig.delaxes(axs[1])
    
    plt.tight_layout()
    #plt.savefig("swiss_great_1k_level2.png")
    plt.show()

    
def plot_simple_biased_dataset(ori_df, synth_df):
    """
    Plot the distribution of categorical and numerical features between the original and synthetic datasets.
    
    Parameters:
    - ori_df: pd.DataFrame - Original dataset with 'gender' and 'salary' columns.
    - synth_df: pd.DataFrame - Synthetic dataset with 'gender' and 'salary' columns.
    """
    # Ensure 'gender' categories in synth_df match ori_df, marking unknown categories
    ori_categories = set(ori_df['gender'].unique())
    synth_df['gender'] = synth_df['gender'].apply(lambda x: x if x in ori_categories else 'Unknown')
    
    # Set up the plot layout
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Comparison of Gender and Salary Distributions between Original and Synthetic Datasets")

    # Plot 1: Bar plot for 'gender' distribution
    gender_counts_ori = ori_df['gender'].value_counts().reset_index()
    gender_counts_ori.columns = ['gender', 'count']
    gender_counts_ori['dataset'] = 'Original'

    gender_counts_synth = synth_df['gender'].value_counts().reset_index()
    gender_counts_synth.columns = ['gender', 'count']
    gender_counts_synth['dataset'] = 'Synthetic'

    gender_counts = pd.concat([gender_counts_ori, gender_counts_synth])

    sns.barplot(x='gender', y='count', hue='dataset', data=gender_counts, ax=axes[0])
    axes[0].set_title('Gender Distribution')
    axes[0].set_xlabel('Gender')
    axes[0].set_ylabel('Frequency')

    # Plot 2: Distribution plot for 'salary'
    sns.kdeplot(ori_df['salary'], label='Original', ax=axes[1], fill=True, common_norm=False)
    sns.kdeplot(synth_df['salary'], label='Synthetic', ax=axes[1], fill=True, common_norm=False)
    axes[1].set_title('Salary Distribution')
    axes[1].set_xlabel('Salary')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the suptitle
    plt.show()

    
def plot_difference_metrics(df, result_type="average", hue="dataset"): 
    df_long = df.copy()
    
    if result_type == 'average':
        df_long = df.melt(id_vars='dataset', var_name='Metric', value_name='Value')
        
    elif result_type == 'result': 
        df_long = df.melt(id_vars=['dataset', 'sample_size', 'k_param'], var_name='Metric', value_name='Value')

    # Set up the plot with custom hue
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_long, x='Metric', y='Value', hue=hue)

    # Rotate x-ticks and add labels
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Metric Values')
    plt.title(f'Metric Comparison with Hue: {hue.capitalize()}')
    plt.legend(title=hue.capitalize())
    plt.tight_layout()
    plt.show()
    
    
# Function to create subplots for metric comparison based on selected grouping
def plot_metric_barplot(df, group_by='sample_size'):
    # Ensure the group_by column is either 'sample_size' or 'k_param'
    if group_by not in ['sample_size', 'k_param']:
        raise ValueError("group_by must be either 'sample_size' or 'k_param'")

    # Drop the unused column based on the selected group_by
    df = df.drop(columns=[col for col in ['sample_size', 'k_param'] if col != group_by])

    # Get unique values for the selected grouping variable
    unique_values = df[group_by].unique()

    # Set up the subplot grid
    num_subplots = len(unique_values)
    fig, axes = plt.subplots(nrows=num_subplots, figsize=(14, 5 * num_subplots), sharex=True)
    fig.suptitle(f'Metric Comparison by {group_by.capitalize()} Across Datasets', fontsize=16)

    # Convert DataFrame to long format for easier plotting
    df_long = df.melt(id_vars=['dataset', group_by], var_name='Metric', value_name='Value')

    # Loop over each unique value to create a subplot
    for i, value in enumerate(unique_values):
        
        subset = df_long[df_long[group_by] == value]

        # Plot on the corresponding subplot axis
        sns.barplot(data=subset, x='Metric', y='Value', hue='dataset', ax=axes[i] if num_subplots > 1 else axes)
        axes[i if num_subplots > 1 else 0].set_title(f'{group_by.capitalize()} = {value}')
        axes[i if num_subplots > 1 else 0].set_ylabel('Metric Value')
        axes[i if num_subplots > 1 else 0].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    
def plot_metrics_lineplot(df, x_axis='k_param', show_plot=False):
    metrics = df.columns.difference(['dataset', 'sample_size', 'k_param', 'column shapes', 'column trends'])
        
    # Get unique x-axis values to use as ticks, ensuring they are sorted
    x_ticks = sorted(df[x_axis].unique())
    
    # Set up 2x4 subplot layout
    fig, axs = plt.subplots(3, 2, figsize=(14, 14))
    axs = axs.flatten()  # Flatten to easily iterate over in a single loop

    for i, metric in enumerate(metrics):
        for dataset_name in ['dataset1', 'dataset2']:
            # Filter and sort the DataFrame by the x_axis column
            df_filtered = df[df['dataset'] == dataset_name].sort_values(by=x_axis)
            
            # Plot the metric
            axs[i].plot(df_filtered[x_axis], df_filtered[metric], label=dataset_name, marker='o')
        
        # Set labels, title, and x-ticks for each subplot
        axs[i].set_ylabel(metric)
        axs[i].set_title(f'{metric}')
        axs[i].set_xticks(x_ticks)  # Set x-ticks to dataset values only
        axs[i].grid()
    
    # Remove any empty subplots
    for j in range(len(metrics), len(axs)):
        fig.delaxes(axs[j])

    title = f"Metrics Comparison by {x_axis.capitalize()}"
    
    # Set a common x-axis label for each subplot
    fig.supxlabel(x_axis)
    fig.legend(['dataset1', 'dataset2'], loc="upper right", title="Datasets")
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    if show_plot:
        plt.show()

    return fig, axs