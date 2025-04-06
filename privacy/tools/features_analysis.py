"""
@author: Ajkuna Seipi, October 2024
@Update: Colin Pelltier, January 2025
"""

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

# custom
from tools.eda import compute_cramer_v_contingency_matrix, plot_cramer_v_heatmap

# Encode categorical features using LabelEncoder
def encode_categorical_features(df, categorical_features):
    label_encoders = {}
    df_encoded = df.copy()
    for col in categorical_features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
    return df_encoded, label_encoders

# Pearson Correlation for numerical features
def correlation_heatmaps(df, numerical_features, df_synth=None, method="pearson"):
    print(f"{method.capitalize()} Correlation for Numerical Features:")
    corr = df[numerical_features].corr(method=method)
    
    if df_synth is not None:
        corr_synth = df_synth[numerical_features].corr(method=method)
        corr_diff = corr - corr_synth
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=axes[0])
        axes[0].set_title(f"{method.capitalize()} Correlation (Original Data)")
        sns.heatmap(corr_synth, annot=True, cmap="coolwarm", ax=axes[1])
        axes[1].set_title(f"{method.capitalize()} Correlation (Synthetic Data)")
        sns.heatmap(corr_diff, annot=True, cmap="coolwarm", ax=axes[2])
        axes[2].set_title(f"{method.capitalize()} Correlation Difference")
    else:
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title(f"{method.capitalize()} Correlation between Numerical Features")
    
    plt.tight_layout()
    plt.show()
    

# Chi-square test
def chi_square_heatmap(df, categorical_features, df_synth=None):
    print("Chi-Square Test p-values:")
    
    # Initialize a matrix to store p-values for the original dataset
    p_value_matrix_df = pd.DataFrame(index=categorical_features, columns=categorical_features)
    chi2_value_matrix_df = pd.DataFrame(index=categorical_features, columns=categorical_features)
    
    # Calculate Chi-Square for df
    for cat_feature1 in categorical_features:
        for cat_feature2 in categorical_features:
            if cat_feature1 != cat_feature2:
                contingency_table = pd.crosstab(df[cat_feature1], df[cat_feature2])
                chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
              #  print(f"Chi-Square between {cat_feature1} and {cat_feature2}: chi2={chi2}, p-value={p}")
                p_value_matrix_df.loc[cat_feature1, cat_feature2] = p
                p_value_matrix_df.loc[cat_feature2, cat_feature1] = p  
                
                chi2_value_matrix_df.loc[cat_feature1, cat_feature2] = chi2
                chi2_value_matrix_df.loc[cat_feature2, cat_feature1] = chi2 # Symmetry

    # If df_synth is provided, calculate chi2-values for df_synth and plot the difference
    if df_synth is not None:
        print("Calculating chi2-values for the synthetic dataset...")
        
        chi2_value_matrix_synth = pd.DataFrame(index=categorical_features, columns=categorical_features)
        
        # Calculate Chi-Square for df_synth
        for cat_feature1 in categorical_features:
            for cat_feature2 in categorical_features:
                if cat_feature1 != cat_feature2:
                    contingency_table = pd.crosstab(df_synth[cat_feature1], df_synth[cat_feature2])
                    chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
                   #print(f"Synthetic Chi-Square between {cat_feature1} and {cat_feature2}: chi2={chi2}, p-value={p}")
                    chi2_value_matrix_synth.loc[cat_feature1, cat_feature2] = chi2
                    chi2_value_matrix_synth.loc[cat_feature2, cat_feature1] = chi2  # Symmetry

        # Calculate the difference between the p-value matrices
        chi2_value_diff = chi2_value_matrix_df.astype(float) - chi2_value_matrix_synth.astype(float)

        # Plot the heatmap for the difference in p-values
        plt.figure(figsize=(10, 6))
        sns.heatmap(chi2_value_diff, annot=True, cmap="coolwarm", fmt=".3f", cbar_kws={'label': 'chi2-value difference'})
        plt.title("Chi-Square Test statistic-value Difference Heatmap (Original vs Synthetic)")
        plt.tight_layout()
        plt.show()

    else:
        # Plot the heatmap for the p-values of the original dataset
        plt.figure(figsize=(10, 6))
        sns.heatmap(p_value_matrix_df.astype(float), annot=True, cmap="coolwarm", fmt=".3f", cbar_kws={'label': 'p-value'})
        plt.title("Chi-Square Test p-value Heatmap")
        plt.tight_layout()
        plt.show()



def cramer_v_heatmap(df, categorical_features, df_synth=None):
    """
    Plot the Cramér's V matrix heatmap, and the coefficients
      difference if df_synth is provided.

    Parameters:
    -----------
    df: pd.DataFrame
        The original dataset.
    categorical_features: list
        The list of categorical features to analyze.

    Returns:
    --------
    None
    """
    n_cols = 3 if df_synth is not None else 1
    fig, axs = plt.subplots(1, n_cols, figsize=(n_cols*12, 8))
    
    if df_synth is not None:
        cramer_v = compute_cramer_v_contingency_matrix(df, categorical_features)
        plot_cramer_v_heatmap(cramer_v, label="Original", ax=axs[0])
        cramer_v_synth = compute_cramer_v_contingency_matrix(df_synth, categorical_features)
        plot_cramer_v_heatmap(cramer_v_synth, label="Synthetic", ax=axs[1])
        cramer_v_diff = cramer_v - cramer_v_synth
        plot_cramer_v_heatmap(cramer_v_diff, label="Difference", ax=axs[2])
    else:
        cramer_v = compute_cramer_v_contingency_matrix(df, categorical_features)
        plot_cramer_v_heatmap(cramer_v, ax=axs)

    plt.show()
    

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


def distribution_plots(df, features, df_synth=None, plot_type='hist', n_cols=3):
    print(f"Comparing {plot_type.capitalize()} Plots:")
    n_rows = math.ceil(len(features) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        if plot_type == 'hist':
            if is_numeric_dtype(df[feature]):
                # define bins beforehand for consistent bins across histograms
                bins = np.histogram_bin_edges(df[feature], bins='auto')
                sns.histplot(df[feature], kde=True, ax=axes[i], color='blue', label='Original', stat="density", bins=bins)
                if df_synth is not None:
                    sns.histplot(df_synth[feature], kde=True, ax=axes[i], color='orange', label='Synthetic', stat="density", bins=bins)
            else:
                if df_synth is None:
                    sns.countplot(df[feature], ax=axes[i], color='blue', label='Original')
                else:
                    sub_df_ori = df[[feature]].copy()
                    sub_df_ori['type']="Original"
                    sub_df_synth = df_synth[[feature]].copy()
                    sub_df_synth['type']="Syntshetic"

                    sns.countplot(pd.concat([sub_df_ori, sub_df_synth], axis=0), x=feature, hue='type', alpha=0.8, ax=axes[i])
        elif plot_type == 'box':
            sns.boxplot(data=df, y=feature, ax=axes[i], color='blue', boxprops=dict(alpha=.5))
            if df_synth is not None:
                sns.boxplot(data=df_synth, y=feature, ax=axes[i], color='orange', boxprops=dict(alpha=.5))

        elif plot_type == 'violin':
            sns.violinplot(data=df, y=feature, ax=axes[i], color='blue', alpha=.5)
            if df_synth is not None:
                sns.violinplot(data=df_synth, y=feature, ax=axes[i], color='orange', alpha=.5)

        axes[i].set_title(f"Distribution of {feature}")
        #axes[i].legend()

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
                             
                             
     # Adding a global legend
    legend_handles = []
    if df_synth is not None:
        legend_handles = [plt.Line2D([0], [0], color='blue', lw=4, label='Original'),
                          plt.Line2D([0], [0], color='orange', lw=4, label='Synthetic')]
    else:
        legend_handles = [plt.Line2D([0], [0], color='blue', lw=4, label='Original')]

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
        sns.pairplot(combined_df, hue='Dataset', diag_kind='kde', palette={'Original': 'blue', 'Synthetic': 'orange'}, plot_kws=dict(alpha=0.3))
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
def run_analysis(df, numerical_features, categorical_features, df_synth=None, is_cat_vs_num=False):
    print("Running analysis...")

    # Comparative Plots
    if len(numerical_features) > 0:
        distribution_plots(df, numerical_features, df_synth, plot_type="hist")
        if df_synth is not None:
            distribution_plots(df, numerical_features, df_synth, plot_type="box")
            distribution_plots(df, numerical_features, df_synth, plot_type="violin")
    if len(categorical_features) > 0:
        distribution_plots(df, categorical_features, df_synth, plot_type="hist")

    # Correlation Heatmaps
    print("Correlations")
    if len(numerical_features) > 0:
        correlation_heatmaps(df, numerical_features, df_synth, method='pearson')
        correlation_heatmaps(df, numerical_features, df_synth, method='spearman')
    if len(categorical_features) > 0:  
        # chi_square_heatmap(df, categorical_features, df_synth)
        cramer_v_heatmap(df, categorical_features, df_synth)
    
    # Pair plot
    if len(numerical_features) > 0:
        pair_plot_numerical(df, numerical_features, df_synth)
    
    # Box and Violin Plots
    if len(numerical_features) > 0 and len(categorical_features) > 0 and is_cat_vs_num:
        plot_categorical_vs_numerical(df, df_synth, numerical_features, categorical_features, plot_type='box')
        plot_categorical_vs_numerical(df, df_synth, numerical_features, categorical_features, plot_type='violin')

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
    

# post processing
def post_process_great(df_ori, df_great, features):
    df_clean = df_great.copy()
    
    # Separate numerical and categorical features
    numerical_features = df_ori.select_dtypes(include='number').columns.intersection(features)
    categorical_features = df_ori.select_dtypes(exclude='number').columns.intersection(features)
    
    range_cond = lambda x, y: (y >= x.min()) & (y <= x.max())
    
    # Apply filtering based on categorical and numerical features
    conditions = []

    # Categorical features filtering
    for feature in categorical_features:
        conditions.append(df_clean[feature].isin(df_ori[feature].unique()))
    
    # Numerical features filtering
    for feature in numerical_features:
        conditions.append(range_cond(df_ori[feature], df_clean[feature]))
    
    final_condition = conditions[0]
    for condition in conditions[1:]:
        final_condition &= condition
    
    return df_clean[final_condition]
