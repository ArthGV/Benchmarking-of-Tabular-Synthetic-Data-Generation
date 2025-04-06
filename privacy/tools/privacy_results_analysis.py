import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

COLORS = ['#001155', '#DD1122', '#11AAFF', '#A63297', '#E61E64', '#0EABA9', '#1781E3', '#5944C6']

def load_results(generator, privacy_summaries, results_path, dataset_id, n_samples, dataset_records, all_metrics=False):
    # adult_BayNetGreatherThan_evaluation_results
    experiment_results = pd.read_csv(results_path / f'{dataset_id}_{generator}_evaluation_results.csv', index_col=0)
    experiment_results = experiment_results.loc[n_samples][['Column Pair Trends', 'Column Shapes', 'Test Mean']] #, 'Authenticity'
    
    privacy_results = privacy_summaries[privacy_summaries['generator'] == generator]
    privacy_vr = privacy_results[privacy_results['target_id'].isin(dataset_records[dataset_id]['vulnerable_records'])]
    privacy_vr_avg = privacy_vr[['auc', 'privacy_gain']].mean().rename(index={'auc': 'MIA AUC VR', 'privacy_gain': 'Privacy Gain VR'})
    privacy_random = privacy_results[privacy_results['target_id'].isin(dataset_records[dataset_id]['random'])]
    privacy_random_avg = privacy_random[['auc', 'privacy_gain']].mean().rename(index={'auc': 'MIA AUC RR', 'privacy_gain': 'Privacy Gain RR'})
    privacy_all_avg = privacy_results[['auc', 'privacy_gain']].mean().rename(index={'auc': 'MIA AUC', 'privacy_gain': 'Privacy Gain'})
    privacy_results = pd.concat([privacy_vr_avg, privacy_random_avg, privacy_all_avg])

    # combine all results for this generator
    combined_results = pd.concat([experiment_results, privacy_results])

    # add the results in a dataframe
    combined_results = pd.DataFrame(combined_results).T
    combined_results = combined_results.rename(index={0: generator}, 
                                            columns={'Test Mean': 'Utility', 'auc': 'MIA AUC', 'privacy_gain': 'Privacy Gain'})

    # inversly the MIA AUC (lower is better)
    combined_results['MIA AUC VR'] = 1 - combined_results['MIA AUC VR']
    combined_results['MIA AUC RR'] = 1 - combined_results['MIA AUC RR']
    combined_results['MIA AUC'] = 1 - combined_results['MIA AUC']
    combined_results = combined_results.rename(columns={'MIA AUC VR': '1 - MIA AUC VR', 'MIA AUC RR': '1 - MIA AUC RR', 'MIA AUC': '1 - MIA AUC'})
    
    # put the MIA AUC and Privacy Gain columns one after the other for easier comparison
    if all_metrics:
        combined_results = combined_results[['Column Pair Trends', 'Column Shapes', 'Utility', '1 - MIA AUC VR', '1 - MIA AUC RR', '1 - MIA AUC']]
    else:
        combined_results = combined_results[['Column Pair Trends', 'Column Shapes', 'Utility', '1 - MIA AUC', 'Privacy Gain']]

    return combined_results

def plot_spider_matplotlib(combined_results, dataset_name, pdf_path=None, svg_path=None, color_mapper=None):
    # Compute angles for each axis
    angles = np.linspace(0, 2 * np.pi, len(combined_results.columns), endpoint=False).tolist()
    angles += angles[:1]  # Repeat first angle to close the plot

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot each generator
    for idx, (_, row) in enumerate(combined_results.iterrows()):
        values = row.tolist()
        values += values[:1]  # Repeat first value to close the plot
        # add small dots on each plotted data point
        color_idx = color_mapper[row.name] if color_mapper else idx
        # as an example, display only the axis and no content in the plot

        ax.plot(angles, values, label=row.name, linewidth=2, color=COLORS[color_idx], linestyle='solid', marker='o')
        ax.fill(angles, values, alpha=0.4, color=COLORS[color_idx])  # Fill the area

    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(combined_results.columns, fontsize=12)

    for label,rot in zip(ax.get_xticklabels(),angles[:-1]):
        if rot < np.pi/2 or rot > 3*np.pi/2:
            ha = 'left'
        else:
            ha = 'right'
        label.set_horizontalalignment(ha)
        label.set_rotation_mode("anchor")

    # Set radial limits (0 to 1 scale)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Add title and legend
    plt.suptitle(f"Generators comparison on {dataset_name}", fontsize=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    if svg_path:
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
    if pdf_path:
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

    # Show the plot
    plt.show()

def plot_privacy_metrics_multi(summaries_df, title, x='generator', metrics=['accuracy', 'true_positive_rate', 
                                                'false_positive_rate', 'mia_advantage',
                                                'privacy_gain', 'auc'], scale=1.0, color_palette=COLORS):
    n_rows = len(metrics) // 3
    n_cols = 3
    if n_rows == 0:
        n_rows = 1
        n_cols=1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*scale, 4*scale))
        axes = [axes]
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24//n_cols*scale, 12//n_rows*scale))
        axes = axes.flatten()

    metric_mapper = {
        'accuracy': 'Accuracy',
        'true_positive_rate': 'True Positive Rate',
        'false_positive_rate': 'False Positive Rate',
        'mia_advantage': 'MIA Advantage',
        'privacy_gain': 'Privacy Gain',
        'auc': 'AUC'
    }

    metric_optimization = {
        'accuracy': 'min',
        'true_positive_rate': 'min',
        'false_positive_rate': 'max',
        'mia_advantage': 'min',
        'privacy_gain': 'max',
        'auc': 'min'
    }

    ticks_rotation = 45 if len(summaries_df[x].unique()) > 2 else 0
    ticks_alignment = 'right' if len(summaries_df[x].unique()) > 2 else 'center'
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        sns.boxplot(data=summaries_df, x=x, y=metric, ax=ax, palette=color_palette, width=.5, medianprops=dict(color="orange", alpha=0.7))
        # ax.set_title(metric_mapper[metric] + f" ({metric_optimization[metric]})")
        # ax.set_title(metric_mapper[metric])
        ax.set_title(title)

        # add a line at the maximal/minimal value
        if x == 'record_type':
            y_max = summaries_df[metric].max()
            y_min = summaries_df[metric].min()
            if metric_optimization[metric] == 'min':
                ax.axhline(y=y_max, color='r', linestyle='--', alpha=0.7)
            else:
                ax.axhline(y=y_min, color='r', linestyle='--', alpha=0.7)

        # set y axis from 0 to 1
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_ylabel(metric_mapper[metric])
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=ticks_rotation, horizontalalignment=ticks_alignment)
        ax.yaxis.grid(True)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()