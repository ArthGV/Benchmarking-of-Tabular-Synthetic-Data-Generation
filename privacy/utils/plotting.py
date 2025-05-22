import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

def plot(complexity: np.array, mean: np.array, var: np.array = None):
    plt.scatter(complexity, mean, c='red')
    if var is not None:
        lower_bound = mean - var
        upper_bound = mean + var
        plt.scatter(complexity, mean, c='red')
        plt.fill_between(complexity, lower_bound, upper_bound, alpha=.3, color='red')

def double_plot(generator: str, complexity: np.array, mean_0: np.array, std_0: np.array, mean_1: np.array, std_1: np.array, baseline_score):
    sns.set_style("whitegrid")  # Clean background with gridlines
    sns.set_palette("muted")  # Muted colors for a more refined look

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Larger plot size for clarity

    # First plot - Target not in DB
    axes[0].plot(complexity, baseline_score*np.ones(complexity.shape[0]), color=sns.color_palette("flare")[3])
    axes[0].scatter(complexity, mean_0, color=sns.color_palette("Reds")[3], label='Mean', s=50, edgecolor='black', zorder=5)
    axes[0].fill_between(complexity, mean_0 - std_0, mean_0 + std_0, alpha=0.3, color=sns.color_palette("Reds")[2], label='Std Dev', zorder=2)
    axes[0].set_title("Target not in DB", fontsize=16, fontweight='bold', color='darkred')
    
    # Second plot - Target in DB
    axes[1].plot(complexity, baseline_score*np.ones(complexity.shape[0]), color=sns.color_palette("flare")[3])
    axes[1].scatter(complexity, mean_1, color=sns.color_palette("Blues")[3], label='Mean', s=50, edgecolor='black', zorder=5)
    axes[1].fill_between(complexity, mean_1 - std_1, mean_1 + std_1, alpha=0.3, color=sns.color_palette("Blues")[2], label='Std Dev', zorder=2)
    axes[1].set_title("Target in DB", fontsize=16, fontweight='bold', color='darkblue')

    # Common settings for both subplots
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.set_xlabel("Complexity", fontsize=14, fontweight='normal', color='black')
        ax.set_ylabel("Probability", fontsize=14, fontweight='normal', color='black')
        ax.legend(loc='upper left', fontsize=12)

        # Add gridlines with less intensity for subtle effect
        ax.grid(True, linestyle='--', alpha=0.5)
    fig.suptitle(generator, color='black', fontsize=16)
    plt.tight_layout()
    # plt.show()
    return fig, axes

def single_plot(generator: str, complexity: np.array, mean: np.array, std: np.array, baseline_score):
    sns.set_style("whitegrid")  # Clean background with gridlines
    sns.set_palette("muted")  # Muted colors for a more refined look

    fig, axes = plt.subplots(1, 1, figsize=(14, 6))  # Larger plot size for clarity

    axes.scatter(complexity, mean, color=sns.color_palette("Reds")[3], label='Mean', s=50, edgecolor='black', zorder=5)
    axes.fill_between(complexity, mean - std, mean + std, alpha=0.3, color=sns.color_palette("Reds")[2], label='Std Dev', zorder=2)
    axes.set_title("Generator Results: " + generator, fontsize=16, fontweight='bold', color='black')
    
    axes.set_ylim(0, 1)
    axes.set_xlabel("Complexity", fontsize=14, fontweight='normal', color='black')
    axes.set_ylabel("Accuracy", fontsize=14, fontweight='normal', color='black')
    axes.legend(loc='upper left', fontsize=12)
    # Add gridlines with less intensity for subtle effect
    axes.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    # plt.show()
    return fig, axes


def plot_generators_ranks(models_metrics, tiers_order=None, store_results=True, result_plot_folder=None):
    if tiers_order is None:
        tiers_order = ["A", "B", "C", "D", "E", "F", "U"]

    df = pd.DataFrame(models_metrics)

    # Set tier column as ordered categorical
    df["Tier"] = pd.Categorical(df["Benchmark_rank"], categories=tiers_order, ordered=True)

    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.figure(figsize=(10, 6))
    # ax = plt.gca()
    ax.set_facecolor('white')  # Axes background



    # Add colored bands behind each tier row
    row_colors = ["#4292B9", "#70C4BC", "#8FD79F", "#B2E782", "#FFF54E", "#FED303", "#A9A9A9"]  # Light alternating shades
    for i, tier in enumerate(tiers_order):
        ax.axhspan(i - 0.5, i + 0.5, color=row_colors[i % len(row_colors)],alpha = 0.8, zorder=0)

    # Plot black points
    sns.stripplot(
        x="Speed",
        y="Tier",
        data=df,
        size=7,
        color="black",
        jitter=False,
        dodge=False,
        linewidth=0.5,
        zorder=1
    )

    # Annotations
    for i, row in df.iterrows():
        ax.text(
            row["Speed"],  # same x as dot
            tiers_order.index(row["Tier"]) + 0.2,  # shift down by 0.15 (use codes for exact numeric y)
            row["Model"],
            horizontalalignment='center',
            verticalalignment='top',  # so text is below the dot
            fontsize=12,
            color="black",
            weight='normal'
        )

    # Tier separator lines
    for tier_index in range(1, len(tiers_order)):
        plt.axhline(y=tier_index - 0.5, color='gray', linestyle='--', linewidth=1, zorder=3)

    # Axis styling
    plt.title("Model Rank List", color='black', fontsize=16)
    plt.xlabel("Generation time [μs / datapoint]", color='black')
    plt.ylabel("Rank", color='black')
    plt.xticks(color='black')
    plt.yticks(color='black')
    plt.xlim(-1500, df["Speed"].max() * 1.3)

    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()
    if store_results:
        # Save the figure
        fig.savefig(result_plot_folder + "generator_ranks.png", dpi=300, bbox_inches='tight')

def breaking_time_plot(models_metrics, store_results=True, result_plot_folder=None):

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Create a background with diagonal color lines
    grid_size = 300
    gx, gy = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))
    background_intensity = gx + gy  # higher values at top-right
    max_scale = 0
    x = []
    y = []
    for gen in range(len(models_metrics)):
        x.append(models_metrics[gen]["Speed"])
        y.append(models_metrics[gen]["Breaking_Time"])
    # Determine plot limits
    max_scale = max(max(x), max(y)) * 1.1
    ax.imshow(background_intensity, extent=[0, max_scale, 0, max_scale], origin='upper', cmap='viridis', alpha=0.5)

    # Scatter plot
    for gen in range(len(models_metrics)):
        if y[gen] > 0:
            ax.scatter(x[gen], y[gen], edgecolor='k', label=models_metrics[gen]["Model"])

    # x_ex = np.linspace(0.00001, max_scale)
    # ax.plot(x_ex, x_ex**2, color='yellow', linestyle='--', linewidth=2, label='y = x^2')
    # ax.plot(x_ex, x_ex, color='orange', linestyle='--', linewidth=2, label='y = x')
    # ax.plot(x_ex, x_ex**(1/2), color='red', linestyle='--', linewidth=2, label='y = x^(1/2)')

    # Labels and grid
    plt.xlabel("Generation time [μs / datapoint]", color='black')
    plt.ylabel("Breaking time [μs / datapoint]", color='black')
    ax.set_title('Generator breaking vs generation times')
    ax.set_ylim(0, max_scale)
    ax.legend()
    plt.grid(True)
    plt.show()
    if store_results:
        # Save the figure
        fig.savefig(result_plot_folder + "breaking_time.png", dpi=300, bbox_inches='tight')

def plot_radar_comparison(
    df,
    metrics=['Speed', 'AUC_MIA', 'Breaking_Time', 'DCR', 'Utility'],
    title="Generators Comparison",
    figsize=(8, 6),
    legend_loc="upper right",
    legend_bbox=(1.7, 1.1),
    fill_alpha=0.2,
    store_results=True,
    result_plot_folder=None
):
    """
    Draws a radar chart comparing models on the given metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain a 'Model' column and one column per metric.
    metrics : list of str, optional
        Column names to plot. Defaults to
        ['Speed', 'AUC_MIA', 'Breaking_Time', 'DCR', 'Utility'].
    title : str
        Plot title.
    figsize : tuple (width, height)
        Figure size in inches.
    legend_loc : str
        Location specifier for plt.legend().
    legend_bbox : tuple (x, y)
        bbox_to_anchor for legend to move it farther out.
    fill_alpha : float
        Alpha for the filled area under each line.
    """
    df = df.fillna(0)

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'polar': True})

    show_star_legend = False  #
    star_angle_idx = metrics.index("Breaking_Time")

    #make line at 1.0 thicker
    gridlines = ax.yaxis.get_gridlines()
    if len(gridlines) > 1:
        gridlines[-1].set_color('black')  # change color (e.g., gray)
        gridlines[-1].set_alpha(0.9)

    for i, line in enumerate(gridlines):
        if i != len(gridlines) - 1:
            line.set_linewidth(0.8)

    for _, row in df.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        model_label = row['Model']

        # Plot the radar line
        line, = ax.plot(angles, values, label=model_label)
        ax.fill(angles, values, alpha=fill_alpha, color=line.get_color())

        # Plot small dots at each metric point
        ax.scatter(angles[:-1], values[:-1], color=line.get_color(), s=30, zorder=3)

        # Add a star if Breaking_Time == 1.05
        if 'Breaking_Time' in metrics and np.isclose(row['Breaking_Time'], 1.05):
            idx = metrics.index('Breaking_Time')
            ax.plot(angles[idx], values[idx], marker='*', markersize=10, color='red', label="_nolegend_", zorder = 3.5)
            show_star_legend = True

    # set labels
    ax.tick_params(axis='x', pad=10)  # increase padding from the axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title(title,fontsize=16,pad=50)

    # push legend out
    ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox)

    # add separate star legend
    if show_star_legend:
        star_patch = Line2D([0], [0], marker='*', color='w', label='Model not broken',
                            markerfacecolor='red', markersize=12)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=[*handles, star_patch], labels=[*labels, 'Attack not succesfull in given complexity range'],
                  loc=legend_loc, bbox_to_anchor=legend_bbox)

    plt.tight_layout()
    plt.show()
    if store_results:
        fig.savefig(result_plot_folder + "spider_plot.png", dpi=300, bbox_inches='tight')

