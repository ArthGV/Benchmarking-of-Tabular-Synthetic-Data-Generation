import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(complexity: np.array, mean: np.array, var: np.array = None):
    plt.scatter(complexity, mean, c='red')
    if var is not None:
        lower_bound = mean - var
        upper_bound = mean + var
        plt.scatter(complexity, mean, c='red')
        plt.fill_between(complexity, lower_bound, upper_bound, alpha=.3, color='red')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def double_plot(complexity: np.array, mean_0: np.array, std_0: np.array, mean_1: np.array, std_1: np.array, baseline_score):
    sns.set_style("whitegrid")  # Clean background with gridlines
    sns.set_palette("muted")  # Muted colors for a more refined look

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Larger plot size for clarity

    # First plot - Target not in DB
    axes[0].plot(complexity, baseline_score*np.ones(complexity.shape[0]), color=sns.color_palette("flare")[3])
    axes[0].scatter(complexity, mean_0, color=sns.color_palette("Reds")[3], label='Mean', s=50, edgecolor='black', zorder=5)
    axes[0].fill_between(complexity, mean_0 - std_0, mean_0 + std_0, alpha=0.3, color=sns.color_palette("Reds")[2], label='Std Dev', zorder=2)
    axes[0].set_title("Target not in DB", fontsize=16, fontweight='bold', color='darkred')
    
    # Second plot - Target in DB
    axes[1].plot(complexity, 1 - baseline_score*np.ones(complexity.shape[0]), color=sns.color_palette("flare")[3])
    axes[1].scatter(complexity, mean_1, color=sns.color_palette("Blues")[3], label='Mean', s=50, edgecolor='black', zorder=5)
    axes[1].fill_between(complexity, mean_1 - std_1, mean_1 + std_1, alpha=0.3, color=sns.color_palette("Blues")[2], label='Std Dev', zorder=2)
    axes[1].set_title("Target in DB", fontsize=16, fontweight='bold', color='darkblue')

    # Common settings for both subplots
    for ax in axes:
        ax.set_ylim(0, 1)
        ax.set_xlabel("Complexity", fontsize=14, fontweight='bold', color='gray')
        ax.set_ylabel("Probability", fontsize=14, fontweight='bold', color='gray')
        ax.legend(title='Legend', loc='upper left', fontsize=12)

        # Add gridlines with less intensity for subtle effect
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def single_plot(complexity: np.array, mean: np.array, std: np.array, baseline_score):
    sns.set_style("whitegrid")  # Clean background with gridlines
    sns.set_palette("muted")  # Muted colors for a more refined look

    fig, axes = plt.subplots(1, 1, figsize=(14, 6))  # Larger plot size for clarity

    axes.plot(complexity, 1 - baseline_score*np.ones(complexity.shape[0]), color=sns.color_palette("flare")[5])
    axes.scatter(complexity, mean, color=sns.color_palette("Reds")[3], label='Mean', s=50, edgecolor='black', zorder=5)
    axes.fill_between(complexity, mean - std, mean + std, alpha=0.3, color=sns.color_palette("Reds")[2], label='Std Dev', zorder=2)
    axes.set_title("Generator Results", fontsize=16, fontweight='bold', color='darkred')
    

    axes.set_ylim(0, 1)
    axes.set_xlabel("Complexity", fontsize=14, fontweight='bold', color='gray')
    axes.set_ylabel("Probability", fontsize=14, fontweight='bold', color='gray')
    axes.legend(title='Legend', loc='upper left', fontsize=12)
    # Add gridlines with less intensity for subtle effect
    axes.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_generators_ranks(models, tiers_order=None, title="Model Rank List"):
    if tiers_order is None:
        tiers_order = ["A", "B", "C", "D", "E", "F", "U"]

    df = pd.DataFrame(models)

    # Set tier column as ordered categorical
    df["Tier"] = pd.Categorical(df["Final_Score"], categories=tiers_order, ordered=True)

    # Setup figure
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.set_facecolor('white')  # Axes background

    # Add colored bands behind each tier row
    row_colors = ["#4292B9", "#70C4BC", "#8FD79F", "#B2E782", "#FFF54E", "#FED303", "#A9A9A9"]  # Light alternating shades
    for i, tier in enumerate(tiers_order):
        ax.axhspan(i - 0.5, i + 0.5, color=row_colors[i % len(row_colors)], zorder=0)

    # Plot white points
    sns.stripplot(
        x="Speed",
        y="Tier",
        data=df,
        size=30,
        color="white",       # Points are white
        jitter=False,
        dodge=False,
        edgecolor="black",   # Optional: black border for visibility
        linewidth=0.5,
        zorder=1
    )

    # Annotations
    for i in range(len(df)):
        plt.text(
            df["Speed"].iloc[i],
            df["Tier"].iloc[i],
            df["Model"].iloc[i],
            ha='center',
            va='center',
            size='small',
            color='black',
            weight='semibold',
            zorder=2
        )

    # Tier separator lines
    for tier_index in range(1, len(tiers_order)):
        plt.axhline(y=tier_index - 0.5, color='gray', linestyle='--', linewidth=1, zorder=3)

    # Axis styling
    plt.title(title, color='black', fontsize=16)
    plt.xlabel("Generation time [datapoint / sec]", color='black')
    plt.ylabel("Rank", color='black')
    plt.xticks(color='black')
    plt.yticks(color='black')

    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.show()