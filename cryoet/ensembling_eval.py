from typing import List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_max_score_vs_iou_threshold(df: pd.DataFrame, output_filename: str):
    """
    1) Plot the maximum overall score on Y-axis vs. the IOU threshold on X-axis.

    :param df: Input dataframe containing at least the columns:
               ['iou_threshold', 'averaged_score', ...]
    :param output_filename: Path (with filename) to save the resulting plot.
    """
    # Group by IOU threshold and find the max averaged_score for each threshold
    grouped = df.groupby("iou_threshold")["averaged_score"].max().reset_index()

    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=grouped, x="iou_threshold", y="averaged_score", marker="o")
    plt.title("Max Overall Score vs. IOU Threshold")
    plt.xlabel("IOU Threshold")
    plt.ylabel("Max Overall Score")
    plt.grid(True)

    # Save the figure
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def plot_2d_score_spatial_depth(df: pd.DataFrame, output_filename: str):
    """
    2) 2D plot with valid_spatial_tiles on X and valid_depth_tiles on Y.
       Each cell value is the maximum score for that combination.

    :param df: Input dataframe containing at least the columns:
               ['valid_spatial_tiles', 'valid_depth_tiles', 'averaged_score', ...]
    :param output_filename: Path (with filename) to save the resulting heatmap plot.
    """
    # Compute the max score for each combination of (valid_spatial_tiles, valid_depth_tiles)
    pivot_df = df.groupby(["valid_spatial_tiles", "valid_depth_tiles"])["averaged_score"].max().reset_index()

    # Create a pivot table suitable for a heatmap
    pivot_table = pivot_df.pivot(index="valid_depth_tiles", columns="valid_spatial_tiles", values="averaged_score")

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Max Score by (valid_spatial_tiles, valid_depth_tiles)")
    plt.xlabel("valid_spatial_tiles")
    plt.ylabel("valid_depth_tiles")

    # Save the figure
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def plot_2d_score_centernet_single_label(df: pd.DataFrame, output_filename: str):
    """
    3) 2D plot with use_centernet_nms on X and use_single_label_per_anchor on Y.
       Each cell value is the maximum score for that combination.

    :param df: Input dataframe containing at least the columns:
               ['use_centernet_nms', 'use_single_label_per_anchor', 'averaged_score', ...]
    :param output_filename: Path (with filename) to save the resulting heatmap plot.
    """
    # Compute the max score for each combination of (use_centernet_nms, use_single_label_per_anchor)
    pivot_df = df.groupby(["use_centernet_nms", "use_single_label_per_anchor"])["averaged_score"].max().reset_index()

    # Convert boolean columns to string if needed, to avoid heatmap or pivot complications
    # pivot_df["use_centernet_nms"] = pivot_df["use_centernet_nms"].astype(str)
    # pivot_df["use_single_label_per_anchor"] = pivot_df["use_single_label_per_anchor"].astype(str)

    # Create a pivot table suitable for a heatmap
    pivot_table = pivot_df.pivot(index="use_single_label_per_anchor", columns="use_centernet_nms", values="averaged_score")

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Max Score by (use_centernet_nms, use_single_label_per_anchor)")
    plt.xlabel("use_centernet_nms")
    plt.ylabel("use_single_label_per_anchor")

    # Save the figure
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def plot_score_distribution(df: pd.DataFrame, output_filename: str):
    """
    Plots the distribution (histogram + KDE) of the 'averaged_score' column.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    sns.histplot(df["averaged_score"], kde=True, bins=20)
    plt.title("Distribution of Averaged Scores")
    plt.xlabel("Averaged Score")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def plot_boxplot_by_parameter(df: pd.DataFrame, parameter: str, output_filename: str):
    """
    Creates a boxplot (or violin plot) of 'averaged_score' grouped by a specified boolean or categorical parameter.
    E.g., parameter='use_weighted_average' or 'use_centernet_nms'.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=parameter, y="averaged_score", data=df)
    # If your parameter is boolean, you can convert it to string:
    # df[parameter] = df[parameter].astype(str)

    plt.title(f"Averaged Score Distribution by {parameter}")
    plt.xlabel(parameter)
    plt.ylabel("Averaged Score")

    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def plot_parallel_coordinates(df: pd.DataFrame, numeric_params: list, class_param: str, output_filename: str):
    """
    Creates a parallel coordinates plot for a set of numeric hyperparameters
    and color-codes by a selected categorical/boolean parameter (e.g., 'use_centernet_nms').

    numeric_params might include 'iou_threshold', 'pre_nms_top_k', 'min_score_threshold', 'averaged_score', etc.
    class_param is a column you want to use for coloring lines (should be categorical or discrete).
    """
    import matplotlib.pyplot as plt
    from pandas.plotting import parallel_coordinates

    # If class_param is boolean, convert to string for better legend:
    if df[class_param].dtype == bool:
        df[class_param] = df[class_param].astype(str)

    # Create a copy with only the columns needed
    plot_df = df[numeric_params + [class_param]].copy()

    plt.figure(figsize=(10, 6))
    parallel_coordinates(plot_df, class_param, colormap=plt.cm.viridis, alpha=0.5)
    plt.title("Parallel Coordinates Plot")
    plt.ylabel("Parameter values or Score")

    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def plot_boxplot_thresholds_vs_score(df: pd.DataFrame, threshold_column: str, group_columns: list, output_filename: str):
    """
    Produces a box plot that shows how the maximum averaged_score
    (per unique hyperparameter combination) varies across different thresholds.

    :param df:              The input DataFrame.
    :param threshold_column: The name of the column representing the threshold (e.g., iou_threshold, min_score_threshold).
    :param group_columns:   A list of column names that define a unique hyperparameter group
                            (e.g., ["valid_depth_tiles", "valid_spatial_tiles", "use_weighted_average", ...]).
    :param output_filename: The filename/path to save the resulting plot.
    """
    # 1. Group by the threshold column + the specified hyperparameter group columns
    # 2. Compute the max of averaged_score for each group
    grouped_df = df.groupby(group_columns + [threshold_column], as_index=False)["averaged_score"].max()

    # 3. Create the box plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=grouped_df, x=threshold_column, y="averaged_score")
    plt.title(f"Max Score per Group vs. {threshold_column}")
    plt.xlabel(threshold_column)
    plt.ylabel("Max Averaged Score")

    # 4. Save and close the figure
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def plot_boxplots_by_multiple_parameters(df: pd.DataFrame, parameters: List[str], output_filename: str):
    """
    Creates multiple box plots in a single figure, one for each parameter in 'parameters',
    showing the distribution of 'averaged_score' across that parameter's unique values.

    :param df:              The input DataFrame.
    :param parameters:      List of column names (parameters) to create boxplots for.
    :param output_filename: Path/filename to save the resulting figure.
    """
    num_params = len(parameters)

    # Decide on a figure layout
    # - For example, if you have many parameters, consider multiple rows.
    #   Here, we'll do a single row for simplicity.
    fig, axes = plt.subplots(1, num_params, figsize=(6 * num_params, 6), squeeze=False)

    # 'axes' is 2D even if we specify 1 row. Let's flatten it for easy indexing.
    axes = axes.flatten()

    for i, param in enumerate(parameters):
        # If parameter is boolean, optionally convert to string for better plotting
        if df[param].dtype == bool:
            df[param] = df[param].astype(str)

        sns.boxplot(x=param, y="averaged_score", data=df, ax=axes[i])
        axes[i].set_title(f"Averaged Score by {param}")
        axes[i].set_xlabel(param)
        axes[i].set_ylabel("Averaged Score")

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


def plot_scatter_by_multiple_parameters(df: pd.DataFrame, parameters: List[str], output_filename: str):
    """
    Creates multiple scatter plots (subplots) in a single figure—one per parameter—
    showing 'averaged_score' (Y-axis) vs. the given parameter (X-axis).

    :param df:              The input DataFrame containing the threshold columns and 'averaged_score'.
    :param parameters:      List of threshold column names to create scatter plots for.
    :param output_filename: Filename/path to save the resulting figure.
    """

    num_params = len(parameters)

    # Create subplots in one row. Adjust 'figsize' as needed.
    fig, axes = plt.subplots(nrows=1, ncols=num_params, figsize=(6 * num_params, 5), squeeze=False)
    axes = axes.flatten()  # Flatten from 2D array to 1D for easy iteration

    for i, param in enumerate(parameters):
        ax = axes[i]

        # Plot scatter of param (X) vs. averaged_score (Y)
        ax.scatter(df[param], df["averaged_score"], alpha=0.5)

        # Set labels and title
        ax.set_title(f"Averaged Score vs. {param}")
        ax.set_xlabel(param)
        ax.set_ylabel("averaged_score")

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()
