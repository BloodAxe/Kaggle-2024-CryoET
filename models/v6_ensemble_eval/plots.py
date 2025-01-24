from pathlib import Path

import pandas as pd
from fire import Fire

from cryoet.ensembling_eval import (
    plot_2d_score_centernet_single_label,
    plot_2d_score_spatial_depth,
    plot_boxplot_by_parameter,
    plot_max_score_vs_iou_threshold,
    plot_parallel_coordinates,
    plot_score_distribution,
    plot_boxplot_thresholds_vs_score,
    plot_boxplots_by_multiple_parameters,
    plot_scatter_by_multiple_parameters,
)


def main(*csv_summary_paths):
    for csv_summary_path in csv_summary_paths:
        csv_summary_path = Path(csv_summary_path)

        df = pd.read_csv(csv_summary_path)
        stem = str(csv_summary_path.stem).replace("_results", "")

        stem = str(csv_summary_path.parent / stem)

        plot_2d_score_centernet_single_label(df, f"{stem}_plot_2d_score_centernet_single_label.png")
        plot_2d_score_spatial_depth(df, f"{stem}_plot_2d_score_spatial_depth.png")
        plot_boxplot_by_parameter(df, "use_centernet_nms", f"{stem}_plot_boxplot_by_use_centernet_nms.png")
        plot_boxplot_by_parameter(df, "use_single_label_per_anchor", f"{stem}_plot_boxplot_by_use_single_label_per_anchor.png")
        plot_boxplot_by_parameter(df, "use_weighted_average", f"{stem}_plot_boxplot_by_use_weighted_average.png")
        plot_boxplot_by_parameter(df, "iou_threshold", f"{stem}_plot_boxplot_by_iou_threshold.png")
        plot_boxplot_by_parameter(df, "pre_nms_top_k", f"{stem}_plot_boxplot_by_pre_nms_top_k.png")
        plot_boxplot_by_parameter(df, "min_score_threshold", f"{stem}_plot_boxplot_by_min_score_threshold.png")
        plot_boxplot_by_parameter(df, "valid_spatial_tiles", f"{stem}_plot_boxplot_by_valid_spatial_tiles.png")

        plot_max_score_vs_iou_threshold(df, f"{stem}_plot_max_score_vs_iou_threshold.png")
        plot_parallel_coordinates(
            df,
            ["averaged_score"],
            "use_centernet_nms",
            f"{stem}_plot_parallel_coordinates.png",
        )
        plot_score_distribution(df, f"{stem}_plot_score_distribution.png")

        plot_boxplot_thresholds_vs_score(
            df,
            threshold_column="apo-ferritin_threshold",
            group_columns=[
                "use_centernet_nms",
                "use_single_label_per_anchor",
                "use_weighted_average",
                "iou_threshold",
                "pre_nms_top_k",
                "min_score_threshold",
                "valid_spatial_tiles",
            ],
            output_filename=f"{stem}_plot_boxplot_thresholds_vs_score.png",
        )

        plot_boxplots_by_multiple_parameters(
            df,
            parameters=[
                "apo-ferritin_threshold",
                "beta-galactosidase_threshold",
                "ribosome_threshold",
                "thyroglobulin_threshold",
                "virus-like-particle_threshold",
            ],
            output_filename=f"{stem}_thresholds_boxplots.png",
        )

        plot_scatter_by_multiple_parameters(
            df,
            parameters=[
                "apo-ferritin_threshold",
                "beta-galactosidase_threshold",
                "ribosome_threshold",
                "thyroglobulin_threshold",
                "virus-like-particle_threshold",
            ],
            output_filename=f"{stem}_thresholds_scatterplots.png",
        )


if __name__ == "__main__":
    Fire(main)
