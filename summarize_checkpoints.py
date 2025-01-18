import collections
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def infer_fold(checkpoint_name):
    for i in range(5):
        if f"fold_{i}" in checkpoint_name:
            return i

    raise ValueError(f"Could not infer fold from checkpoint name: {checkpoint_name}")


weights = {
    "apo-ferritin": 1,
    "beta-amylase": 0,
    "beta-galactosidase": 2,
    "ribosome": 1,
    "thyroglobulin": 2,
    "virus-like-particle": 1,
}


def main(*checkpoints):
    summary_table = collections.OrderedDict(
        {
            "fold": [],
            "checkpoint": [],
            "score": [],
            "AFRT": [],
            "BGT": [],
            "RBSM": [],
            "TRGLB": [],
            "VLP": [],
        }
    )

    per_class_scores = 0

    for checkpoint in checkpoints:

        # 250118_1253_dynunet_fold_4_6x96x128x128_rc_ic_s2_re_0.05_1060-score-0.8420-at-0.230-0.340-0.255-0.375-0.140.ckpt
        checkpoint_path = Path(checkpoint)
        checkpoint_name = checkpoint_path.stem
        parts = checkpoint_name.split("-")

        fold = infer_fold(checkpoint_name)

        checkpoint_state_dict = torch.load(checkpoint, weights_only=True, map_location="cpu")
        thresholds = checkpoint_state_dict["state_dict"]["thresholds"].numpy()
        per_class_scores = per_class_scores + checkpoint_state_dict["state_dict"]["per_class_scores"].numpy()

        summary_table["fold"].append(fold)
        summary_table["score"].append(parts[-7])
        summary_table["AFRT"].append(parts[-5])
        summary_table["BGT"].append(parts[-4])
        summary_table["RBSM"].append(parts[-3])
        summary_table["TRGLB"].append(parts[-2])
        summary_table["VLP"].append(parts[-1])
        summary_table["checkpoint"].append(checkpoint_name)

    # Find optimal threshold

    summary_table["fold"].append("mean")
    summary_table["checkpoint"].append("")
    summary_table["score"].append(sum(float(score) for score in summary_table["score"]) / len(summary_table["score"]))
    summary_table["AFRT"].append(sum(float(score) for score in summary_table["AFRT"]) / len(summary_table["AFRT"]))
    summary_table["BGT"].append(sum(float(score) for score in summary_table["BGT"]) / len(summary_table["BGT"]))
    summary_table["RBSM"].append(sum(float(score) for score in summary_table["RBSM"]) / len(summary_table["RBSM"]))
    summary_table["TRGLB"].append(sum(float(score) for score in summary_table["TRGLB"]) / len(summary_table["TRGLB"]))
    summary_table["VLP"].append(sum(float(score) for score in summary_table["VLP"]) / len(summary_table["VLP"]))

    # Find optimal threshold by averaging curves
    per_class_scores /= len(checkpoints)
    max_scores_index = np.argmax(per_class_scores, axis=0)
    optimal_thresholds = thresholds[max_scores_index]
    max_scores = np.max(per_class_scores, axis=0)

    summary_table["AFRT"].append(optimal_thresholds[0])
    summary_table["BGT"].append(optimal_thresholds[1])
    summary_table["RBSM"].append(optimal_thresholds[2])
    summary_table["TRGLB"].append(optimal_thresholds[3])
    summary_table["VLP"].append(optimal_thresholds[4])

    #
    score = np.sum([weights[k] * v for k, v in zip(weights.keys(), max_scores)]) / sum(weights.values())
    summary_table["score"].append(score)
    summary_table["fold"].append("mean (curve)")
    summary_table["checkpoint"].append("")

    summary_table = pd.DataFrame.from_dict(summary_table)
    print(summary_table[["fold", "score", "AFRT", "BGT", "RBSM", "TRGLB", "VLP", "checkpoint"]].to_markdown(index=False))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
