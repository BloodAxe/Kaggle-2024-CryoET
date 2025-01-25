import numpy as np
from fire import Fire
from matplotlib import pyplot as plt

weights = {
    "apo-ferritin": 1,
    "beta-amylase": 0,
    "beta-galactosidase": 2,
    "ribosome": 1,
    "thyroglobulin": 2,
    "virus-like-particle": 1,
}

class_names = ["apo-ferritin", "beta-galactosidase", "ribosome", "thyroglobulin", "virus-like-particle"]


def main(*npz_files):
    per_class_scores = [np.load(f)["per_class_scores"] for f in npz_files]  # (n_files, n_thresholds, classes)

    per_class_scores = np.stack(per_class_scores).mean(axis=0)

    score_thresholds = np.load(npz_files[0])["score_thresholds"]

    best_index_per_class = np.argmax(per_class_scores, axis=0)  # [class]
    best_threshold_per_class = np.array([score_thresholds[i] for i in best_index_per_class])  # [class]
    best_score_per_class = np.array([per_class_scores[i, j] for j, i in enumerate(best_index_per_class)])  # [class]

    averaged_score = np.sum([weights[k] * best_score_per_class[i] for i, k in enumerate(class_names)]) / sum(weights.values())

    print("Best threshold per class: ", best_threshold_per_class)
    print("Average score:            ", averaged_score)

    plt.figure()
    for i, class_name in enumerate(class_names):
        plt.plot(score_thresholds, per_class_scores[:, i], label=class_name)
    plt.legend()
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Per class scores")
    plt.show()


if __name__ == "__main__":
    Fire(main)
