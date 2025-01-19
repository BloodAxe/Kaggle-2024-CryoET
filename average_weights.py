from typing import List

import torch
import fire
import os


def average_checkpoints(*ckpt_paths: List[str], output_path="averaged_model.pt"):
    """
    Averages the 'state_dict' weights of multiple PyTorch checkpoints.

    Args:
        ckpt_paths (list of str): List of paths to checkpoint files.
        output_path (str): Path to save the averaged checkpoint.
    """

    if not ckpt_paths:
        raise ValueError("No checkpoint paths were provided.")

    # Load state dicts from each checkpoint
    state_dicts = []
    for path in ckpt_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint file '{path}' does not exist.")
        checkpoint = torch.load(path, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise KeyError(f"'state_dict' not found in checkpoint '{path}'.")
        state_dicts.append(checkpoint["state_dict"])

    # Get all keys from the first checkpoint
    keys = state_dicts[0].keys()
    final_state_dict = {}

    for key in keys:
        # Collect the values (tensors) for this key from all checkpoints
        values = [sd[key] for sd in state_dicts]

        # Check the dtype of the first value (assuming all dtypes match)
        first_val = values[0]

        if not all(v.shape == first_val.shape for v in values):
            raise ValueError(f"Tensor shapes for key '{key}' are not consistent across checkpoints.")

        if first_val.dtype == torch.bool:
            # For bool, ensure all are identical
            for val in values[1:]:
                if not torch.equal(val, first_val):
                    raise ValueError(f"Boolean values for key '{key}' differ between checkpoints.")
            final_state_dict[key] = first_val  # Use the first if all identical

        elif torch.is_floating_point(first_val):
            # Average float values
            stacked = torch.stack(values, dim=0)
            averaged = stacked.mean(dim=0)
            final_state_dict[key] = averaged

        elif torch.is_integral(first_val):
            # Average integer values (using integer division)
            stacked = torch.stack(values, dim=0)
            summed = stacked.sum(dim=0)
            averaged = summed // len(values)
            final_state_dict[key] = averaged

        else:
            # If you have other special dtypes to handle, add logic here
            # or simply copy the first value if that is your intended behavior.
            raise TypeError(f"Unsupported dtype '{first_val.dtype}' encountered for key '{key}'.")

    # Save the final averaged state_dict
    torch.save({"state_dict": final_state_dict}, output_path)
    print(f"Averaged checkpoint saved to: {output_path}")


if __name__ == "__main__":
    # Expose the function via Fire
    fire.Fire(average_checkpoints)
