from typing import List

import torch
import fire
import os

from cryoet.ensembling import average_checkpoints

if __name__ == "__main__":
    from fire import Fire

    Fire(average_checkpoints)
