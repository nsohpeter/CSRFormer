# Copyright (c) Facebook, Inc. and its affiliates.
from . import datasets

# from . import register_custom_datasets

try:
    from ..datasets.register_custom_datasets import *

    print("[mask2former.datasets] Custom datasets imported from custom-datasets")
except ImportError as e:
    print(f"[mask2former.datasets] Note: Could not import custom datasets: {e}")
