#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


from fvcore.common.config import CfgNode


def add_custom_config(_C):
    # Add your own customized configs.

    _C.DATA.COTTON = CfgNode()
    # Path to the metadata file for the cotton dataset.
    _C.DATA.COTTON.PATH_TO_METADATA = ""
    # Which cameras to use in SimCLR training.
    _C.DATA.COTTON.USE_CAMERAS = [0, 1, 2]

    # If true, will allow clips from different times in the same video to be
    # from different cameras.
    _C.DATA.COTTON.TEMPORAL_MULTI_CAMERA = False
    # Resize images to this input size before doing any other processing.
    _C.DATA.COTTON.PRE_RESIZE_IMAGE = [-1, -1]
    # Allow a certain probability of randomly reversing videos during training.
    _C.DATA.COTTON.RND_REVERSE = 0.0
    # Which labels to use for cotton finetuning.
    _C.DATA.COTTON.LABEL_TYPE = "plot_status"

    # Wandb configuration.
    _C.WANDB = CfgNode()
    _C.WANDB.ENTITY = ""
