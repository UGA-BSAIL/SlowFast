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

    # Wandb configuration.
    _C.WANDB = CfgNode()
    _C.WANDB.ENTITY = ""
