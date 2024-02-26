#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Add custom configs and default values"""


from fvcore.common.config import CfgNode


def add_custom_config(_C):
    # Add your own customized configs.

    # Path to the metadata file for the cotton dataset.
    _C.DATA.PATH_TO_METADATA = ""

    # Wandb configuration.
    _C.WANDB = CfgNode()
    _C.WANDB.ENTITY = ""
