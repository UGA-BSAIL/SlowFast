from pathlib import Path
import random
from typing import Tuple, List, Any

import torch
import torchvision.transforms
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import functional as F

import pandas as pd

from yacs.config import CfgNode

import slowfast.utils.logging as logging
from .build import DATASET_REGISTRY
from . import utils
from .schemas import MarsMetadata
from .random_erasing import RandomErasing

logger = logging.get_logger(__name__)


class CottonClips(Dataset):
    """
    Dataset that reads single clips from the cotton dataset.
    """

    def __init__(
        self,
        config: CfgNode,
        _mode: str,
    ):
        """
        Args:
            config: The configuration.
            _mode: Ignored for now, since this dataset is currently used only
                for self-supervised pre-training.

        """
        # Load the metadata.
        mars_metadata = pd.read_feather(config.DATA.PATH_TO_METADATA)
        self.__metadata = self.__filter_zero_length_clips(mars_metadata).set_index(
            MarsMetadata.CLIP.value
        )

        self.__image_folder = Path(config.DATA.PATH_TO_DATA_DIR)
        logger.info(f"Loading dataset images from {self.__image_folder}.")
        self._config = config

        # Find the clips from the metadata.
        self.__clip_ids = self.__metadata.index.unique()

    @staticmethod
    def __filter_zero_length_clips(metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Filters clips from the dataset that have no frames.

        Args:
            metadata: The metadata for the dataset.

        Returns:
            The filtered metadata.

        """
        # Compute lengths for each clip. Limit to a single camera, so we get
        # correct lengths. (All cameras should be synchronized and thus have
        # the same number of frames.)
        single_camera = metadata[metadata[MarsMetadata.CAMERA.value] == 0]
        single_camera = single_camera[
            [MarsMetadata.CLIP.value, MarsMetadata.TIMESTAMP.value]
        ]
        by_clip = single_camera.groupby(MarsMetadata.CLIP.value)
        clip_lengths = by_clip.max() - by_clip.min()

        # Throw out anything that's too short for proper example selection.
        short_clips = clip_lengths[clip_lengths == 0]
        short_clips.dropna(inplace=True)
        metadata = metadata[~metadata[MarsMetadata.CLIP.value].isin(short_clips.index)]
        logger.debug(
            f"Have {len(metadata[MarsMetadata.CLIP.value].unique())} clips of sufficient length."
        )

        return metadata

    def __len__(self) -> int:
        return len(self.__clip_ids)

    def __sample_video_frames(self, frame_files: List[Path]) -> List[Path]:
        """
        Samples frames from a video.

        Args:
            frame_files: The list of image files that make up the video.

        Returns:
            The list of files to sample.

        """
        num_frames = self._config.DATA.NUM_FRAMES
        sampling_rate = utils.get_random_sampling_rate(
            self._config.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self._config.DATA.SAMPLING_RATE,
        )
        video_length = len(frame_files)

        clip_length = (num_frames - 1) * sampling_rate + 1
        if clip_length > video_length:
            start = random.randint(video_length - clip_length, 0)
        else:
            start = random.randint(0, video_length - clip_length)

        seq = [
            max(min(start + i * sampling_rate, video_length - 1), 0)
            for i in range(num_frames)
        ]

        return [frame_files[i] for i in seq]

    def __getitem__(self, item: int) -> Tensor:
        """
        Args:
            item: The index for the item to get.

        Returns:
            frames: the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`. This will
                be provided as a list, with one tensor for each pathway.

        """
        # Figure out which file we should read.
        clip_id = self.__clip_ids[item]
        clip_files = self.__metadata.loc[
            clip_id, [MarsMetadata.CAMERA.value, MarsMetadata.FILE_ID.value]
        ]
        # Select a single camera.
        all_cameras = clip_files[MarsMetadata.CAMERA.value].unique()
        clip_files = clip_files[
            clip_files[MarsMetadata.CAMERA.value] == random.choice(all_cameras)
        ][MarsMetadata.FILE_ID.value]
        clip_paths = [self.__image_folder / f"{p}.jpg" for p in clip_files]
        sample_paths = self.__sample_video_frames(clip_paths)

        return torch.stack([read_image(p.as_posix()) for p in sample_paths])


@DATASET_REGISTRY.register()
class Cotton(CottonClips):
    """
    A dataset that reads paired and augmented clips from the cotton dataset,
    for use with contrastive learning.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # Annoyingly, this is required by some downstream code. We don't have
        # any labels so we set them all to zero.
        self._labels = [0] * len(self)

        self.__ssl_aug_transform = torchvision.transforms.RandAugment()

    def __spatially_augment_clip(self, clip: Tensor) -> Tensor:
        """
        Performs spatial augmentation on an input clip.

        Args:
            clip: The clip to augment.

        Returns:
            The augmented clip.

        """
        augmented_clip = clip
        if self._config.DATA.SSL_COLOR_JITTER:
            augmented_clip = self.__ssl_aug_transform(augmented_clip)

        # Perform color normalization.
        augmented_clip = F.normalize(
            augmented_clip.float(), self._config.DATA.MEAN, self._config.DATA.STD
        )
        augmented_clip = augmented_clip.permute(1, 0, 2, 3)

        scl, asp = (
            self._config.DATA.TRAIN_JITTER_SCALES_RELATIVE,
            self._config.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
        )
        relative_scales = None if len(scl) == 0 else scl
        relative_aspect = None if len(asp) == 0 else asp
        min_scale = self._config.DATA.TRAIN_JITTER_SCALES[0]
        max_scale = self._config.DATA.TRAIN_JITTER_SCALES[1]
        crop_size = self._config.DATA.TRAIN_CROP_SIZE
        if self._config.MULTIGRID.DEFAULT_S > 0:
            # Decreasing the scale is equivalent to using a larger "span"
            # in a sampling grid.
            min_scale = int(
                round(float(min_scale) * crop_size / self._config.MULTIGRID.DEFAULT_S)
            )
        augmented_clip = utils.spatial_sampling(
            augmented_clip,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self._config.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self._config.DATA.INV_UNIFORM_SAMPLE,
            aspect_ratio=relative_aspect,
            scale=relative_scales,
            motion_shift=self._config.DATA.TRAIN_JITTER_MOTION_SHIFT,
        )

        if self._config.AUG.ENABLE and self._config.AUG.RE_PROB > 0:
            erase_transform = RandomErasing(
                self._config.AUG.RE_PROB,
                mode=self._config.AUG.RE_MODE,
                max_count=self._config.AUG.RE_COUNT,
                num_splits=self._config.AUG.RE_COUNT,
            )
            augmented_clip = erase_transform(
                augmented_clip.permute(1, 0, 2, 3)
            ).permute(1, 0, 2, 3)

        return augmented_clip

    def __getitem__(
        self, item: int
    ) -> Tuple[List[List[Tensor]] | List[Tensor], Tensor, int, Tensor, dict]:
        """
        Gets an item from the dataset at a particular index.

        Args:
            item: The index of the item to get.

        Returns:
             frames: the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`. This will
                be provided as a list, with one tensor for each pathway.
            label: The labels for the video.
            index: The index of the video.
            time_index: Unused, set to zero.
            extra_data: Unused, set to empty dict.
        """
        # Sample the correct number of clips from the input video.
        num_temporal_samples = self._config.DATA.TRAIN_CROP_NUM_TEMPORAL
        # Also augment each clip multiple times.
        num_augmentations = (
            self._config.DATA.TRAIN_CROP_NUM_SPATIAL * self._config.AUG.NUM_SAMPLE
        )
        sampled_clips = []

        for temporal_i in range(num_temporal_samples):
            # Sample temporally.
            clip = super().__getitem__(item)

            for spatial_i in range(num_augmentations):
                # Do augmentation.
                augmented_clip = self.__spatially_augment_clip(clip)
                augmented_clip = utils.pack_pathway_output(self._config, augmented_clip)

                sampled_clips.append(augmented_clip)

        labels = torch.zeros(len(sampled_clips))
        if len(sampled_clips) == 1:
            # For validation, we don't provide the input frames as a list.
            sampled_clips = sampled_clips[0]
        return sampled_clips, labels, item, torch.zeros((1, 1)), {}
