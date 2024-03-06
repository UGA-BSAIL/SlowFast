from pathlib import Path
import random
from typing import Tuple, List, Any, Iterable
from dataclasses import dataclass

import torch
import torchvision.transforms
from torchvision.transforms import InterpolationMode
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_image, read_video
from torchvision.transforms import functional as F

import pandas as pd

from yacs.config import CfgNode

from yaml import CLoader
import yaml

import slowfast.utils.logging as logging
from .build import DATASET_REGISTRY
from . import utils
from .schemas import MarsMetadata
from .random_erasing import RandomErasing

logger = logging.get_logger(__name__)


def _spatial_sampling(clip: Tensor, *, config: CfgNode) -> Tensor:
    """
    Performs spatial sampling on a clip.

    Args:
        clip: The clip to sample.
        config: The configuration to use.

    Returns:
        The augmented clip.

    """
    scl, asp = (
        config.DATA.TRAIN_JITTER_SCALES_RELATIVE,
        config.DATA.TRAIN_JITTER_ASPECT_RELATIVE,
    )
    relative_scales = None if len(scl) == 0 else scl
    relative_aspect = None if len(asp) == 0 else asp
    min_scale = config.DATA.TRAIN_JITTER_SCALES[0]
    max_scale = config.DATA.TRAIN_JITTER_SCALES[1]
    crop_size = config.DATA.TRAIN_CROP_SIZE
    if config.MULTIGRID.DEFAULT_S > 0:
        # Decreasing the scale is equivalent to using a larger "span"
        # in a sampling grid.
        min_scale = int(
            round(float(min_scale) * crop_size / config.MULTIGRID.DEFAULT_S)
        )
    return utils.spatial_sampling(
        clip,
        min_scale=min_scale,
        max_scale=max_scale,
        crop_size=crop_size,
        random_horizontal_flip=config.DATA.RANDOM_FLIP,
        inverse_uniform_sampling=config.DATA.INV_UNIFORM_SAMPLE,
        aspect_ratio=relative_aspect,
        scale=relative_scales,
        motion_shift=config.DATA.TRAIN_JITTER_MOTION_SHIFT,
    )


class CottonClips(Dataset):
    """
    Dataset that reads single clips from the cotton dataset.
    """

    def __init__(self, config: CfgNode, *, camera: int):
        """
        Args:
            config: The configuration.
            camera: Which camera we should source data from.

        """
        # Load the metadata.
        mars_metadata = pd.read_feather(config.DATA.COTTON.PATH_TO_METADATA)
        self.__metadata = self.__filter_zero_length_clips(mars_metadata).set_index(
            MarsMetadata.CLIP.value
        )
        # Filter to just this camera.
        self.__metadata = self.__metadata[
            self.__metadata[MarsMetadata.CAMERA.value] == camera
        ]

        self.__image_folder = Path(config.DATA.PATH_TO_DATA_DIR)
        logger.info(f"Loading dataset images from {self.__image_folder}.")
        self.__config = config

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
        num_frames = self.__config.DATA.NUM_FRAMES
        sampling_rate = utils.get_random_sampling_rate(
            self.__config.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.__config.DATA.SAMPLING_RATE,
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
                is `num frames` x `channel` x `height` x `width`.

        """
        # Figure out which file we should read.
        clip_id = self.__clip_ids[item]
        clip_files = self.__metadata.loc[clip_id, MarsMetadata.FILE_ID.value]
        clip_paths = [self.__image_folder / f"{p}.jpg" for p in clip_files]
        sample_paths = self.__sample_video_frames(clip_paths)

        clip = torch.stack([read_image(p.as_posix()) for p in sample_paths])

        if self.__config.DATA.COTTON.PRE_RESIZE_IMAGE[0] > 0:
            # Resize images before processing.
            clip = F.resize(
                clip,
                self.__config.DATA.COTTON.PRE_RESIZE_IMAGE,
                interpolation=InterpolationMode.NEAREST,
            )
        return clip


@DATASET_REGISTRY.register()
class Cotton:
    """
    A dataset that reads paired and augmented clips from the cotton dataset,
    for use with contrastive learning.
    """

    def __init__(self, config: CfgNode, _mode: str):
        """
        Args:
            config: The configuration to use.
            _mode: The mode. Will be ignored for this dataset.

        """
        self.__config = config

        # Create clip datasets for each camera.
        self.__clip_datasets = []
        for camera in config.DATA.COTTON.USE_CAMERAS:
            self.__clip_datasets.append(CottonClips(config, camera=camera))

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
        if self.__config.DATA.SSL_COLOR_JITTER:
            augmented_clip = self.__ssl_aug_transform(augmented_clip)

        # Perform color normalization.
        augmented_clip = F.normalize(
            augmented_clip.float(), self.__config.DATA.MEAN, self.__config.DATA.STD
        )
        augmented_clip = augmented_clip.permute(1, 0, 2, 3)

        augmented_clip = _spatial_sampling(augmented_clip, config=self.__config)

        if self.__config.AUG.ENABLE and self.__config.AUG.RE_PROB > 0:
            erase_transform = RandomErasing(
                self.__config.AUG.RE_PROB,
                mode=self.__config.AUG.RE_MODE,
                max_count=self.__config.AUG.RE_COUNT,
                num_splits=self.__config.AUG.RE_COUNT,
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
        num_temporal_samples = self.__config.DATA.TRAIN_CROP_NUM_TEMPORAL
        # Also augment each clip multiple times.
        num_augmentations = (
            self.__config.DATA.TRAIN_CROP_NUM_SPATIAL * self.__config.AUG.NUM_SAMPLE
        )
        sampled_clips = []

        camera_dataset = random.choice(self.__clip_datasets)
        for temporal_i in range(num_temporal_samples):
            # Sample temporally.
            if self.__config.DATA.COTTON.TEMPORAL_MULTI_CAMERA:
                # Allow each clip to be from a different camera.
                camera_dataset = random.choice(self.__clip_datasets)
            clip = camera_dataset[item]

            if random.random() < self.__config.DATA.COTTON.RND_REVERSE:
                # Reverse the clip.
                clip = torch.flip(clip, [0])

            for spatial_i in range(num_augmentations):
                # Do augmentation.
                augmented_clip = self.__spatially_augment_clip(clip)
                augmented_clip = utils.pack_pathway_output(
                    self.__config, augmented_clip
                )

                sampled_clips.append(augmented_clip)

        labels = torch.zeros(len(sampled_clips))

        # Make the order indefinite here to spice up contrastive training.
        random.shuffle(sampled_clips)

        if len(sampled_clips) == 1:
            # For validation, we don't provide the input frames as a list.
            sampled_clips = sampled_clips[0]
        return sampled_clips, labels, item, torch.zeros((1, 1)), {}

    def __len__(self) -> int:
        return len(self.__clip_datasets[0])


@DATASET_REGISTRY.register()
class CottonLabeled:
    """
    A dataset of labeled clips drawn from the larger MARS dataset.
    """

    @dataclass
    class Label:
        """
        Represents a labeled clip.

        Attributes:
            num_flowers: The number of flowers in the clip.
            row_status: The row status label for the clip.

        """

        num_flowers: int
        row_status: int

    def __init__(self, config: CfgNode, mode: str):
        """
        Args:
            config: The configuration to use.
            mode: The mode, either "train", "test", or "val".

        """
        assert mode in {"train", "test", "val"}, f"Invalid mode '{mode}'."
        self.__mode = mode

        self.__config = config
        video_dir = Path(config.DATA.PATH_TO_DATA_DIR) / mode
        self.__videos = sorted(list(video_dir.glob("*.mp4")))
        self.__labels = self.__load_labels(self.__videos)

    def __load_labels(self, video_paths: Iterable[Path]) -> List[Label]:
        """
        Loads all the labels into memory.

        Args:
            video_paths: The paths to the videos.

        Returns:
            A list of the labels corresponding to the video paths.

        """
        labels = []
        for video_path in video_paths:
            # Read the label file.
            label_path = video_path.parent / f"{video_path.stem}.yml"
            label_data = yaml.load(label_path.open(), Loader=CLoader)

            labels.append(
                self.Label(
                    num_flowers=int(label_data["num_flowers"]),
                    row_status=int(label_data["row_status"]),
                )
            )

        return labels

    def __len__(self) -> int:
        return len(self.__videos)

    @property
    def num_videos(self) -> int:
        return len(self)

    def __getitem__(self, item: int) -> Tuple[List[Tensor], Tensor, int, Tensor, dict]:
        """
        Gets the specified video.

        Args:
            item: The index of the video to read.

        Returns:
            - The video frames, in the form `[channels, num_frames, height,
                width]`
            - The row status and flower count labels.
            - The input index.
            - The time index, currently unused.
            - Additional metadata, currently unused.

        """
        # Load the video.
        video_path = self.__videos[item]
        video, _, _ = read_video(
            video_path.as_posix(), pts_unit="sec", output_format="TCHW"
        )

        # Perform color normalization.
        video = F.normalize(
            video.float(), self.__config.DATA.MEAN, self.__config.DATA.STD
        )
        # Convert from TCHW to CTHW.
        video = video.permute(1, 0, 2, 3)

        video = _spatial_sampling(video, config=self.__config)

        # Load the labels.
        labels = self.__labels[item]
        class_1 = torch.as_tensor(labels.num_flowers > 0, dtype=torch.long)
        class_2 = torch.as_tensor(labels.num_flowers > 1, dtype=torch.long)
        class_3 = torch.as_tensor(labels.num_flowers > 3, dtype=torch.long)
        class_4 = torch.as_tensor(labels.num_flowers > 5, dtype=torch.long)
        row_status = torch.as_tensor(labels.row_status, dtype=torch.long)
        return (
            [video],
            row_status,
            item,
            torch.zeros((1, 1)),
            {},
        )
