import enum


@enum.unique
class MarsMetadata(enum.Enum):
    """
    Schema for metadata from the MARS dataset.
    """

    CLIP = "clip"
    """
    Clip ID that this frame is associated with.
    """
    FRAME_NUM = "frame_num"
    """
    Frame number within the clip.
    """
    CAMERA = "camera"
    """
    The camera number that this frame is from.
    """
    TIMESTAMP = "timestamp"
    """
    The timestamp from the original video file that is associated with this
    frame. All frames in a clip should have monotonically-increasing timestamps.
    """

    FILE_ID = "file_id"
    """
    The unique ID that identifies this file on the disk.
    """
