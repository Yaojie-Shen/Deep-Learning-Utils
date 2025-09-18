# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 14:37
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : video.py

import os
import subprocess
from typing import *

import cv2
import numpy as np
from joblib import Parallel, delayed

from .array import to_numpy
from .. import make_parent_dirs
from ..type_hint import FilePath, ArrayLike


def save_video(
        frames: ArrayLike,
        save_path: FilePath,
        fps: Union[int, float] = 30,
        codec: str = "avc1"
):
    """

    Args:
        frames: Video frames in shape (F, H, W, C). The pixel values should be in range [0, 255].
        save_path: Path to save video.
        fps: FPS of video, default 30.
        codec: Codec of video, default avc1.
    """
    frames = to_numpy(frames)

    height, width = frames.shape[1:3]

    make_parent_dirs(save_path)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()


def load_video(
        video_path: FilePath,
        resize: Union[Tuple[int, int], int] = None,
        center_crop: Union[Tuple[int, int], int] = None,
        max_frames: int = None,
) -> np.ndarray:
    """
    Load a video file.

    Args:
        video_path: Path to the video file.
        resize: Resize frames to the specified size. If None, no resizing. Accepts (width, height) or int.
        center_crop: Center crop frames to the specified size. If None, no cropping. Accepts (width, height) or int.
        max_frames: Maximum number of frames to load. If None, load all frames.

    Returns:
        Frames as a NumPy array with shape (F, H, W, C). Pixel values are in [0, 255], color order is RGB.

    Note:
        - If the video is grayscale, the color channel will be replicated to 3.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, _ = frame.shape

        if resize is not None:
            if isinstance(resize, int):
                # Resize by short edge
                if h > w:
                    frame = cv2.resize(frame, (512, int(512 * h / w)))
                else:
                    frame = cv2.resize(frame, (int(512 * w / h), 512))
            elif isinstance(resize, tuple) or isinstance(resize, list):
                frame = cv2.resize(frame, resize)
            else:
                raise ValueError(f"Invalid resize value: {resize}, expected int or tuple/list of two.")

        # Center crop
        if center_crop is not None:
            if isinstance(center_crop, int):
                frame = frame[h // 2 - center_crop // 2:h // 2 + center_crop // 2,
                        w // 2 - center_crop // 2:w // 2 + center_crop // 2]
            elif isinstance(center_crop, tuple) or isinstance(center_crop, list):
                frame = frame[h // 2 - center_crop[1] // 2:h // 2 + center_crop[1] // 2,
                        w // 2 - center_crop[0] // 2:w // 2 + center_crop[0] // 2]
            else:
                raise ValueError(f"Invalid center_crop value: {center_crop}, expected int or tuple/list of two.")

        frames.append(frame)
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    cap.release()

    return np.stack(frames)  # (f,h,w,c)


def get_video_fps(video_path: FilePath) -> float:
    """
    Retrieve the FPS of a video.

    Args:
        video_path: Path to the video file.

    Returns:
        The FPS of the video.
    """
    import cv2
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    return fps


def get_video_frame_count(video_path: FilePath) -> int:
    """
    Retrieve the total number of frames in a video.

    Args:
        video_path: Path to the video file.

    Returns:
        The number of frames in the video.
    """
    import cv2
    video = cv2.VideoCapture(video_path)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    return int(frame_count)


def get_video_duration(video_path: FilePath) -> Tuple[float, int, float]:
    """
    Retrieve the FPS, frame count, and duration (in seconds) of a video.

    Args:
        video_path: Path to the video file.
    Returns:
        A tuple containing FPS, frame count, and duration in seconds.
    """
    import cv2
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, frame_count, frame_count / fps


def get_video_duration_batch(video_paths: List[FilePath]) -> List[float]:
    """
    Get duration of videos in batch.

    Args:
        video_paths: List of paths to videos.

    Returns:
        A list of tuples, each containing FPS, frame count, and duration in seconds.
    """
    return Parallel(n_jobs=os.cpu_count())(
        delayed(get_video_duration)(p) for p in video_paths
    )


def convert_to_h265(input_file: AnyStr, output_file: AnyStr,
                    ffmpeg_exec: AnyStr = "/usr/bin/ffmpeg",
                    keyint: int = None,
                    overwrite: bool = False,
                    verbose: bool = False) -> None:
    """
    convert video to h265 format using ffmpeg
    @param input_file: input path
    @param output_file: output path
    @param ffmpeg_exec:
    @param keyint:
    @param overwrite: overwrite the existing file
    @param verbose: show ffmpeg output
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # `-max_muxing_queue_size 9999` is for the problem reported in:
    # https://stackoverflow.com/questions/49686244/ffmpeg-too-many-packets-buffered-for-output-stream-01
    # <!> This may cause OOM error.
    if keyint is None:
        command = [ffmpeg_exec, "-i", f"{input_file}", "-max_muxing_queue_size", "9999",
                   "-c:v", "libx265", "-vtag", "hvc1",
                   "-c:a", "copy", "-movflags", "faststart", f"{output_file}"]
    else:
        command = [ffmpeg_exec, "-i", f"{input_file}", "-max_muxing_queue_size", "9999",
                   "-c:v", "libx265", "-vtag", "hvc1", "-x265-params", f"keyint={keyint}",
                   "-c:a", "copy", "-movflags", "faststart", f"{output_file}"]
    if overwrite:
        command += ["-y"]
    else:
        command += ["-n"]
    subprocess.run(command,
                   stderr=subprocess.DEVNULL if not verbose else None,
                   stdout=subprocess.DEVNULL if not verbose else None)
    # TODO: return


def convert_to_h264(input_file: AnyStr, output_file: AnyStr,
                    ffmpeg_exec: AnyStr = "/usr/bin/ffmpeg",
                    keyint: int = None,
                    overwrite: bool = False,
                    verbose: bool = False) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if keyint is None:
        command = [ffmpeg_exec, "-i", f"{input_file}", "-max_muxing_queue_size", "9999",
                   "-c:v", "libx264",
                   "-c:a", "copy", "-movflags", "faststart", f"{output_file}"]
    else:
        command = [ffmpeg_exec, "-i", f"{input_file}", "-max_muxing_queue_size", "9999",
                   "-c:v", "libx264", "-x264-params", f"keyint={keyint}",
                   "-c:a", "copy", "-movflags", "faststart", f"{output_file}"]
    if overwrite:
        command += ["-y"]
    else:
        command += ["-n"]
    subprocess.run(command,
                   stderr=subprocess.DEVNULL if not verbose else None,
                   stdout=subprocess.DEVNULL if not verbose else None)
    # TODO: return


__all__ = [
    "save_video",
    "load_video",
    "get_video_fps",
    "get_video_frame_count",
    "get_video_duration",
    "get_video_duration_batch",
    "convert_to_h265",
    "convert_to_h264"
]
