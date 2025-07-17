# -*- coding: utf-8 -*-
# @Time    : 2022/9/25 14:37
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : video.py

import os
import subprocess
from typing import *

from joblib import Parallel, delayed

from .array import to_numpy
from .. import make_parent_dirs
from ..type_hint import FilePath, ArrayLike


def save_video(frames: ArrayLike, save_path: FilePath, fps: Union[int, float] = 30):
    """

    Args:
        frames: Video frames in shape (F, H, W, C). The pix
        save_path: Path to save video.
        fps: FPS of video, default 30.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python is required to save video")

    frames = to_numpy(frames)

    height, width = frames.shape[1:3]

    make_parent_dirs(save_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()


def _get_single_video_duration_info(video_path) -> (float, float, int):
    """
    return video duration in seconds
    :param video_path: video path
    :return: video duration, fps, frame count
    """
    import cv2
    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    return frame_count / fps, fps, int(frame_count)


def get_duration_info(video_paths: Union[str, Iterable]) -> (float, float, int):
    """

    :param video_paths: video path or a list of video path
    :return: video duration, fps, frame count
    """
    if isinstance(video_paths, str):
        return _get_single_video_duration_info(video_paths)
    else:
        return Parallel(n_jobs=os.cpu_count())(
            delayed(_get_single_video_duration_info)(path) for path in video_paths
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
    "get_duration_info",
    "convert_to_h265",
    "convert_to_h264"
]
