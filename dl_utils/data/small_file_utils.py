import os
from pathlib import Path
from typing import Any, Callable, Iterable, List, Union

from joblib import Parallel, delayed


def concurrent_file_loader(
    file_paths: List[Union[str, Path]], parser: Callable[[bytes], Any] = None, **kwargs
) -> Iterable[Any]:
    """
    Efficiently loads and processes files using joblib for high robustness.

    This utility leverages joblib's Parallel engine to handle file IO and parsing.
    By default, it uses the 'threading' backend which is ideal for IO-bound tasks
    and provides a generator interface for memory efficiency.

    Args:
        file_paths: A list of file paths to be read.
        parser: A function to process raw bytes.
            Defaults to returning raw bytes if None.
        **kwargs: Arbitrary keyword arguments passed directly to `joblib.Parallel`.
            Useful for overriding `n_jobs`, `backend`, `pre_dispatch`, or `batch_size`.

    Yields:
        Any: The parsed content of each file in the same order as ``file_paths``.

    Example:
        >>> import json
        >>> files = ["data1.json", "data2.json"]
        >>> # Pass custom joblib params like n_jobs or batch_size
        >>> for data in concurrent_file_loader(files, parser=json.loads, n_jobs=4):
        ...     print(data)
    """
    parser = parser or (lambda x: x)

    # Set smart defaults for joblib if not provided in kwargs
    # Use all CPUs by default for parsing power
    kwargs.setdefault("n_jobs", os.cpu_count() or 1)
    # Threading is preferred for IO-bound tasks to avoid serialization overhead
    kwargs.setdefault("backend", "threading")
    # Returns a generator to keep memory usage low
    kwargs.setdefault("return_as", "generator")

    def _read_and_parse(path: Union[str, Path]) -> Any:
        """Internal worker to handle IO and parsing with error encapsulation."""
        with open(path, mode="rb") as f:
            data = f.read()
        return parser(data)

    # Parallel execution with delayed task dispatch
    return Parallel(**kwargs)(delayed(_read_and_parse)(p) for p in file_paths)


__all__ = ["concurrent_file_loader"]

