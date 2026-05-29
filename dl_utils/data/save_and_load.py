# -*- coding: utf-8 -*-
# @Time    : 1/19/26
# @Author  : Yaojie Shen
# @Project : Deep-Learning-Utils
# @File    : save_and_load.py

import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Union

from joblib import Parallel, delayed

PathLike = Union[str, Path]


def save_text(text: str, file):
    """Save text content to a UTF-8 text file.

    Parent directories are created automatically before writing.

    Args:
        text: Text content to write.
        file: Destination file path.
    """

    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as fp:
        fp.write(text)


def load_text(file) -> str:
    """Load the full content of a text file.

    Args:
        file: Source file path.

    Returns:
        The text content read from ``file``.
    """

    with open(file, "r") as fp:
        return fp.read()


def save_bytes(data: bytes, file):
    """Save raw bytes to a binary file.

    Parent directories are created automatically before writing.

    Args:
        data: Bytes content to write.
        file: Destination file path.
    """

    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "wb") as fp:
        fp.write(data)


def load_bytes(file) -> bytes:
    """Load the full content of a binary file.

    Args:
        file: Source file path.

    Returns:
        The bytes read from ``file``.
    """

    with open(file, "rb") as fp:
        return fp.read()


def save_pickle(obj, file):
    """Serialize an object to a pickle file.

    Warning:
        Pickle is not safe for untrusted data. Only load pickle files from
        trusted sources.

    Args:
        obj: Python object to serialize.
        file: Destination pickle file path.
    """

    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "wb") as fp:
        pickle.dump(obj, fp)


def load_pickle(file):
    """Load an object from a pickle file.

    Warning:
        Pickle can execute arbitrary code while loading. Only load pickle files
        from trusted sources.

    Args:
        file: Source pickle file path.

    Returns:
        The Python object deserialized from ``file``.
    """

    with open(file, "rb") as fp:
        return pickle.load(fp)


class _JsonBytesEncoder(json.JSONEncoder):
    """Object of type bytes is not JSON serializable, convert it to string before saving"""

    def default(self, obj):
        """Convert ``bytes`` values to UTF-8 strings before JSON encoding.

        Args:
            obj: Object to encode.

        Returns:
            A JSON-serializable representation of ``obj``.
        """

        if isinstance(obj, bytes):  # bytes->str
            return str(obj, encoding="utf-8")
        return json.JSONEncoder.default(self, obj)


def save_json(data, file, save_pretty=False, **kwargs):
    """Save an object as JSON.

    ``bytes`` values are converted to UTF-8 strings by the default custom JSON
    encoder. The JSON string is fully serialized before writing, so serialization
    errors do not leave a partially written file.

    Args:
        data: JSON-serializable object to save.
        file: Destination file path.
        save_pretty: If ``True``, write human-readable JSON with indentation and
            ``ensure_ascii=False``.
        **kwargs: Extra keyword arguments forwarded to :func:`json.dumps`.
    """

    _kwargs = {"cls": _JsonBytesEncoder}
    if save_pretty:
        _kwargs.update({"indent": 4, "ensure_ascii": False})
    _kwargs.update(kwargs)

    # Dump to string first to avoid writing if error occurs
    s = json.dumps(data, **_kwargs)
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as fp:
        fp.write(s)


def load_json(file):
    """Load a JSON file.

    Args:
        file: Source JSON file path.

    Returns:
        The Python object decoded from the JSON file.
    """

    with open(file, "r") as fp:
        return json.load(fp)


def save_jsonl(data, file, **kwargs):
    """Save an iterable of objects as JSONL or a JSON array.

    The function writes one item at a time, so ``data`` can be any iterable,
    including a generator. For normal file suffixes such as ``.jsonl``, each item
    is serialized as one JSON object per line. If ``file`` ends with ``.json``,
    the same stream of items is written as a valid JSON array instead.

    ``bytes`` values are converted to UTF-8 strings by the default custom JSON
    encoder.

    Args:
        data: Iterable of JSON-serializable objects to save.
        file: Destination file path. A ``.json`` suffix switches the output
            format from JSONL to a JSON array.
        **kwargs: Extra keyword arguments forwarded to :func:`json.dumps`.

    Raises:
        ValueError: If JSONL output would contain a newline inside one line, for
            example when passing pretty-print options such as ``indent=2``.
    """

    _kwargs = {"cls": _JsonBytesEncoder}
    _kwargs.update(kwargs)

    path = Path(file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(file, "w") as fp:
        if path.suffix.lower() == ".json":
            fp.write("[\n")
            for idx, item in enumerate(data):
                if idx > 0:
                    fp.write(",\n")
                fp.write(json.dumps(item, **_kwargs))
            fp.write("\n]")
            return
        else:
            for idx, item in enumerate(data):
                line = json.dumps(item, **_kwargs)
                if "\n" in line:
                    raise ValueError(
                        "JSONL line contains newline. Avoid pretty JSON (e.g., indent=2). "
                        f"Line index: {idx}."
                    )
                fp.write(line)
                fp.write("\n")


def iter_jsonl(file, max_samples: Optional[int] = None):
    """Create an iterable view over a JSONL file.

    Empty lines are skipped. The returned object supports iteration, ``len()``,
    and non-negative integer indexing. Length and indexing are backed by a lazy
    offset index built on first use.

    Args:
        file: Source JSONL file path.
        max_samples: Optional maximum number of non-empty JSONL records to expose.

    Returns:
        An iterable object yielding one decoded JSON object per non-empty line.
    """

    class _JsonlIterable:
        def __init__(self, file_path, max_samples: Optional[int] = None):
            self.file_path = file_path
            self._offsets = None
            self._max_samples = max_samples

        def _ensure_offsets(self):
            if self._offsets is not None:
                return
            offsets = []
            with open(self.file_path, "r") as fp:
                while True:
                    pos = fp.tell()
                    line = fp.readline()
                    if not line:
                        break
                    if line.strip():
                        offsets.append(pos)
            self._offsets = offsets

        def __iter__(self):
            with open(self.file_path, "r") as fp:
                count = 0
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    if self._max_samples is not None and count >= self._max_samples:
                        break
                    count += 1
                    yield json.loads(line)

        def __len__(self):
            self._ensure_offsets()
            if self._max_samples is None:
                return len(self._offsets)
            return min(len(self._offsets), self._max_samples)

        def __getitem__(self, index):
            if not isinstance(index, int):
                raise TypeError("index must be int")
            if index < 0:
                raise IndexError("negative index is not supported")
            self._ensure_offsets()
            effective_len = len(self)
            if index >= effective_len:
                raise IndexError("index out of range")

            with open(self.file_path, "r") as fp:
                fp.seek(self._offsets[index])
                line = fp.readline()
                return json.loads(line)

    return _JsonlIterable(file, max_samples=max_samples)


def load_jsonl(file, max_samples: Optional[int] = None):
    """Load a JSONL file into a list.

    Args:
        file: Source JSONL file path.
        max_samples: Optional maximum number of non-empty JSONL records to load.

    Returns:
        A list of decoded JSON objects.
    """

    return list(iter_jsonl(file, max_samples=max_samples))


def _resolve_files(
    files_or_dir: Union[PathLike, Iterable[PathLike]],
    pattern: Optional[str] = None,
    sort: bool = True,
) -> List[Path]:
    """Resolve file inputs into a concrete list of paths.

    Args:
        files_or_dir: A directory, a single file path, or an iterable containing
            file and/or directory paths.
        pattern: Glob pattern used when expanding directories. Required if any
            input path is a directory; ignored for explicit file paths.
        sort: Whether files found from directory expansion should be sorted.

    Returns:
        A list of resolved file paths. Explicit file-list order is preserved,
        while directories inside that list are expanded deterministically when
        ``sort=True``.
    """

    def _expand_path(path: Path) -> List[Path]:
        if path.is_dir():
            if pattern is None:
                raise ValueError(
                    "pattern must be provided when files_or_dir contains a directory. "
                    "For example: pattern='*.json' or pattern='**/*.json'."
                )
            paths = path.glob(pattern)
            files = [p for p in paths if p.is_file()]
            return sorted(files) if sort else files
        return [path]

    if isinstance(files_or_dir, (str, Path)):
        return _expand_path(Path(files_or_dir))

    files = []
    for item in files_or_dir:
        # Explicit file lists preserve caller-provided order. Directories inside
        # that list are expanded deterministically when ``sort=True``.
        files.extend(_expand_path(Path(item)))
    return files


def concurrent_file_loader(
    file_paths: Iterable[PathLike],
    loader: Optional[Callable[..., Any]] = None,
    load_kwargs: Optional[Dict[str, Any]] = None,
    concurrency_limit: Optional[int] = None,
    chunk_size: Optional[int] = None,
    **kwargs,
) -> Iterable[Any]:
    """Load many files concurrently.

    This is a small IO-bound primitive used by higher-level save/load helpers.
    ``loader`` receives one file path and returns the loaded object. Results are
    yielded in the same order as ``file_paths`` when using joblib's default
    ordered generator mode.

    Args:
        file_paths: File paths to read.
        loader: Function used to load one file path. If ``None``,
            :func:`load_bytes` is used.
        load_kwargs: Extra keyword arguments passed to ``loader``.
        concurrency_limit: Alias for joblib ``n_jobs``.
        chunk_size: Alias for joblib ``batch_size``.
        **kwargs: Extra keyword arguments passed to :class:`joblib.Parallel`.

    Returns:
        An iterable of loaded file contents.
    """

    loader = loader or load_bytes
    load_kwargs = load_kwargs or {}

    if concurrency_limit is not None:
        kwargs.setdefault("n_jobs", concurrency_limit)
    if chunk_size is not None:
        kwargs.setdefault("batch_size", chunk_size)

    kwargs.setdefault("n_jobs", os.cpu_count() or 1)
    kwargs.setdefault("backend", "threading")
    kwargs.setdefault("return_as", "generator")

    def _load_file(path: PathLike) -> Any:
        return loader(path, **load_kwargs)

    return Parallel(**kwargs)(delayed(_load_file)(p) for p in file_paths)


def iter_files(
    files_or_dir: Union[PathLike, Iterable[PathLike]],
    *,
    pattern: Optional[str] = None,
    sort: bool = True,
    n_jobs: Optional[int] = None,
    flatten: bool = False,
    loader: Optional[Callable[..., Any]] = None,
    load_kwargs: Optional[Dict[str, Any]] = None,
    **parallel_kwargs,
) -> Iterator[Any]:
    """Iterate over objects loaded from a directory or file list.

    ``loader`` is any callable that accepts a file path and returns the loaded
    object, such as :func:`load_json`, :func:`load_text`, :func:`load_pickle`, or
    a custom function. :func:`load_bytes` is used by default.

    Args:
        files_or_dir: A directory, a single file, or an iterable of files and/or
            directories. If all inputs are explicit files, ``pattern`` can be
            omitted.
        pattern: Optional glob pattern used when expanding directories. Required
            if ``files_or_dir`` is a directory or contains directories. Common
            examples are ``"*.json"`` for direct JSON children, ``"*.jsonl"`` for
            direct JSONL children, ``"**/*.json"`` for recursive JSON matching
            with :meth:`pathlib.Path.glob`, and ``"part-*.json"`` for prefixed
            shard files.
        sort: Whether directory expansion should be sorted. Explicit file lists
            keep caller-provided order.
        n_jobs: Number of parallel reader jobs. Defaults to all CPUs.
        flatten: If ``True`` and a loaded object is a list, yield each list item;
            otherwise yield one object per file.
        loader: Callable used to load one file path. Defaults to
            :func:`load_bytes`.
        load_kwargs: Extra keyword arguments passed to ``loader``.
        **parallel_kwargs: Extra keyword arguments passed to
            :class:`joblib.Parallel`.

    Yields:
        Loaded objects, or items inside loaded lists when ``flatten=True``.

    Examples:
        Skip files that fail to load by wrapping the loader with ``try`` /
        ``except`` and filtering out ``None`` values::

            def safe_load_json(path):
                try:
                    return load_json(path)
                except Exception as exc:
                    print(f"Failed to load {path}: {exc}")
                    return None

            items = (
                item
                for item in iter_files(folder, pattern="**/*.json", loader=safe_load_json)
                if item is not None
            )
    """

    files = _resolve_files(
        files_or_dir,
        pattern=pattern,
        sort=sort,
    )
    loader = loader or load_bytes
    load_kwargs = load_kwargs or {}

    if n_jobs is not None:
        parallel_kwargs.setdefault("n_jobs", n_jobs)

    for data in concurrent_file_loader(
        files,
        loader=loader,
        load_kwargs=load_kwargs,
        **parallel_kwargs,
    ):
        if flatten and isinstance(data, list):
            yield from data
        else:
            yield data


def load_files(
    files_or_dir: Union[PathLike, Iterable[PathLike]],
    *,
    pattern: Optional[str] = None,
    sort: bool = True,
    n_jobs: Optional[int] = None,
    flatten: bool = False,
    loader: Optional[Callable[..., Any]] = None,
    load_kwargs: Optional[Dict[str, Any]] = None,
    **parallel_kwargs,
) -> List[Any]:
    """Load many files into memory as a list.

    This is the eager counterpart of :func:`iter_files`.

    Args:
        files_or_dir: A directory, a single file, or an iterable of files and/or
            directories. If all inputs are explicit files, ``pattern`` can be
            omitted.
        pattern: Optional glob pattern used when expanding directories. Required
            if ``files_or_dir`` is a directory or contains directories. Common
            examples are ``"*.json"`` for direct JSON children, ``"*.jsonl"`` for
            direct JSONL children, ``"**/*.json"`` for recursive JSON matching
            with :meth:`pathlib.Path.glob`, and ``"part-*.json"`` for prefixed
            shard files.
        sort: Whether directory expansion should be sorted. Explicit file lists
            keep caller-provided order.
        n_jobs: Number of parallel reader jobs. Defaults to all CPUs.
        flatten: If ``True`` and a loaded object is a list, append each list item;
            otherwise append one object per file.
        loader: Callable used to load one file path. Defaults to
            :func:`load_bytes`.
        load_kwargs: Extra keyword arguments passed to ``loader``.
        **parallel_kwargs: Extra keyword arguments passed to
            :class:`joblib.Parallel`.

    Returns:
        A list of loaded objects, or flattened list items when ``flatten=True``.
    """

    return list(
        iter_files(
            files_or_dir,
            pattern=pattern,
            sort=sort,
            n_jobs=n_jobs,
            flatten=flatten,
            loader=loader,
            load_kwargs=load_kwargs,
            **parallel_kwargs,
        )
    )


__all__ = [
    "save_text",
    "load_text",
    "save_bytes",
    "load_bytes",
    "save_pickle",
    "load_pickle",
    "save_json",
    "load_json",
    "save_jsonl",
    "iter_jsonl",
    "load_jsonl",
    "concurrent_file_loader",
    "iter_files",
    "load_files",
]
