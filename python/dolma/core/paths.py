import glob
import re
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Union
from urllib.parse import urlparse

from fsspec import AbstractFileSystem, get_filesystem_class

__all__ = [
    "glob_path",
    "sub_prefix",
    "add_suffix",
    "sub_suffix",
    "make_relative",
    "mkdir_p",
    "split_path",
    "join_path",
    "is_glob",
    "split_glob",
]


FS_KWARGS: Dict[str, Dict[str, Any]] = {
    "": {"auto_mkdir": True},
}


def _get_fs(path: Union[Path, str]) -> AbstractFileSystem:
    """
    Get the filesystem class for a given path.
    """
    path = str(path)
    protocol = urlparse(path).scheme
    fs = get_filesystem_class(protocol)(**FS_KWARGS.get(protocol, {}))

    # patch glob method to support recursive globbing
    if protocol == "":
        fs.glob = partial(glob.glob, recursive=True)

    return fs


def _escape_glob(s: Union[str, Path]) -> str:
    """
    Escape glob characters in a string.
    """
    r"(?<!\\)[*?[\]]"
    s = str(s)
    s = re.sub(r"(?<!\\)\*", "\u2581", s)
    s = re.sub(r"(?<!\\)\?", "\u2582", s)
    s = re.sub(r"(?<!\\)\[", "\u2583", s)
    s = re.sub(r"(?<!\\)\]", "\u2584", s)
    return s


def _unescape_glob(s: Union[str, Path]) -> str:
    """
    Unescape glob characters in a string.
    """
    s = str(s)
    s = re.sub("\u2581", "*", s)
    s = re.sub("\u2582", "?", s)
    s = re.sub("\u2583", "[", s)
    s = re.sub("\u2584", "]", s)
    return s


def _pathify(path: Union[Path, str]) -> Tuple[str, Path]:
    """
    Return the protocol and path of a given path.
    """
    path = _escape_glob(str(path))
    parsed = urlparse(path)
    path = Path(f"{parsed.netloc}/{parsed.path}") if parsed.netloc else Path(parsed.path)
    return parsed.scheme, path


def _unpathify(protocol: str, path: Path) -> str:
    """
    Return a path from its protocol and path components.
    """
    path_str = _unescape_glob(str(path))
    if protocol:
        path_str = f"{protocol}://{path_str.lstrip('/')}"
    return path_str


def split_path(path: str) -> Tuple[str, Tuple[str, ...]]:
    """
    Split a path into its protocol and path components.
    """
    protocol, _path = _pathify(path)
    return protocol, tuple(_unescape_glob(p) for p in _path.parts)


def join_path(protocol: str, *parts: Union[str, Iterable[str]]) -> str:
    """
    Join a path from its protocol and path components.
    """
    all_parts = (_escape_glob(p) for p in chain.from_iterable([p] if isinstance(p, str) else p for p in parts))
    path = str(Path(*all_parts)).rstrip("/")
    if protocol:
        path = f"{protocol}://{path.lstrip('/')}"
    return _unescape_glob(path)


def glob_path(path: Union[Path, str], hidden_files: bool = False) -> Iterator[str]:
    """
    Expand a glob path into a list of paths.
    """
    path = str(path)
    protocol = urlparse(path).scheme
    fs = _get_fs(path)

    for gl in fs.glob(path):
        gl = str(gl)

        if not hidden_files and Path(gl).name.startswith("."):
            continue

        yield join_path(protocol, gl)


def sub_prefix(a: str, b: str) -> str:
    """
    Return the relative path of b from a.
    """
    prot_a, path_a = _pathify(a)
    prot_b, path_b = _pathify(b)

    if prot_a != prot_b:
        raise ValueError(f"Protocols of {a} and {b} do not match")

    try:
        diff = str(path_a.relative_to(path_b))
    except ValueError:
        diff = join_path(prot_a, path_a.parts)

    return _unescape_glob(diff)


def sub_suffix(a: str, b: str) -> str:
    """
    Remove b from the end of a.
    """
    prot_a, path_a = _pathify(a)
    prot_b, path_b = _pathify(b)

    if prot_b:
        raise ValueError(f"{b} is not a relative path")

    sub_path = re.sub(f"{path_b}$", "", str(path_a))
    sub_prot = f"{prot_a}://" if prot_a else ""

    # need to trim '/' from the end if (a) '/' is not the only symbol in the path or
    # (b) there is a protocol so absolute paths don't make sense
    if sub_path != "/" or sub_prot:
        sub_path = sub_path.rstrip("/")

    return _unescape_glob(sub_prot + sub_path)


def add_suffix(a: str, b: str) -> str:
    """
    Return the the path of a joined with b.
    """
    prot_a, path_a = _pathify(a)
    prot_b, path_b = _pathify(b)

    if prot_b:
        raise ValueError(f"{b} is not a relative path")

    return join_path(prot_a, str(path_a / path_b))


def mkdir_p(path: str) -> None:
    """
    Create a directory if it does not exist.
    """
    if is_glob(path):
        raise ValueError(f"Cannot create directory with glob pattern: {path}")

    fs = _get_fs(path)
    fs.makedirs(path, exist_ok=True)


def make_relative(paths: List[str]) -> Tuple[str, List[str]]:
    """Find minimum longest root shared among all paths"""
    if len(paths) == 0:
        raise ValueError("Cannot make relative path of empty list")

    common_prot, common_parts = (p := _pathify(paths[0]))[0], p[1].parts

    for path in paths:
        current_prot, current_path = _pathify(path)
        if current_prot != common_prot:
            raise ValueError(f"Protocols of {path} and {paths[0]} do not match")

        current_parts = current_path.parts
        for i in range(min(len(common_parts), len(current_parts))):
            if common_parts[i] != current_parts[i]:
                common_parts = common_parts[:i]
                break

    if len(common_parts) > 0:
        common_path = (f"{common_prot}://" if common_prot else "") + str(Path(*common_parts))
        relative_paths = [sub_prefix(path, common_path) for path in paths]
    else:
        common_path = f"{common_prot}://" if common_prot else ""
        relative_paths = [_unpathify("", _pathify(path)[1]) for path in paths]

    return common_path, relative_paths


def is_glob(path: str) -> bool:
    """
    Check if a path contains a glob wildcard.
    """
    return bool(re.search(r"(?<!\\)[*?[\]]", path))


def split_glob(path: str) -> Tuple[str, str]:
    """
    Partition a path on the first wildcard.
    """
    if not is_glob(path):
        return path, ""

    protocol, parts = split_path(path)

    i = min(i for i, c in enumerate(parts) if is_glob(c))

    path = join_path(protocol, *parts[:i])
    rest = join_path("", *parts[i:])
    return path, rest
