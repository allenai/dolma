import glob
from itertools import chain
import re
from functools import partial
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


def _pathify(path: Union[Path, str]) -> Tuple[str, Path]:
    """
    Return the protocol and path of a given path.
    """
    parsed = urlparse(str(path))
    path = Path(f"{parsed.netloc}/{parsed.path}") if parsed.netloc else Path(parsed.path)
    return parsed.scheme, path


def split_path(path: str) -> Tuple[str, Tuple[str, ...]]:
    """
    Split a path into its protocol and path components.
    """
    protocol, _path = _pathify(path)
    return protocol, _path.parts


def join_path(protocol: str, *parts: Union[str, Iterable[str]]) -> str:
    """
    Join a path from its protocol and path components.
    """
    all_parts = chain.from_iterable([p] if isinstance(p, str) else p for p in parts)
    path = str(Path(*all_parts)).rstrip("/")
    if protocol:
        path = f"{protocol}://{path.lstrip('/')}"
    return path


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
        elif protocol:
            gl = f"{protocol}://{gl}"
        yield gl


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
        diff = f"{prot_a}://{path_a}" if prot_a else str(path_a)

    return str(diff)


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

    return sub_prot + sub_path


def add_suffix(a: str, b: str) -> str:
    """
    Return the the path of a joined with b.
    """
    prot_a, path_a = _pathify(a)
    prot_b, path_b = _pathify(b)

    if prot_b:
        raise ValueError(f"{b} is not a relative path")

    return (f"{prot_a}://" if prot_a else "") + str(path_a / path_b)


def mkdir_p(path: str) -> None:
    """
    Create a directory if it does not exist.
    """
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
        relative_paths = [str(_pathify(path)[1]) for path in paths]

    return common_path, relative_paths
