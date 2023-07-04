import glob
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, Union
from urllib.parse import urlparse

from fsspec import AbstractFileSystem, get_filesystem_class

__all__ = ["glob_path", "sub_path", "add_path"]


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


def glob_path(path: Union[Path, str]) -> Iterator[str]:
    """
    Expand a glob path into a list of paths.
    """
    path = str(path)
    protocol = urlparse(path).scheme
    fs = _get_fs(path)

    for gl in fs.glob(path):
        gl = str(gl)
        if protocol:
            gl = f"{protocol}://{gl}"
        yield gl


def sub_path(a: str, b: str) -> str:
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


def add_path(a: str, b: str) -> str:
    """
    Return the the path of a joined with b.
    """
    prot_a, path_a = _pathify(a)
    prot_b, path_b = _pathify(b)

    if prot_b:
        raise ValueError(f"{b} is not a relative path")

    # breakpoint()
    return (f"{prot_a}://" if prot_a else '') + str(path_a / path_b)
