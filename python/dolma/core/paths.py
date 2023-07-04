import glob
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, Union
from urllib.parse import urlparse

from fsspec import AbstractFileSystem, get_filesystem_class

FS_KWARGS: Dict[str, Dict[str, Any]] = {
    "": {"auto_mkdir": True},
}


def get_fs(path: Union[Path, str]) -> AbstractFileSystem:
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


def glob_path(path: Union[Path, str]) -> Iterator[str]:
    """
    Expand a glob path into a list of paths.
    """
    path = str(path)
    protocol = urlparse(path).scheme
    fs = get_fs(path)

    for gl in fs.glob(path):
        gl = str(gl)
        if protocol:
            gl = f"{protocol}://{gl}"
        yield gl
