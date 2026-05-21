import io
import importlib.util
import tarfile
from pathlib import Path

import pytest


def _load_download_url_blocklist_module():
    script_path = Path(__file__).parents[2] / "scripts" / "download_url_blocklist.py"
    spec = importlib.util.spec_from_file_location("download_url_blocklist", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_tar(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in members:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))


def test_extract_tar_gz_extracts_valid_member(tmp_path: Path) -> None:
    module = _load_download_url_blocklist_module()
    archive_path = tmp_path / "blacklists.tar.gz"
    extract_to = tmp_path / "extract"
    _write_tar(archive_path, [("blacklists/ads/domains", b"example.com\n")])

    module.extract_tar_gz(str(archive_path), str(extract_to))

    assert (extract_to / "blacklists" / "ads" / "domains").read_text() == "example.com\n"


@pytest.mark.parametrize("member_name", ["../escaped.txt", "/tmp/escaped.txt", "C:/escaped.txt"])
def test_extract_tar_gz_rejects_unsafe_member(tmp_path: Path, member_name: str) -> None:
    module = _load_download_url_blocklist_module()
    archive_path = tmp_path / "blacklists.tar.gz"
    extract_to = tmp_path / "extract"
    _write_tar(archive_path, [(member_name, b"escaped")])

    with pytest.raises(RuntimeError, match="unsafe archive member"):
        module.extract_tar_gz(str(archive_path), str(extract_to))
