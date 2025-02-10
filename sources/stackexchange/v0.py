import argparse
import os
import re
from contextlib import ExitStack
from typing import Any, Callable, Iterator

import libarchive  # pyright: ignore
import py7zr  # pyright: ignore
from resiliparse.extract.html2text import extract_plain_text  # pyright: ignore
import pyarrow as pa
import pyarrow.parquet as pq
from libarchive.entry import ArchiveEntry  # pyright: ignore
from lxml import etree  # pyright: ignore
from tqdm import tqdm

os.environ["PYTHONBREAKPOINT"] = "ipdb.set_trace"


post_types = {
    "1": "Question",
    "2": "Answer",
    "3": "Orphaned tag wiki",
    "4": "Tag wiki excerpt",
    "5": "Tag wiki",
    "6": "Moderator nomination",
    "7": "Wiki placeholder",
    "8": "Privilege wiki",
    "9": "Article",
    "10": "HelpArticle",
    "11": "Unknown",
    "12": "Collection",
    "13": "ModeratorQuestionnaireResponse",
    "14": "Announcement",
    "15": "CollectiveDiscussion",
    "16": "CollectiveCollection",
}

POSTS_MAP: dict[str, Callable[[str | None], Any]] = {
    "AcceptedAnswerId": lambda x: int(x or 0),
    "AnswerCount": lambda x: int(x or 0),
    "Body": lambda x: extract_plain_text(x or "").strip(),
    "ClosedDate": lambda x: str(x or ""),
    "CommentCount": lambda x: int(x or 0),
    "CommunityOwnedDate": lambda x: str(x or ""),
    "ContentLicense": lambda x: str(x or ""),
    "CreationDate": lambda x: str(x or ""),
    "Id": lambda x: int(x or 0),
    "LastActivityDate": lambda x: str(x or ""),
    "LastEditDate": lambda x: str(x or ""),
    "LastEditorDisplayName": lambda x: str(x or ""),
    "LastEditorUserId": lambda x: int(x or 0),
    "OwnerDisplayName": lambda x: str(x or ""),
    "OwnerUserId": lambda x: int(x or 0),
    "ParentId": lambda x: int(x or 0),
    "PostTypeId": lambda x: post_types.get(x or "11", "Unknown"),
    "Score": lambda x: int(x or 0),
    "Tags": lambda x: str(x or ""),
    "Title": lambda x: str(x or ""),
    "ViewCount": lambda x: int(x or 0),
}

COMMENTS_MAP: dict[str, Callable[[str | None], Any]] = {
    "ContentLicense": lambda x: str(x or ""),
    "CreationDate": lambda x: str(x or ""),
    "Id": lambda x: int(x or 0),
    "PostId": lambda x: int(x or 0),
    "Score": lambda x: int(x or 0),
    "Text": lambda x: str(x or ""),
    "UserDisplayName": lambda x: str(x or ""),
    "UserId": lambda x: int(x or 0),
}

USERS_MAP: dict[str, Callable[[str | None], Any]] = {
    "Id": lambda x: int(x or 0),
    "Reputation": lambda x: int(x or 0),
    "CreationDate": lambda x: str(x or ""),
    "DisplayName": lambda x: str(x or ""),
    "LastAccessDate": lambda x: str(x or ""),
    "WebsiteUrl": lambda x: str(x or ""),
    "Location": lambda x: str(x or ""),
    "AboutMe": lambda x: str(x or ""),
    "Views": lambda x: int(x or 0),
    "UpVotes": lambda x: int(x or 0),
    "DownVotes": lambda x: int(x or 0),
    "ProfileImageUrl": lambda x: str(x or ""),
    "EmailHash": lambda x: str(x or ""),
    "AccountId": lambda x: int(x or 0),
}


def get_7z_uncompressed_size(sz_path, entry_name):
    with py7zr.SevenZipFile(sz_path, mode="r") as z:
        for entry in z.list():
            if entry.filename == entry_name:
                return entry.uncompressed
        raise FileNotFoundError(f"File {entry_name} not found in archive {sz_path}")


def stream_xml_from_7z(
    archive_path: str, filename: str, target_xpath: str = "//*", block_size: int = 8192
) -> Iterator[etree._Element]:
    """
    Stream XML nodes from a file within a 7z archive, parsing them lazily.

    Args:
        archive_path (str): Path to the 7z archive
        filename (str): Name of the XML file within the archive
        target_xpath (str, optional): XPath expression to filter nodes. Defaults to "//*".
        block_size (int, optional): Size of blocks to read. Defaults to 8192.

    Yields:
        lxml.etree._Element: XML nodes matching the target_xpath

    Raises:
        FileNotFoundError: If archive or file within archive is not found
        ValueError: If file is not valid XML
    """
    # Initialize the XML parser that will receive chunks of data
    parser = etree.XMLPullParser(events=("end",), recover=True)

    with ExitStack() as stack:
        archive = stack.enter_context(libarchive.file_reader(archive_path))
        # Find the target file in the archive
        for entry in archive:
            if entry.pathname != filename:
                continue

            archive_name = os.path.basename(archive_path)
            pbar = tqdm(
                total=get_7z_uncompressed_size(archive_path, filename),
                desc=f"Bytes {archive_name}::{filename}",
                unit="B",
                unit_scale=True,
            )
            prev_line = b""
            for chunk in entry.get_blocks(block_size):
                pbar.update(len(chunk))
                first_seg, *segments = re.split(b"\r*\n|\r", chunk)
                if segments:
                    # there's at least one line break in the chunk, so we can yield the previous line
                    yield prev_line + first_seg
                    yield from segments[:-1]
                    prev_line = segments[-1]
                else:
                    # no line breaks in the chunk, so we need to accumulate it
                    prev_line += chunk


def process_file(
    archive_path: str,
    output_dir: str,
    entry_name: str,
    entry_map: dict[str, Callable[[str| None], Any]],
    batch_size: int = 100_000,
    block_size: int = 8192,
):
    entry_prefix, _ = os.path.basename(entry_name.lower()).split(".", 1)
    archive_name = os.path.basename(archive_path)

    os.makedirs(output_dir, exist_ok=True)
    data = []
    schema = None

    with ExitStack() as stack:
        xml_elements = stream_xml_from_7z(archive_path, entry_name, block_size=block_size)
        files_pbar = tqdm(desc=f"Files {archive_name}::{entry_name}")
        elements_pbar = tqdm(xml_elements, desc=f"Rows {archive_name}::{entry_name}")

        for row in elements_pbar:
            if not row.strip().startswith(b"<row"):
                continue

            row = etree.fromstring(row)

            if not row.attrib:
                continue

            data.append({k: v(row.attrib.get(k, None)) for k, v in entry_map.items()})

            if schema is None:
                schema = pa.Table.from_pylist(data).schema

            if len(data) >= batch_size:
                table = pa.Table.from_pylist(data, schema=schema)
                pq.write_table(
                    table,
                    os.path.join(output_dir, f"{entry_prefix}_{files_pbar.n:06d}.parquet"),
                )
                data = []
                files_pbar.update(1)
        # Write any remaining data

        if data:
            table = pa.Table.from_pylist(data, schema=schema)
            pq.write_table(
                table,
                os.path.join(output_dir, f"{entry_prefix}_{files_pbar.n:06d}.parquet"),
            )
            files_pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description="Convert Stack Exchange 7z XML dumps to Parquet format")
    parser.add_argument("archive_path", help="Path to the 7z archive")
    parser.add_argument("output_dir", help="Directory where Parquet files will be saved")
    parser.add_argument(
        "--batch-size", type=int, default=100000, help="Number of rows to process at once (default: 100000)"
    )
    parser.add_argument("--block-size", type=int, default=8192, help="Size of blocks to read (default: 8192)")

    args = parser.parse_args()

    if os.path.isdir(args.archive_path):
        archive_paths = [
            os.path.join(args.archive_path, p) for p in os.listdir(args.archive_path) if p.endswith("7z")
        ]
    else:
        archive_paths = [args.archive_path]

    for archive_path in tqdm(archive_paths, desc="Archives"):
        for entry_name, entry_map in [("Posts.xml", POSTS_MAP), ("Comments.xml", COMMENTS_MAP), ("Users.xml", USERS_MAP)]:
            clean_entry_name = entry_name.split(".", 1)[0].lower()
            clean_forum_name = archive_path.split("/")[-1].rsplit(".", 1)[0].lower().replace(".", "_")
            output_path = os.path.join(args.output_dir, clean_entry_name, f"forum={clean_forum_name}")
            process_file(
                archive_path=archive_path,
                output_dir=output_path,
                entry_name=entry_name,
                entry_map=entry_map,  # pyright: ignore
                batch_size=args.batch_size,
                block_size=args.block_size,
            )


if __name__ == "__main__":
    main()
