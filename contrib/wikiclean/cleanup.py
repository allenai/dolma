#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "click",
#     "smart_open",
#     "tqdm",
# ]
# ///

import json
from pathlib import Path
import click
import smart_open
import tqdm


@click.command()
@click.option("--input-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
def main(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_files = list(input_dir.glob("*.gz"))
    for input_file in tqdm.tqdm(all_files, desc="Processing files", position=0, leave=True):
        output_file = output_dir / input_file.name

        with smart_open.open(input_file, "r", encoding="utf-8") as infile, \
             smart_open.open(output_file, "w", encoding="utf-8") as outfile:

            for line in tqdm.tqdm(infile, desc=f"Processing {input_file}", position=1, leave=False, unit_scale=True):
                doc = json.loads(line)

                if "text" not in doc:
                    continue

                text = doc["text"].strip()
                lines = text.split("\n")

                if any(line.startswith("Question:") for line in lines) and not any(line.startswith("Answer:") for line in lines):
                    # skip documents that contain questions but no answers
                    continue

                if lines[0].endswith(":") and lines[0].lower().startswith("flashcard"):
                    lines = lines[1:]
                    # first line is likely "Flashcard ...:"
                    continue

                # Remove the last line if it doesn't end with a full stop and doesn't start with "Answer:"
                if lines and not lines[-1].endswith(".") and not lines[-1].startswith("Answer:"):
                    lines = lines[:-1]

                # if there any no blank lines, add one blank after each line
                if not any(line.strip() == "" for line in lines):
                    lines = [line + "\n" for line in lines]

                doc["text"] = "\n".join(lines)

                outfile.write(json.dumps(doc) + "\n")


if __name__ == "__main__":
    main()
