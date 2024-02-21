import json
import re
import smart_open
from tqdm import tqdm
import yaml

LIST_CATALOG_PATH = "s3://dolma-artifacts/blocklist_brave/blocklist_brave-20240206/list_catalog.json"


def normalize_title(title: str) -> str:
    title_no_aph = title.replace("'s", "")
    lower_title = title_no_aph.lower()
    replaced_nonalpha_title = re.sub(r"[^a-zA-Z0-9]", "_", lower_title)
    removed_duplicate_underscores_title = re.sub(r"_+", "_", replaced_nonalpha_title)
    removed_leading_trailing_underscores_title = removed_duplicate_underscores_title.strip("_")
    return removed_leading_trailing_underscores_title


def main():
    destination_path, _ = LIST_CATALOG_PATH.rsplit("/", 1)

    # load the list catalog
    with smart_open.open(LIST_CATALOG_PATH, 'rt') as list_catalog_file:
        list_catalog = json.load(list_catalog_file)

    for catalog in tqdm(list_catalog, desc="Downloading lists"):
        catalog_name = normalize_title(catalog["title"])
        headers = '\n'.join(f'!{ln}' for ln in yaml.dump(catalog, sort_keys=False).split('\n') if ln.strip())
        content = []

        for source in catalog["sources"]:
            with smart_open.open(source['url'], 'rt') as source_file:
                if (title := source.get("title")):
                    source_name = normalize_title(title)
                else:
                    _, filename = source["url"].rsplit("/", 1)
                    source_name = normalize_title(f"__{filename}")

                with smart_open.open(f"{destination_path}/{catalog_name}/{source_name}.txt", "wt") as destination_file:
                    destination_file.write(source_file.read())
                    source_file.seek(0)
                content.extend([sln for ln in source_file if (sln := ln.strip()) and sln[0] != "!"])

        with smart_open.open(f"{destination_path}/{catalog_name}.txt", "wt") as destination_file:
            destination_file.write(headers)
            destination_file.write('\n')
            destination_file.write('\n'.join(sorted(set(content))))


if __name__ == "__main__":
    main()
