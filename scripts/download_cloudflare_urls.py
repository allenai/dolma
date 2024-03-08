'''
Use Cloudflare API to download a list of top URLs
'''

import click
import requests
import json
from time import sleep
import smart_open
import tqdm

from dolma.core.paths import exists, mkdir_p, parent


@click.command()
@click.option('--api-key', help='Cloudflare API key', envvar='CLOUDFLARE_API_KEY', required=True)
@click.option('--destination', help='Destination prefix', default='s3://dolma-artifacts/cloudflare_ranking_buckets')
@click.option('--limit', help='Number of URLs to download', default=10, type=int)
@click.option('--sleep-time', help='Sleep time between requests', default=0.5, type=float)
def download_cloudflare_urls(api_key: str, destination: str, limit: int, sleep_time: float):
    BASE_URL = (
        "https://api.cloudflare.com/client/v4/radar/datasets"
        "?datasetType=RANKING_BUCKET&limit={limit}&offset={offset}"
    )
    # https://api.cloudflare.com/client/v4/radar/datasets?limit=10&datasetType=RANKING_BUCKET&offset=200

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    prog = tqdm.tqdm(desc="Downloading Cloudflare URLs")
    offset = 0
    _, base_name = destination.rsplit("/", 1)

    while True:
        url = BASE_URL.format(limit=limit, offset=offset)
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        if not data.get("success"):
            break

        for row in data["result"]["datasets"]:
            dataset_id = row["id"]
            end_date = row["meta"]["targetDateEnd"].replace("-", "")
            name = row["alias"]

            file_destination = f"{destination}/{base_name}-{end_date}/{name}.csv.gz"
            if exists(file_destination):
                prog.update(1)
                # print(f"Skipping {end_date}/{name}")
                continue

            mkdir_p(parent(file_destination))

            dataset_response = requests.post(
                "https://api.cloudflare.com/client/v4/radar/datasets/download",
                headers=headers,
                data=json.dumps({"datasetId": dataset_id}),
            )
            dataset_response.raise_for_status()
            dataset_data = dataset_response.json()

            if not dataset_data.get("success"):
                print(f"Error downloading {end_date}/{name}")
                continue

            with smart_open.open(dataset_data['result']['dataset']['url'], 'rt') as fin:
                with smart_open.open(file_destination, "wt") as fout:
                    fout.write(fin.read())

            sleep(sleep_time)
            prog.update(1)

        offset += limit


if __name__ == '__main__':
    download_cloudflare_urls()
