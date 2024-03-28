import hashlib
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread
from contextlib import ExitStack
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING
from datetime import datetime

import necessary
import smart_open

from dolma.core.paths import cached_path

with necessary.necessary("requests") as REQUESTS_AVAILABLE:
    if REQUESTS_AVAILABLE or TYPE_CHECKING:
        import requests


with necessary.necessary("click") as CLICK_AVAILABLE:
    if CLICK_AVAILABLE or TYPE_CHECKING:
        import click


with necessary.necessary("tqdm") as TQDM_AVAILABLE:
    if TQDM_AVAILABLE or TYPE_CHECKING:
        import tqdm


with necessary.necessary("fake_useragent") as FAKE_USERAGENT_AVAILABLE:
    if FAKE_USERAGENT_AVAILABLE or TYPE_CHECKING:
        from fake_useragent import UserAgent


UA = UserAgent()
URL_SOURCE = "https://dolma-artifacts.org/smallweb/smallweb_20240306/smallweb.txt"
RAW_DESTINATION = "s3://ai2-llm/pretraining-data/sources/smallweb/raw/feeds"
DATE = datetime.now().strftime("%Y%m%d")


class ThreadedRequest(Thread):
    # constructor
    def __init__(self, url: str, timeout: int = 5):
        super().__init__()
        self.value = None
        self.url = url
        self.timeout = timeout

    # function executed in a new thread
    def run(self):
        try:
            self.value = requests.get(self.url, timeout=self.timeout, headers={"User-Agent": UA.random})
        except Exception:
            self.value = None


def download_rss(url: str, dest_prefix: str, timeout: int = 5, retries: int = 3) -> bool:
    try:
        while retries > 0:
            retries -= 1
            req_thread = ThreadedRequest(url, timeout)
            req_thread.start()
            req_thread.join(timeout**2)
            if req_thread.is_alive():
                # thread has timed out
                continue

            resp: Union[None, requests.Response] = req_thread.value

            if resp is not None and resp.status_code < 400:
                fn = hashlib.md5(url.encode('utf-8')).hexdigest()[:16]
                dest = f"{dest_prefix}/{DATE}/{fn[:2]}/{fn}.xml.gz"
                with smart_open.open(dest, mode='wt') as f:
                    f.write(resp.text)
                return True
    except Exception:
        ...

    return False


@click.command()
@click.option("--source", default=URL_SOURCE)
@click.option("--destination", default=RAW_DESTINATION)
@click.option("--metadata", default=None, type=click.Path(path_type=Path))
@click.option("--timeout", type=int, default=5)
@click.option("--max-workers", type=int, default=200)
def main(source: str, destination: str, metadata: Optional[Path], timeout: int, max_workers: int) -> None:

    with ExitStack() as stack:
        url_file = stack.enter_context(smart_open.open(cached_path(source), mode='rt', encoding='utf-8'))
        urls = set(ln.strip() for ln in url_file)
        url_file.close()

        if metadata is None:
            metadata_file = stack.enter_context(NamedTemporaryFile(mode='a+', delete=True))
        else:
            metadata.parent.mkdir(parents=True, exist_ok=True)
            metadata_file = stack.enter_context(smart_open.open(metadata, mode='a+', encoding='utf-8'))

        pbar = stack.enter_context(tqdm.tqdm(total=len(urls), desc="URLs", position=0))
        fail = stack.enter_context(tqdm.tqdm(desc="Failed URLs", position=1))

        metadata_file.seek(0, 0)
        already_processed = [ln.strip() for ln in metadata_file]
        metadata_file.seek(0, 0)

        for url in already_processed:
            urls.discard(url)
            metadata_file.write(url + "\n")
            pbar.update(1)
        pbar.refresh()
        print(f"Already processed {len(already_processed)} URLs")

        executor = stack.enter_context(ThreadPoolExecutor(max_workers=max_workers))

        futures = {
            executor.submit(download_rss, dest_prefix=destination, url=url, timeout=timeout): url
            for url in urls
        }
        for future in as_completed(futures):
            url = futures[future]
            pbar.update(1)
            pbar.refresh()

            if future.result():
                metadata_file.write(url + "\n")
            else:
                fail.update(1)
                fail.refresh()


if __name__ == "__main__":
    main()
