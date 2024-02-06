from pathlib import Path
from unittest import TestCase

from dolma.core.url_blocker import UrlBlocker

LOCAL_DATA = Path(__file__).parent.parent / "data"


class TestUrlBlocker(TestCase):
    def test_brave_adblocker(self):
        rules = [
            "-advertisement-icon.",
            "-advertisement-management/",
            "-advertisement.",
            "-advertisement/script.",
        ]
        engine = UrlBlocker(rules)

        to_block = "http://example.com/-advertisement-icon."
        base_url = "http://example.com/helloworld"
        self.assertTrue(engine.check_network_urls(to_block, base_url, "image"))
        self.assertTrue(engine.check_network_urls(to_block, base_url))
        self.assertTrue(engine.check_network_urls(to_block))
        self.assertFalse(engine.check_network_urls(to_block, None, "document"))

        not_to_block = "http://example.com/main-icon"
        self.assertFalse(engine.check_network_urls(not_to_block, base_url, "image"))
        self.assertFalse(engine.check_network_urls(not_to_block, base_url))
        self.assertFalse(engine.check_network_urls(not_to_block))

    def test_load_from_file(self):
        engine = UrlBlocker.from_adb_paths(LOCAL_DATA / "urls/easylist.txt.gz")

        # global rules
        self.assertTrue(engine.check_network_urls("berush.com"))
        self.assertFalse(engine.check_network_urls("example.com"))

        # image rules
        self.assertTrue(engine.check_network_urls("pjatr.com", None, "image"))
        self.assertFalse(engine.check_network_urls("pjatr.com", None, "document"))
