import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest import TestCase

from dolma.core.data_types import DocumentWithMetadata
from dolma.core.url_blocker import UrlBlocker
from dolma.taggers.url import BaseDomainTagger, BaseUrlTagger

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


class TestUrlMatcher(TestCase):
    links_tagger: BaseUrlTagger
    domains_tagger: BaseDomainTagger

    def setUp(self):
        self.stack = ExitStack()
        temp_dir = self.stack.enter_context(tempfile.TemporaryDirectory())

        test_links = temp_dir + "/test_links.txt"
        test_domains = temp_dir + "/test_domains.txt"

        with open(test_links, "w") as f:
            f.write("http://example.com/foo/bar\n")
            f.write("https://example2.com/foo\n")

        with open(test_domains, "w") as f:
            f.write("example.com\n")
            f.write("0.0.0.0 example2.com\n")
            f.write("::1 example3.com\n")

        class TestLinksTagger(BaseUrlTagger):
            BLOCKLIST_PATHS = [test_links]

        class TestDomainTagger(BaseDomainTagger):
            BLOCKLIST_PATHS = [test_domains]

        self.links_tagger = TestLinksTagger()
        self.domains_tagger = TestDomainTagger()

    def tearDown(self):
        self.stack.close()

    def make_doc(self, url: str) -> DocumentWithMetadata:
        return DocumentWithMetadata(
            source=__file__,
            version="0",
            id="0",
            text="",
            metadata={"url": url},
        )

    def test_links_tagger(self):
        doc = self.make_doc("http://example.com/foo/bar")
        self.assertTrue(self.links_tagger.predict(doc).spans)

        doc = self.make_doc("https://example.com/foo/bar")
        self.assertTrue(self.links_tagger.predict(doc).spans)

        doc = self.make_doc("example.com/foo/bar/")
        self.assertTrue(self.links_tagger.predict(doc).spans)

        doc = self.make_doc("http://example.com/foo/")
        self.assertFalse(self.links_tagger.predict(doc).spans)

        doc = self.make_doc("https://example.com/foo/bar/baz")
        self.assertFalse(self.links_tagger.predict(doc).spans)

        doc = self.make_doc("s3://example2.com/foo")
        self.assertTrue(self.links_tagger.predict(doc).spans)

        doc = self.make_doc("https://example2.com/foo/bar")
        self.assertFalse(self.links_tagger.predict(doc).spans)

    def test_domains_tagger(self):
        doc = self.make_doc("http://example.com")
        self.assertTrue(self.domains_tagger.predict(doc).spans)

        doc = self.make_doc("EXAMPLE.COM")
        self.assertTrue(self.domains_tagger.predict(doc).spans)

        doc = self.make_doc("https://example2.com")
        self.assertTrue(self.domains_tagger.predict(doc).spans)

        doc = self.make_doc("example3.com")
        self.assertTrue(self.domains_tagger.predict(doc).spans)

        doc = self.make_doc("http://example4.com")
        self.assertFalse(self.domains_tagger.predict(doc).spans)

        doc = self.make_doc("http://example.com/foo")
        self.assertTrue(self.domains_tagger.predict(doc).spans)
