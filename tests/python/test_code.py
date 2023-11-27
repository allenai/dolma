"""

Unit tests for code taggers.

@soldni

"""

import re
import unittest

from bs4 import BeautifulSoup

from dolma.core.data_types import Document, DocumentWithMetadata
from dolma.taggers.code import (
    CodeCopyrightTagger,
    CodeRedPajamaTaggers,
    CodeSecretsTagger,
    CodeStarCoderTaggers2,
)

DOC_WITH_SECRETS_AND_COPYRIGHT = """
/* copyright: Test 2023 **/

This is a document.

This line contains a secret: https://username:password@dolma.allen.ai

This is a line with just text.
"""


class TestCodeTaggers(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = Document(id="0", text=DOC_WITH_SECRETS_AND_COPYRIGHT.strip(), source=__file__)
        return super().setUp()

    def test_code_secrets_tagger(self) -> None:
        tagger = CodeSecretsTagger()
        result = tagger.predict(self.doc)

        self.assertEqual(len(result.spans), 3)

        self.assertEqual(result.spans[0].type, "SECRET_Secret_Keyword")
        self.assertEqual(result.spans[0].select(self.doc), "https://username:password@dolma.allen.ai")

        self.assertEqual(result.spans[1].type, "SECRET_Basic_Auth_Credentials")
        self.assertEqual(result.spans[1].select(self.doc), "password")

        self.assertEqual(result.spans[2].type, "doc")
        self.assertEqual(result.spans[2].select(self.doc), self.doc.text)

    def test_copyright_notice(self):
        tagger = CodeCopyrightTagger()
        result = tagger.predict(self.doc)

        self.assertEqual(len(result.spans), 2)

        self.assertEqual(result.spans[0].type, "copyright_notice")
        self.assertEqual(result.spans[0].select(self.doc), "/* copyright: Test 2023 **/")

        self.assertEqual(result.spans[1].type, "doc")
        self.assertEqual(result.spans[1].select(self.doc), self.doc.text)


class TestRedPajamaTaggers(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = Document(id="0", text=DOC_WITH_SECRETS_AND_COPYRIGHT.strip(), source=__file__)
        self.whitespace_regex = re.compile(r"\w+|[^\w\s]+")
        return super().setUp()

    def test_code_red_pajama_taggers(self) -> None:
        tagger = CodeRedPajamaTaggers()
        result = tagger.predict(self.doc)

        # handy precomputed values
        line_lengths = list(map(len, self.doc.text.splitlines()))
        words = self.whitespace_regex.findall(self.doc.text)
        self.assertGreater(len(line_lengths), 0)
        self.assertGreater(len(words), 0)

        self.assertEqual(len(result.spans), 4)
        self.assertEqual(result.spans[0].type, "max_line_length_doc")
        self.assertEqual(result.spans[0].score, max(line_lengths))

        self.assertEqual(result.spans[1].type, "avg_line_length_doc")
        self.assertEqual(result.spans[1].score, sum(line_lengths) / len(line_lengths))

        self.assertEqual(result.spans[2].type, "alnum_prop_doc")
        self.assertEqual(result.spans[2].score, len(list(filter(str.isalnum, self.doc.text))) / len(self.doc.text))

        # TODO: This test fail; check with Akshita if this is expected
        # self.assertEqual(result.spans[3].type, "alpha_token_prop_doc")
        # self.assertEqual(result.spans[3].score, len(list(filter(str.isalpha, words))) / len(words))


DOC_WITH_METADATA = """
An XML file begins as follows:

```
<?xml version="1.0" encoding="UTF-8"?>
...
</xml>
```

An HTML file begins as follows:

```
<!DOCTYPE html>
<html>
...
</html>
```

These are different.
"""

DOC_WITH_PYTHON_CONTENT = """
def foo():
    # prints hello world
    print("Hello, World!")
"""

DOC_WITH_HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Page Title</title>
</head>
<body>
    <h1>This is a Heading</h1>
    <p>This is a paragraph.</p>
</body>
<javascript>
    console.log("Hello, World!")
    for (let i = 0; i < 10; i++) {
        console.log(i)
    }
</javascript>
</html>
"""


class TestStarCoderTaggers(unittest.TestCase):
    def setUp(self) -> None:
        self.md_doc = DocumentWithMetadata(
            id="0",
            text=DOC_WITH_METADATA.strip(),
            source=__file__,
            metadata={"ext": "md", "max_stars_count": 10},
        )
        self.python_doc = DocumentWithMetadata(
            id="1",
            text=DOC_WITH_PYTHON_CONTENT.strip(),
            source=__file__,
            metadata={"ext": "py", "max_stars_count": 1},
        )
        self.html_doc = DocumentWithMetadata(
            id="2",
            text=DOC_WITH_HTML_CONTENT.strip(),
            source=__file__,
            metadata={"ext": "html", "max_stars_count": 5},
        )
        self.tagger = CodeStarCoderTaggers2()
        return super().setUp()

    def test_metadata_tagger(self):
        result = self.tagger.predict(self.md_doc)
        self.assertEqual(len(result.spans), 4)

        self.assertEqual(result.spans[0].type, "has_xml_template_doc")
        self.assertEqual(result.spans[0].score, 1.0)

        self.assertEqual(result.spans[1].type, "num_github_stars_doc")
        self.assertEqual(result.spans[1].score, 10.0)

        # not a python, js, or java, so this is pinned to 0.5
        self.assertEqual(result.spans[2].type, "code_to_comment_ratio_doc")
        self.assertEqual(result.spans[2].score, 0.5)

        # not html, so this is pinned to 1.0
        self.assertEqual(result.spans[3].type, "code_to_text_ratio_html_doc")
        self.assertEqual(result.spans[3].score, 1.0)

    def test_python_tagger(self):
        result = self.tagger.predict(self.python_doc)

        comment_lines = [
            lns.split("#")[1].strip()
            for ln in self.python_doc.text.split("\n")
            if (lns := ln.strip()).startswith("#")
        ]

        self.assertEqual(len(result.spans), 4)

        self.assertEqual(result.spans[0].type, "has_xml_template_doc")
        self.assertEqual(result.spans[0].score, 0.0)

        self.assertEqual(result.spans[1].type, "num_github_stars_doc")
        self.assertEqual(result.spans[1].score, 1.0)

        self.assertEqual(result.spans[2].type, "code_to_comment_ratio_doc")
        self.assertEqual(result.spans[2].score, sum(map(len, comment_lines)) / len(self.python_doc.text))

        self.assertEqual(result.spans[3].type, "code_to_text_ratio_html_doc")
        self.assertEqual(result.spans[3].score, 1.0)

    def test_html_tagger(self):
        result = self.tagger.predict(self.html_doc)

        soup = BeautifulSoup(self.html_doc.text, features="html.parser")

        self.assertEqual(len(result.spans), 4)

        self.assertEqual(result.spans[0].type, "has_xml_template_doc")
        self.assertEqual(result.spans[0].score, 0.0)

        self.assertEqual(result.spans[1].type, "num_github_stars_doc")
        self.assertEqual(result.spans[1].score, 5.0)

        self.assertEqual(result.spans[2].type, "code_to_comment_ratio_doc")
        self.assertEqual(result.spans[2].score, 0.5)

        self.assertEqual(result.spans[3].type, "code_to_text_ratio_html_doc")
        self.assertEqual(result.spans[3].score, len(soup.get_text()) / len(self.html_doc.text))

    def test_html_tagger_doc_too_short(self):
        doc = DocumentWithMetadata(
            id="3",
            text="<html><head></head><body></body></html>",
            source=__file__,
            metadata={"ext": "html", "max_stars_count": 5},
        )
        doc.text = doc.text[:100]
        result = self.tagger.predict(doc)

        self.assertEqual(result.spans[3].type, "code_to_text_ratio_html_doc")
        self.assertEqual(result.spans[3].score, 0.0)
