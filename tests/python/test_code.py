"""

Unit tests for code taggers.

@soldni

"""

import base64
import re
import unittest
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup

from dolma.core.data_types import Document, DocumentWithMetadata
from dolma.taggers.code import (
    CodeCopyrightTagger,
    CodeRedPajamaTaggers,
    CodeSecretsTagger,
    CodeStarCoderTaggers2,
    Learn2CodeTaggers,
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


class TestLearn2CodeTaggers(unittest.TestCase):
    def mk_doc(self, id="0", text="asdf", source=__file__, metadata_overrides=None) -> DocumentWithMetadata:
        return DocumentWithMetadata(
            id=id,
            text=text,
            source=source,
            metadata={
                "ext": "py",
                "path": "/some/path/to/this/file.py",
                **(metadata_overrides or {}),
            },
        )

    def assert_doc_span_score(
        self, doc: DocumentWithMetadata, span_type: str, span_score: float, msg: Optional[str] = None
    ) -> None:
        for span in doc.spans:
            if span.type is span_type:
                self.assertEqual(span.score, span_score, msg=msg)
                break
        else:
            raise AssertionError(f"No span of type `{span_type}`")

    def test_doc_length(self) -> None:
        tagger = Learn2CodeTaggers()
        doc = self.mk_doc(text="a" * 9001)
        tagged_doc = tagger.predict(doc)
        self.assert_doc_span_score(tagged_doc, "num_chars_doc", 9001)

    def test_gh_star_tagging(self) -> None:
        tagger = Learn2CodeTaggers()

        doc1 = self.mk_doc(metadata_overrides={"max_stars_count": 9000})
        tagged_doc1 = tagger.predict(doc1)
        self.assert_doc_span_score(tagged_doc1, "num_github_stars_doc", 9000)

        doc2 = self.mk_doc(metadata_overrides={"star_events_count": 9000})
        tagged_doc2 = tagger.predict(doc2)
        self.assert_doc_span_score(tagged_doc2, "num_github_stars_doc", 9000)

    def test_proportion_alpha_doc(self) -> None:
        tagger = Learn2CodeTaggers()

        texts_percentage_pairs = [
            ("12345!@#%$", 0.0),
            ("f234567890", 0.1),
            ("f2a4567890", 0.2),
            ("fDSa5j7890", 0.5),
            ("ABcdEFgHij", 1.0),
        ]

        for doc_text, expected_percent_alpha in texts_percentage_pairs:
            doc = self.mk_doc(text=doc_text)
            tagged_doc = tagger.predict(doc)
            self.assert_doc_span_score(tagged_doc, "proportion_alpha_doc", expected_percent_alpha)

    def test_has_xml_template_doc(self) -> None:
        tagger = Learn2CodeTaggers()

        not_xml = "a" * 100 + "<?xml version="  # doesn't occur in first hundred characters
        not_xml_doc = self.mk_doc(text=not_xml)
        tagged_not_xml_doc = tagger.predict(not_xml_doc)
        self.assert_doc_span_score(tagged_not_xml_doc, "has_xml_template_doc", 0.0)

        is_xml = '<?xml version="1.0" encoding="UTF-8"?>\n<iamtag></iamtag>'
        is_xml_doc = self.mk_doc(text=is_xml)
        tagged_is_xml_doc = tagger.predict(is_xml_doc)
        self.assert_doc_span_score(tagged_is_xml_doc, "has_xml_template_doc", 1.0)

    def test_line_stats(self) -> None:
        tagger = Learn2CodeTaggers()

        lines = ["1" * 50, "2" * 150, "3" * 100]
        text = "\n".join(lines)

        doc = self.mk_doc(text=text)
        tagged_doc = tagger.predict(doc)
        self.assert_doc_span_score(tagged_doc, "num_lines_doc", 3)
        self.assert_doc_span_score(tagged_doc, "mean_line_length_doc", 100.0)
        self.assert_doc_span_score(tagged_doc, "max_line_length_doc", 150)

    def test_b64_encoding_tagging(self) -> None:
        tagger = Learn2CodeTaggers()

        to_encode = b"hello i am some text that is sufficiently long to encode and trigger the regexes we have"
        base64_encoded_substring = base64.b64encode(to_encode).decode("utf-8")
        full_text = (
            " " * len(base64_encoded_substring)
            + base64_encoded_substring
            + " " * 2 * len(base64_encoded_substring)
        )

        doc = self.mk_doc(text=full_text)
        tagged_doc = tagger.predict(doc)
        self.assert_doc_span_score(tagged_doc, "longest_seq_b64_doc", len(base64_encoded_substring))
        self.assert_doc_span_score(tagged_doc, "proportion_b64_doc", 0.25)

    def test_hexadecimal_filter(self) -> None:
        tagger = Learn2CodeTaggers()

        hex_examples = [
            ("c_style_array", "0x1A,0x2B,0x3C,0x4D,0x5E,0x6F,0x7A,0x8B "),
            ("hex_dump", "1A 2B 3C 4D 5E 6F 7A 8B "),
        ]

        for name, example in hex_examples:
            full_text = "a" * (len(example) - 1) + " " + example + "z" * 2 * len(example)
            doc = self.mk_doc(text=full_text)
            tagged_doc = tagger.predict(doc)
            self.assert_doc_span_score(
                tagged_doc, "longest_seq_hexadecimal_doc", len(example), f"{name} is wrong length"
            )
            self.assert_doc_span_score(tagged_doc, "proportion_hexadecimal_doc", 0.25, f"{name} wrong percentage")

    def test_unicode_filter(self) -> None:
        tagger = Learn2CodeTaggers()
        unicode = "\\u0041\\u0042\\u0043\\u0044\\u0045\\u0046\\u0047\\u0048"
        full_text = " " * len(unicode) + unicode + " " * len(unicode) + unicode
        doc = self.mk_doc(text=full_text)
        tagged_doc = tagger.predict(doc)
        self.assert_doc_span_score(tagged_doc, "longest_seq_unicode_doc", len(unicode))
        self.assert_doc_span_score(tagged_doc, "proportion_unicode_doc", 0.5)

    def test_proportion_comments_doc(self) -> None:
        tagger = Learn2CodeTaggers()
        text = """
        def thingy(asdf):
            '''I am a docstring'''
            return asdf
        """.strip()

        expected_comment_length = len("I am a docstring")

        doc = self.mk_doc(text=text, metadata_overrides={"ext": "py"})
        tagged_doc = tagger.predict(doc)
        self.assert_doc_span_score(tagged_doc, "proportion_comments_doc", expected_comment_length / len(text))

    def test_proportion_text_in_html(self) -> None:
        tagger = Learn2CodeTaggers()
        visible_text = "I am the visible text" * 20
        text = f"""<html><body><div>{visible_text}</div></body></html>""".strip()

        expected_visible_text_length = len(visible_text)
        doc = self.mk_doc(text=text, metadata_overrides={"ext": "html"})
        tagged_doc = tagger.predict(doc)
        self.assert_doc_span_score(
            tagged_doc, "proportion_text_in_html_doc", expected_visible_text_length / len(text)
        )

    def test_is_special_text_file_doc(self) -> None:
        tagger = Learn2CodeTaggers()

        special_ones = [
            "asdf/requirement.txt",
            "asdf/requirements.txt",
            "asdf/dev_requirements.txt",
            "asdf/readme.txt",
            "asdf/todo.txt",
            "asdf/Description.txt",
            "asdf/CMAKELISTS.txt",
        ]

        not_so_special_ones = ["asdf/birthdays.txt", "asdf/readme.md", "readme"]

        for special_one in special_ones:
            extension = Path(special_one).suffix.replace(".", "")
            doc = self.mk_doc(special_one, metadata_overrides={"path": special_one, "ext": extension})
            tagged_doc = tagger.predict(doc)
            self.assert_doc_span_score(
                tagged_doc, "is_special_text_file_doc", 1, msg=f"`{special_one}` wasn't so special"
            )

        for not_special in not_so_special_ones:
            extension = Path(not_special).suffix.replace(".", "")
            doc = self.mk_doc(not_special, metadata_overrides={"path": not_special, "ext": extension})
            tagged_doc = tagger.predict(doc)
            self.assert_doc_span_score(
                tagged_doc, "is_special_text_file_doc", 0, msg=f"`{not_special}` was seen as special"
            )
