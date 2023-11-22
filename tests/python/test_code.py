"""

Unit tests for code taggers.

@soldni

"""

from unittest import TestCase
import unittest

from dolma.core.data_types import Document, DocumentWithMetadata
from dolma.taggers.code import (
    CodeCopyrightTagger,
    CodeRedPajamaTaggers,
    CodeSecretsTagger,
    CodeStarCoderTaggers,
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
