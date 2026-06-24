"""Unit tests for dolma.taggers.punctuation."""

import time
from unittest import TestCase

from dolma.core.data_types import Document
from dolma.taggers.punctuation import NotAlphanumParagraphV1


class TestNotAlphanumParagraphV1(TestCase):
    def setUp(self) -> None:
        self.tagger = NotAlphanumParagraphV1()

    def test_long_emoji_paragraph_completes_quickly(self) -> None:
        emoji = (
            "😄 😃 😊 😉 😍 😚 😗 😜 😛 😳 😁 😬 😌 😞 😢 😂 😭 😅 😓 😩 😮 😱 "
            "😠 😡 😤 😋 😎 😴 😈 😇 😕 😏 😑 👲 👮 💂 👶 ❤ 💔 💕 💘 💌 💋 🎁 "
            "💰 💍 👍 👎 👌 ✌️ 🤘 👏 🎵 ☕️ 🍵 🍺 🍷 🍼 ☀️ 🌤 🌦 🌧 🌜 🌈 🏝 🎅"
        )
        doc = Document(source="", version="", id="", text=f"\nGuestbook entry:\n{emoji}\n")

        start = time.monotonic()
        result = self.tagger.predict(doc=doc)
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 1.0, "Tagger should not hang on long emoji paragraphs")
        self.assertTrue(
            any(span.type == "all_punct" and span.score == 1.0 for span in result.spans),
        )

    def test_alphanumeric_paragraph_not_flagged(self) -> None:
        doc = Document(source="", version="", id="", text="Hello world.")
        result = self.tagger.predict(doc=doc)
        self.assertEqual(
            [span.to_json() for span in result.spans if span.type == "all_punct" and span.score == 1.0],
            [],
        )
