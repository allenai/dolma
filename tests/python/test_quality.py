from pathlib import Path
from unittest import TestCase

import smart_open

from dolma.core.data_types import Document
from dolma.taggers.quality import Dolma17QualityClassifier

WIKIPEDIA_TEXT = """
The Allen Institute for AI (abbreviated AI2) is a 501(c)(3) non-profit research institute founded by late Microsoft co-founder and philanthropist Paul Allen in 2014. The institute seeks to conduct high-impact AI research and engineering in service of the common good. Oren Etzioni was appointed by Paul Allenin September 2013 to direct the research at the institute. After leading the organization for nine years, Oren Etzioni stepped down from his role as CEO on September 30, 2022. He was replaced in an interim capacity by the leading researcher of the company's Aristo project, Peter Clark. On June 20, 2023, AI2 announced Ali Farhadi as its next CEO starting July 31, 2023. The company's board formed a search committee for a new CEO. AI2 also has an active office in Tel Aviv, Israel.
"""

CREATIVE_COMMONS_BLOG_TEXT = """
The CC team has been evaluating our progress toward our 2021-2025 strategy. Through that process, we have noticed the ways we have been organically adjusting to the social and technical shifts around us, as well as the ebbs and flows of funding availability. It would be an understatement to say that much has changed since we developed the strategy in 2020 and launched it in 2021. Turns out that our predictions and plans set forth in 2020 are not as helpful in the reality of 2024 and likely even less so for 2025 and beyond. Rather than continuing to progress through the existing strategy, we have determined that the stronger, and dare we say more strategic, approach is to conduct a strategy refresh.
"""

LOW_QUALITY_TEXT = smart_open.open(
    Path(__file__).parent.parent.parent / "python/dolma/data/naughty_words_en.txt"
).read()


class TestDolma17QualityClassifier(TestCase):
    def setUp(self) -> None:
        self.quality_tagger = Dolma17QualityClassifier()

    def test_wikipedia_text(self):
        doc = Document(source="wikipedia", id="1", text=WIKIPEDIA_TEXT, version="v0")
        pred = self.quality_tagger.predict(doc)
        self.assertEqual(len(pred.spans), 2)
        self.assertEqual({s.type for s in pred.spans}, {"hq", "lq"})

        scores = {s.type: s.score for s in pred.spans}
        self.assertGreater(scores["hq"], 0.25)
        self.assertAlmostEqual(sum(scores.values()), 1.0, delta=0.01)

    def test_creative_commons_blog_text(self):
        doc = Document(source="creative_commons", id="1", text=CREATIVE_COMMONS_BLOG_TEXT, version="v0")
        pred = self.quality_tagger.predict(doc)
        self.assertEqual(len(pred.spans), 2)
        self.assertEqual({s.type for s in pred.spans}, {"hq", "lq"})

        scores = {s.type: s.score for s in pred.spans}
        self.assertGreater(scores["hq"], 0.25)
        self.assertAlmostEqual(sum(scores.values()), 1.0, delta=0.01)

    def test_low_quality_text(self):
        doc = Document(source="low_quality", id="1", text=LOW_QUALITY_TEXT, version="v0")
        pred = self.quality_tagger.predict(doc)
        self.assertEqual(len(pred.spans), 2)
        self.assertEqual({s.type for s in pred.spans}, {"hq", "lq"})

        scores = {s.type: s.score for s in pred.spans}
        self.assertGreater(scores["lq"], 0.25)
        self.assertAlmostEqual(sum(scores.values()), 1.0, delta=0.01)
