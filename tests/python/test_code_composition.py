from unittest import TestCase

from dolma.core.data_types import Document
from dolma.taggers.code_composition import CodeProseCompositionClassifier

PROSE_TEXT = """
The Allen Institute for AI (abbreviated AI2) is a 501(c)(3) non-profit research institute founded by late Microsoft co-founder and philanthropist Paul Allen in 2014. The institute seeks to conduct high-impact AI research and engineering in service of the common good. Oren Etzioni was appointed by Paul Allenin September 2013 to direct the research at the institute. After leading the organization for nine years, Oren Etzioni stepped down from his role as CEO on September 30, 2022. He was replaced in an interim capacity by the leading researcher of the company's Aristo project, Peter Clark. On June 20, 2023, AI2 announced Ali Farhadi as its next CEO starting July 31, 2023. The company's board formed a search committee for a new CEO. AI2 also has an active office in Tel Aviv, Israel.
"""

CODE_TEXT = """
def foo():
    if True:
        print("Hello, world!")
"""

CODE_PROSE_TEXT = """
The following function adds two numbers together.
Then it returns the result.

def foo():
    x = 1 + 1
    return x

Next we demonstrate multiplying two numbers together.
Note that these are floats.
We return the result rounded to 2 decimal places.

def bar():
    x = 1.1 * 2.2
    return x

Finally, we show how to divide two numbers.

def baz():
    x = 1 / 2
    return x
"""


class TestDolmaCodeProseCompositionClassifier(TestCase):
    def setUp(self) -> None:
        self.code_composition_tagger = CodeProseCompositionClassifier()

    def test_prose_text(self):
        doc = Document(source="fixtures", id="1", text=PROSE_TEXT, version="v0")
        pred = self.code_composition_tagger.predict(doc)

        self.assertEqual(len(pred.spans), 4)
        self.assertEqual(
            {s.type for s in pred.spans},
            {"prose_mean_entropy", "code_prose_boundaries", "prose_composition", "prose_count"},
        )

        scores = {s.type: s.score for s in pred.spans}
        self.assertEqual(scores["code_prose_boundaries"], 0)
        self.assertEqual(scores["prose_composition"], 1)
        self.assertEqual(scores["prose_count"], 1)
        self.assertLess(scores["prose_mean_entropy"], 0.5)

    def test_code_text(self):
        doc = Document(source="fixtures", id="1", text=CODE_TEXT, version="v0")
        pred = self.code_composition_tagger.predict(doc)

        self.assertEqual(len(pred.spans), 4)
        self.assertEqual(
            {s.type for s in pred.spans},
            {"code_mean_entropy", "code_composition", "code_count", "code_prose_boundaries"},
        )

        scores = {s.type: s.score for s in pred.spans}
        self.assertEqual(scores["code_prose_boundaries"], 0)
        self.assertEqual(scores["code_composition"], 1)
        self.assertEqual(scores["code_count"], 3)
        self.assertLess(scores["code_mean_entropy"], 0.5)

    def test_code_prose_text(self):
        doc = Document(source="fixtures", id="1", text=CODE_PROSE_TEXT, version="v0")
        pred = self.code_composition_tagger.predict(doc)

        self.assertEqual(len(pred.spans), 7)
        self.assertEqual(
            {s.type for s in pred.spans},
            {
                "code_count",
                "prose_count",
                "prose_mean_entropy",
                "code_composition",
                "prose_composition",
                "code_prose_boundaries",
                "code_mean_entropy",
            },
        )

        scores = {s.type: s.score for s in pred.spans}
        self.assertEqual(scores["code_prose_boundaries"], 5)
        self.assertGreater(scores["code_composition"], 0.5)
        self.assertEqual(scores["code_count"], 9)
        self.assertLess(scores["code_mean_entropy"], 0.3)
