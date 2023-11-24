import unittest

from dolma.core.data_types import Document
from dolma.taggers.repetitions import ParagraphRepetitionsTagger, RepetitionsTagger

DOCUMENT_WITH_REPETITIONS = """
This is a text with repetitions.
This is a text with repetitions repetitions.
This is a text with repetitions repetitions repetitions.
This is a text with repetitions repetitions repetitions repetitions.

blah blah blah blah blah

No reps at the beginning of this sentence but ffffffffff

Seeing doubles: bass banana bass banana bass banana bass banana
"""

D0M0 = " repetitions repetitions repetitions repetitions"
D0M1 = "blah blah blah blah blah"
D0M2 = "ffffffffff"
D0M3 = " bass banana bass banana bass banana bass banana"

DOCUMENT_WITHOUT_REPETITIONS = """
This is a text without repetitions.

Absolutely no repetitions here.

There are double letters here and there but those don't count.
"""


class TestRepetitionsTagger(unittest.TestCase):
    def setUp(self) -> None:
        self.doc_with_reps = Document(source=__file__, id="0", text=DOCUMENT_WITH_REPETITIONS)
        self.doc_without_reps = Document(source=__file__, id="1", text=DOCUMENT_WITHOUT_REPETITIONS)

        self.repetitions_tagger = RepetitionsTagger()
        self.para_reps_tagger = ParagraphRepetitionsTagger()

        return super().setUp()

    def test_doc_with_repetitions(self):
        all_result = self.repetitions_tagger.predict(self.doc_with_reps)
        par_result = self.para_reps_tagger.predict(self.doc_with_reps)
        self.assertEqual(len(all_result.spans), 7)
        self.assertEqual(len(par_result.spans), 7)

        self.assertEqual(all_result.spans[0].type, "char_repetition")
        self.assertEqual(all_result.spans[0].select(self.doc_with_reps), D0M0)
        self.assertEqual(all_result.spans[0].score, len(D0M0))
        self.assertEqual(all_result.spans[0], par_result.spans[0])

        self.assertEqual(all_result.spans[1].type, "char_repetition")
        self.assertEqual(all_result.spans[1].select(self.doc_with_reps), D0M1)
        self.assertEqual(all_result.spans[1].score, len(D0M1))
        self.assertEqual(all_result.spans[1], par_result.spans[1])

        self.assertEqual(all_result.spans[2].type, "char_repetition")
        self.assertEqual(all_result.spans[2].select(self.doc_with_reps), D0M2)
        self.assertEqual(all_result.spans[2].score, len(D0M2))
        self.assertEqual(all_result.spans[2], par_result.spans[2])

        self.assertEqual(all_result.spans[3].type, "char_repetition")
        self.assertEqual(all_result.spans[3].select(self.doc_with_reps), D0M3)
        self.assertEqual(all_result.spans[3].score, len(D0M3))
        self.assertEqual(all_result.spans[3], par_result.spans[3])

        self.assertEqual(all_result.spans[4].type, "doc_max_char_repetition")
        self.assertEqual(all_result.spans[4].score, len(D0M0))
        self.assertEqual(all_result.spans[4], par_result.spans[4])

        matches_length = len(D0M0) + len(D0M1) + len(D0M2) + len(D0M3)
        self.assertEqual(all_result.spans[5].type, "doc_mean_char_repetition")
        self.assertEqual(all_result.spans[5].score, matches_length / 4)
        self.assertEqual(all_result.spans[5], par_result.spans[5])

        self.assertEqual(all_result.spans[6].type, "doc_frac_char_repetition")
        self.assertEqual(all_result.spans[6].score, matches_length / len(self.doc_with_reps.text))
        self.assertEqual(all_result.spans[6], par_result.spans[6])

    def test_doc_without_repetitions(self):
        all_result = self.repetitions_tagger.predict(self.doc_without_reps)
        par_result = self.para_reps_tagger.predict(self.doc_without_reps)
        self.assertEqual(len(all_result.spans), 3)
        self.assertEqual(len(par_result.spans), 3)

        self.assertEqual(all_result.spans[0].type, "doc_max_char_repetition")
        self.assertEqual(all_result.spans[0].score, 0)
        self.assertEqual(all_result.spans[0], par_result.spans[0])

        self.assertEqual(all_result.spans[1].type, "doc_mean_char_repetition")
        self.assertEqual(all_result.spans[1].score, 0)
        self.assertEqual(all_result.spans[1], par_result.spans[1])

        self.assertEqual(all_result.spans[2].type, "doc_frac_char_repetition")
        self.assertEqual(all_result.spans[2].score, 0)
        self.assertEqual(all_result.spans[2], par_result.spans[2])
