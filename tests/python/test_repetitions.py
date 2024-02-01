import unittest

from dolma.core.data_types import Document
from dolma.taggers.repetitions import (
    ParagraphRepetitionsTagger,
    RepetitionsTagger,
    TokenizerRepetitionsSkipEmptyTagger,
    TokenizerRepetitionsTagger,
)

DOCUMENT_WITH_REPETITIONS = """
This is a text with repetitions.
This is a text with repetitions repetitions.
This is a text with repetitions repetitions repetitions.
This is a text with repetitions repetitions repetitions repetitions.

blah blah blah blah blah

No reps at the beginning of this sentence but MMMMMMMMMM

Seeing doubles: bass banana bass banana bass banana bass banana
"""

D0M0 = " repetitions repetitions repetitions repetitions"
D0M1 = "blah blah blah blah blah"
D0M2 = "MMMMMMMMMM"
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

        self.assertEqual(all_result.spans[0].type, "repetition")
        self.assertEqual(all_result.spans[0].select(self.doc_with_reps), D0M0)
        self.assertEqual(all_result.spans[0].score, D0M0.count("repetition"))
        self.assertEqual(all_result.spans[0], par_result.spans[0])

        self.assertEqual(all_result.spans[1].type, "repetition")
        self.assertEqual(all_result.spans[1].select(self.doc_with_reps), D0M1)
        self.assertEqual(all_result.spans[1].score, D0M1.count("blah"))
        self.assertEqual(all_result.spans[1], par_result.spans[1])

        self.assertEqual(all_result.spans[2].type, "repetition")
        self.assertEqual(all_result.spans[2].select(self.doc_with_reps), D0M2)
        self.assertEqual(all_result.spans[2].score, D0M2.count("M"))
        self.assertEqual(all_result.spans[2], par_result.spans[2])

        self.assertEqual(all_result.spans[3].type, "repetition")
        self.assertEqual(all_result.spans[3].select(self.doc_with_reps), D0M3)
        self.assertEqual(all_result.spans[3].score, D0M3.count("bass"))
        self.assertEqual(all_result.spans[3], par_result.spans[3])

        self.assertEqual(all_result.spans[4].type, "doc_max_score_repetition")
        self.assertEqual(all_result.spans[4].score, D0M2.count("M"))
        self.assertEqual(all_result.spans[4], par_result.spans[4])

        self.assertEqual(all_result.spans[5].type, "doc_max_length_repetition")
        self.assertEqual(all_result.spans[5].score, len(D0M0))
        self.assertEqual(all_result.spans[5], par_result.spans[5])

        matches_length = len(D0M0) + len(D0M1) + len(D0M2) + len(D0M3)
        self.assertEqual(all_result.spans[6].type, "doc_frac_repetition")
        self.assertEqual(all_result.spans[6].score, matches_length / len(self.doc_with_reps.text))
        self.assertEqual(all_result.spans[6], par_result.spans[6])

    def test_doc_without_repetitions(self):
        all_result = self.repetitions_tagger.predict(self.doc_without_reps)
        par_result = self.para_reps_tagger.predict(self.doc_without_reps)
        self.assertEqual(len(all_result.spans), 3)
        self.assertEqual(len(par_result.spans), 3)

        self.assertEqual(all_result.spans[0].type, "doc_max_score_repetition")
        self.assertEqual(all_result.spans[0].score, 0)
        self.assertEqual(all_result.spans[0], par_result.spans[0])

        self.assertEqual(all_result.spans[1].type, "doc_max_length_repetition")
        self.assertEqual(all_result.spans[1].score, 0)
        self.assertEqual(all_result.spans[1], par_result.spans[1])

        self.assertEqual(all_result.spans[2].type, "doc_frac_repetition")
        self.assertEqual(all_result.spans[2].score, 0)
        self.assertEqual(all_result.spans[2], par_result.spans[2])


class TestTokenizerRepetitionsTagger(unittest.TestCase):
    def setUp(self) -> None:
        self.doc_with_reps = Document(source=__file__, id="0", text=DOCUMENT_WITH_REPETITIONS)
        self.doc_without_reps = Document(source=__file__, id="1", text=DOCUMENT_WITHOUT_REPETITIONS)
        self.repetitions_tagger = TokenizerRepetitionsTagger()

        return super().setUp()

    def test_doc_with_repetitions(self):
        repeated_strings = [
            ("repetitions repetitions repetitions", 3),
            ("repetitions repetitions repetitions repetitions", 4),
            ("blah blah blah blah", 4),  # missing a blah bc the first element in this seq has diff token id
            ("MMMMMMMM", 4),  # shorter bc sequence is tokenized as 'ĠM', 'MM', 'MM', 'MM', 'MM', 'M'
            ("bass banana bass banana bass banana bass banana", 4),
        ]

        all_results = self.repetitions_tagger.predict(self.doc_with_reps)
        self.assertEqual(len(all_results.spans), len(repeated_strings) + 3)

        i = 0
        for string, score in repeated_strings:
            self.assertEqual(all_results.spans[i].type, "repetition")
            self.assertEqual(string, all_results.spans[i].select(self.doc_with_reps))
            self.assertEqual(all_results.spans[i].score, score)
            i += 1

        self.assertEqual(all_results.spans[i].type, "doc_max_score_repetition")
        self.assertEqual(all_results.spans[i].score, max(v for _, v in repeated_strings))
        i += 1

        matches_length = sum(len(s) for s, _ in repeated_strings)
        self.assertEqual(all_results.spans[i].type, "doc_max_length_repetition")
        self.assertEqual(all_results.spans[i].score, max(len(s) for s, _ in repeated_strings))
        i += 1

        self.assertEqual(all_results.spans[i].type, "doc_frac_repetition")
        self.assertEqual(all_results.spans[i].score, matches_length / len(self.doc_with_reps.text))

    def test_multiple_matches(self):
        text = "NOOOOOOOOOOOOOO If it is a Pizza Oven, then it's first meal MUST be PIZZA!!!!!! otherwise it will fall apart!!!!!"
        doc = Document(source=__file__, id="0", text=text)

        rep_tagger = TokenizerRepetitionsTagger()
        rep_tagger_uniq = TokenizerRepetitionsSkipEmptyTagger()

        all_results = rep_tagger.predict(doc)
        uniq_results = rep_tagger_uniq.predict(doc)

        self.assertEqual(len(all_results.spans), 5)
        self.assertEqual(len(uniq_results.spans), 4)

        self.assertEqual(all_results.spans[0].start, 1)
        self.assertEqual(all_results.spans[0].end, 15)
        self.assertEqual(all_results.spans[0].score, 7)

        self.assertEqual(all_results.spans[1].start, 1)
        self.assertEqual(all_results.spans[1].end, 15)
        self.assertEqual(all_results.spans[1].score, 3)

        self.assertEqual(uniq_results.spans[0].start, 1)
        self.assertEqual(uniq_results.spans[0].end, 15)
        self.assertEqual(uniq_results.spans[0].score, 7)

    def test_skip_empty(self):
        text = "Nothing to note."
        doc = Document(source=__file__, id="0", text=text)

        rep_tagger = TokenizerRepetitionsTagger()
        rep_tagger_uniq = TokenizerRepetitionsSkipEmptyTagger()

        all_results = rep_tagger.predict(doc)
        uniq_results = rep_tagger_uniq.predict(doc)

        self.assertEqual(len(all_results.spans), 3)
        self.assertEqual(len(uniq_results.spans), 0)
