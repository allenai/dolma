import unittest
from typing import Callable, Dict, List, Optional, Tuple, Type

from dolma.core import BaseTagger, Document, Span
from dolma.taggers.language import (
    Cld2EnglishLanguageParagraphTagger,
    Cld2EnglishLanguageTagger,
    Cld2LanguageFilterParagraphWithDocScoreTagger,
    Cld2LanguageTagger,
    Cld2LanguageTaggerParagraph,
    FastTextAllLanguageParagraphTagger,
    FastTextAllLanguagesDocumentTagger,
    FastTextEnglishLanguageDocumentTagger,
    FastTextEnglishLanguageParagraphTagger,
    FastTextEnglishLanguageParagraphWithDocScoreTagger,
    LangdetectEnglishTagger,
    LangdetectEnglishTaggerParagraph,
    LinguaEnglishTagger,
    LinguaEnglishTaggerParagraph,
    LinguaTagger,
    LinguaTaggerParagraph,
)

ENGLISH_PARAGRAPH = """
English is a West Germanic language in the Indo-European language family, whose speakers, called Anglophones, originated in early medieval England. The namesake of the language is the Angles, one of the ancient Germanic peoples that migrated to the island of Great Britain. Modern English is both the most spoken language in the world and the third-most spoken native language, after Mandarin Chinese and Spanish. It is also the most widely learned second language in the world, with more second-language speakers than native speakers. English is either the official language or one of the official languages in 59 sovereign states (such as in India, Ireland, and Canada). In some other countries, it is the sole or dominant language for historical reasons without being explicitly defined by law (such as in the United States or United Kingdom). It is a co-official language of the United Nations, the European Union, and many other international and regional organizations. It has also become the de facto language of diplomacy, science, international trade, tourism, aviation, entertainment and the internet. English accounts for at least 70% of total speakers of the Germanic language branch, and as of 2005, it was estimated that there were over two billion speakers worldwide.
""".strip()

FRENCH_PARAGRAPH = """
Le français est une langue indo-européenne de la famille des langues romanes dont les locuteurs sont appelés francophones. Elle est parfois surnommée la «langue de Molière». Le français est parlé, en 2023, sur tous les continents par environ 321 millions de personnes: 235 millions l'emploient quotidiennement et 81 millions en sont des locuteurs natifs. En 2018, 80 millions d'élèves et étudiants s'instruisent en français dans le monde. Selon l'Organisation internationale de la francophonie (OIF), il pourrait y avoir 700 millions de francophones sur Terre en 2050. Le français est la cinquième langue parlée au monde après l'anglais, le mandarin, le hindi et l'espagnol. Elle est également la deuxième langue apprise sur le globe, la troisième langue des affaires et du commerce, la quatrième langue employée sur Internet. Le français se classe deuxième parmi les langues étrangères les plus fréquemment enseignées à travers le monde. Il est également la quatrième langue utilisée sur internet après l'espagnol, le mandarin et l'anglais, langue dont le vocabulaire a été fortement enrichi par le français.
""".strip()

ITALIAN_PARAGRAPH = """
L'italiano è una lingua romanza parlata principalmente in Italia. Per ragioni storiche e geografiche, l'italiano è la lingua romanza meno divergente dal latino (complessivamente a pari merito, anche se in parametri diversi, con la lingua sarda). L'italiano è classificato al 23º posto tra le lingue per numero di parlanti nel mondo e, in Italia, è utilizzato da circa 58 milioni di residenti.[6] Nel 2015 era la lingua materna del 90,4% dei residenti in Italia, che spesso lo acquisiscono e lo usano insieme alle varianti regionali dell'italiano, alle lingue regionali e ai dialetti. In Italia viene ampiamente usato per tutti i tipi di comunicazione della vita quotidiana e prevale largamente nei mezzi di comunicazione nazionali, nell'amministrazione pubblica dello Stato italiano e nell'editoria. Oltre ad essere la lingua ufficiale dell'Italia, è anche una delle lingue ufficiali dell'Unione europea, di San Marino, della Svizzera, della Città del Vaticano e del Sovrano militare ordine di Malta. È inoltre riconosciuto e tutelato come «lingua della minoranza nazionale italiana» dalla Costituzione slovena e croata nei territori in cui vivono popolazioni di dialetto istriano. È diffuso nelle comunità di emigrazione italiana, è ampiamente noto anche per ragioni pratiche in diverse aree geografiche ed è una delle lingue straniere più studiate nel mondo.
""".strip()

JAPANESE_PARAGRAPH = """
日本語 は、日本国内や、かつての日本領だった国、そして国外移民や移住者を含む日本人同士の間で使用されている言語。日本は法令によって公用語を規定していないが、法令その他の公用文は全て日本語で記述され、各種法令において日本語を用いることが規定され、学校教育においては「国語」の教科として学習を行うなど、事実上日本国内において唯一の公用語となっている。使用人口について正確な統計はないが、日本国内の人口、及び日本国外に住む日本人や日系人、日本がかつて統治した地域の一部住民など、約1億3,000万人以上と考えられている。統計によって前後する場合もあるが、この数は世界の母語話者数で上位10位以内に入る人数である。また第一次世界大戦後、日本に委任統治 されていたパラオでは、現在も一部地域で日本語を公用語と定めている。日本語の音韻は、「っ」「ん」を除いて母音で終わる開音節言語の性格が強く、また標準語（共通語）を含め多くの方言がモーラを持つ。アクセントは高低アクセントである。日本語は、主に日本国内で使用される。話者人口についての調査は国内・国外を問わずいまだないが、日本の人口に基づいて考えられることが一般的である。「日本語」の範囲を本土方言のみとした場合、琉球語が日本語と同系統の言語になり両者は日琉語族を形成する。琉球列島（旧琉球王国領域）の言葉は、日本語と系統を同じくする別言語（琉球語ないしは琉球諸語）とし、日本語とまとめて日琉語族とされている。共通点が多いので「日本語の一方言（琉球方言）」とする場合もあり、このような場合は日本語は「孤立した言語」という位置づけにされる。アルタイ諸語に属するとする説は、明治時代末から特に注目されてきた。その根拠として、古代の日本語（大和言葉）において語頭にr音（流音）が立たないこと、一種の母音調和が見られることなどが挙げられる。古代日本語に上記の特徴が見られることは、日本語が類型として「アルタイ型」の言語である根拠とされる。アルタイ諸語に属するとされるそれぞれの言語の親族関係を支持する学者のほうがまだ多いが、最近のイギリスではアルタイ諸語の親族関係を否定する学者も現れている。
""".strip()


class BaseEnglishTaggerTest:
    doc_tagger_cls: Type[BaseTagger]
    par_tagger_cls: Type[BaseTagger]
    par_tagger_w_doc_score_cls: Optional[Type[BaseTagger]] = None

    assertEqual: Callable
    assertGreater: Callable
    assertLess: Callable

    def setUp(self) -> None:
        self.doc_tagger = self.doc_tagger_cls()
        self.par_tagger = self.par_tagger_cls()
        self.par_tagger_w_doc_score = (
            self.par_tagger_w_doc_score_cls() if self.par_tagger_w_doc_score_cls else None
        )

        self.single_paragraph_docs = [
            Document(text=ENGLISH_PARAGRAPH, id="en", source=__file__),
            Document(text=FRENCH_PARAGRAPH, id="fr", source=__file__),
            Document(text=ITALIAN_PARAGRAPH, id="it", source=__file__),
            Document(text=JAPANESE_PARAGRAPH, id="ja", source=__file__),
        ]

        self.multi_paragraph_docs = [
            Document(text=f"{ENGLISH_PARAGRAPH}\n{JAPANESE_PARAGRAPH}", id="en_ja", source=__file__),
            Document(text=f"{FRENCH_PARAGRAPH}\n{ITALIAN_PARAGRAPH}", id="fr_it", source=__file__),
        ]

    def test_document(self):
        for doc in self.single_paragraph_docs:
            result = self.doc_tagger.predict(doc)
            try:
                self.assertEqual(len(result.spans), 2)
                self.assertEqual(sorted(t.type for t in result.spans), ["en", "not_en"])

                span = max(result.spans, key=lambda s: s.score)
                if doc.id == "en":
                    self.assertEqual(span.type, "en")
                else:
                    self.assertEqual(span.type, "not_en")
                self.assertEqual(sum(s.score for s in result.spans), 1.0)
            except AssertionError:
                breakpoint()

    def test_paragraph(self):
        for doc in self.multi_paragraph_docs:
            result = self.par_tagger.predict(doc)
            self.assertEqual(len(result.spans), 4)

            english_matches = [s for s in result.spans if s.type == "en"]
            non_en_matches = [s for s in result.spans if s.type == "not_en"]
            self.assertEqual(len(english_matches), 2)
            self.assertEqual(len(non_en_matches), 2)

            grouped_spans = {}
            for span in result.spans:
                grouped_spans.setdefault((span.start, span.end), []).append(span)

            # make sure 2 paragraphs are found
            self.assertEqual(len(grouped_spans), 2)

            # check that the scores sum to 1.0.
            self.assertEqual(all(sum(s.score for s in spans) == 1.0 for spans in grouped_spans.values()), True)

            for paragraph_spans, paragraph_language in zip(
                sorted(grouped_spans.values(), key=lambda spans: spans[0].start), doc.id.split("_")
            ):
                expected_order = ["en", "not_en"] if paragraph_language == "en" else ["not_en", "en"]
                actual_order = [s.type for s in sorted(paragraph_spans, key=lambda s: -s.score)]
                self.assertEqual(actual_order, expected_order)

    def test_paragraph_with_doc_score(self):
        if self.par_tagger_w_doc_score is None:
            return

        for doc in self.multi_paragraph_docs:
            results_regular = self.par_tagger.predict(doc)
            results_with_doc_score = self.par_tagger_w_doc_score.predict(doc)

            self.assertEqual(len(results_with_doc_score.spans), 6)
            for span_regular, span_with_doc_score in zip(results_regular.spans, results_with_doc_score.spans):
                self.assertEqual(span_regular, span_with_doc_score)

            expected_doc_score = sum(
                (s.end - s.start) * s.score for s in results_regular.spans if s.type == "en"
            ) / len(results_regular.doc.text)
            self.assertEqual(results_with_doc_score.spans[4].score, expected_doc_score)
            self.assertEqual(results_with_doc_score.spans[5].score, 1 - results_with_doc_score.spans[4].score)


class BaseMultilingualTaggerTest(BaseEnglishTaggerTest):
    def test_document(self):
        for doc in self.single_paragraph_docs:
            result = self.doc_tagger.predict(doc)
            best_lang = max(result.spans, key=lambda s: s.score)
            self.assertEqual(best_lang.type, doc.id)
            self.assertGreater(best_lang.score, 0.7)

    def test_paragraph(self):
        for doc in self.multi_paragraph_docs:
            result = self.par_tagger.predict(doc)
            languages = doc.id.split("_")

            group_by_paragraph: Dict[Tuple[int, int], List[Span]] = {}
            for span in result.spans:
                group_by_paragraph.setdefault((span.start, span.end), []).append(span)

            self.assertEqual(len(group_by_paragraph), 2)

            for spans, lang in zip(group_by_paragraph.values(), languages):
                best_lang = max(spans, key=lambda s: s.score)
                self.assertGreater(best_lang.score, 0.7)
                self.assertEqual(best_lang.type, lang)

    def test_paragraph_with_doc_score(self):
        return


class TestCld2AllLanganuges(BaseMultilingualTaggerTest, unittest.TestCase):
    doc_tagger_cls = Cld2LanguageTagger
    par_tagger_cls = Cld2LanguageTaggerParagraph


class TestPyCld2(BaseEnglishTaggerTest, unittest.TestCase):
    doc_tagger_cls = Cld2EnglishLanguageTagger
    par_tagger_cls = Cld2EnglishLanguageParagraphTagger
    par_tagger_w_doc_score_cls = Cld2LanguageFilterParagraphWithDocScoreTagger


class TestFastText(BaseEnglishTaggerTest, unittest.TestCase):
    doc_tagger_cls = FastTextEnglishLanguageDocumentTagger
    par_tagger_cls = FastTextEnglishLanguageParagraphTagger
    par_tagger_w_doc_score_cls = FastTextEnglishLanguageParagraphWithDocScoreTagger


class TestFastTextAllLanguages(BaseMultilingualTaggerTest, unittest.TestCase):
    doc_tagger_cls = FastTextAllLanguagesDocumentTagger
    par_tagger_cls = FastTextAllLanguageParagraphTagger


class TestLangdetect(BaseEnglishTaggerTest, unittest.TestCase):
    doc_tagger_cls = LangdetectEnglishTagger
    par_tagger_cls = LangdetectEnglishTaggerParagraph


class TestLingua(BaseMultilingualTaggerTest, unittest.TestCase):
    doc_tagger_cls = LinguaTagger
    par_tagger_cls = LinguaTaggerParagraph


class TestLinguaEnglish(BaseEnglishTaggerTest, unittest.TestCase):
    doc_tagger_cls = LinguaEnglishTagger
    par_tagger_cls = LinguaEnglishTaggerParagraph
