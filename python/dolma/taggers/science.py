from typing import TYPE_CHECKING, Any, Iterable, List, Optional
from dolma.core.ft_tagger import Prediction

from necessary import necessary

with necessary("acora", soft=True) as ACORA_AVAILABLE:
    if TYPE_CHECKING or ACORA_AVAILABLE:
        from acora import AcoraBuilder  # pyright: ignore


with necessary("hyperscan", soft=True) as HYPERSCAN_AVAILABLE:
    if TYPE_CHECKING or HYPERSCAN_AVAILABLE:
        from hyperscan import Database

from ..core.data_types import DocResult, DocumentWithMetadata, Span, TextSlice
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTaggerWithMetadata
from ..core.ft_tagger import BaseFastTextTagger


class BaseHTMLKeywordLookupTagger(BaseTaggerWithMetadata):
    KEYWORDS: List[bytes]
    TYPE: str

    def __init__(self):
        assert ACORA_AVAILABLE, "Acora is not available; please install with `pip install acora`."

        builder = AcoraBuilder()
        builder.update(self.KEYWORDS)
        self.acora = builder.build()

    def _get_content(self, doc: DocumentWithMetadata) -> bytes:
        html: Optional[bytes] = doc.metadata.get("html", None)
        if html is None:
            raise ValueError("Cannot find `html` key in metadata.")
        return html

    def predict(self, doc: DocumentWithMetadata) -> DocResult:  # type: ignore
        content = self._get_content(doc)

        # check if there's a match; if yes, return immediately
        for kw, pos in self.acora.finditer(content):
            return DocResult(
                doc=doc,
                spans=[Span(start=pos, end=pos + len(kw), type=self.TYPE, score=1, location="metadata.html")],
            )

        # if no match, return empty spans
        return DocResult(doc=doc, spans=[])


@TaggerRegistry.add("owm_math_v1")
class OpenWebMathContainsMathTagger(BaseHTMLKeywordLookupTagger):
    TYPE = "math"
    KEYWORDS = [
        b"MathJax",
        b"mathjax",
        b"<math",
        b"math-container",
        b"katex.min.css",
        b"latex.php",
        b"codecogs",
        b"tex.cgi",
        b'class="tex"',
        b"class='tex'",
    ]


@TaggerRegistry.add("owm_latex_v1")
class OpenWebMathContainsLatexTagger(BaseHTMLKeywordLookupTagger):
    TYPE = "latex"
    KEYWORDS = [
        b"\\alpha",
        b"\\bar",
        b"\\be",
        b"\\begin",
        b"\\beta",
        b"\\bf",
        b"\\big",
        b"\\cal",
        b"\\cdot",
        b"\\chi",
        b"\\cos",
        b"\\dagger",
        b"\\def",
        b"\\delta",
        b"\\dot",
        b"\\ee",
        b"\\ell",
        b"\\em",
        b"\\emph",
        b"\\end",
        b"\\epsilon",
        b"\\eqref",
        b"\\equiv",
        b"\\eta",
        b"\\frac",
        b"\\gamma",
        b"\\hat",
        b"\\hbox",
        b"\\hline",
        b"\\hspace",
        b"\\in",
        b"\\infty",
        b"\\int",
        b"\\it",
        b"\\item",
        b"\\kappa",
        b"\\label",
        b"\\lambda",
        b"\\langle",
        b"\\left",
        b"\\leq",
        b"\\ln",
        b"\\mathbf",
        b"\\mathcal",
        b"\\mathrm",
        b"\\mu",
        b"\\noindent",
        b"\\nonumber",
        b"\\nu",
        b"\\omega",
        b"\\otimes",
        b"\\over",
        b"\\overline",
        b"\\partial",
        b"\\phi",
        b"\\pi",
        b"\\pm",
        b"\\prime",
        b"\\psi",
        b"\\qquad",
        b"\\quad",
        b"\\rangle",
        b"\\ref",
        b"\\rho",
        b"\\right",
        b"\\rightarrow",
        b"\\rm",
        b"\\sigma",
        b"\\sin",
        b"\\sqrt",
        b"\\sum",
        b"\\tau",
        b"\\text",
        b"\\textit",
        b"\\theta",
        b"\\tilde",
        b"\\times",
        b"\\to",
        b"\\varepsilon",
        b"\\varphi",
        b"\\vec",
        b"\\vspace",
        b"\\xi",
        b"\\zeta",
    ]


@TaggerRegistry.add("science_kw_v1")
class ScienceKeywordsTagger(BaseHTMLKeywordLookupTagger):
    TYPE = "science"
    KEYWORDS = [
        b"acceleration",
        b"acid",
        b"acids",
        b"angular",
        b"appendix",
        b"atmosphere",
        b"atoms",
        b"blood",
        b"bmatrix",
        b"bond",
        b"bonds",
        b"carbon",
        b"chemical",
        b"circuit",
        b"climate",
        b"compound",
        b"compounds",
        b"computer",
        b"concentration",
        b"definition",
        b"determine",
        b"deviation",
        b"disease",
        b"distance",
        b"earth",
        b"earthquake",
        b"electron",
        b"electrons",
        b"equation",
        b"equations",
        b"equilibrium",
        b"exercise",
        b"exercises",
        b"financial",
        b"forces",
        b"frequency",
        b"hydrogen",
        b"hypothesis",
        b"income",
        b"learning",
        b"levels",
        b"located",
        b"magnetic",
        b"minerals",
        b"molecular",
        b"molecule",
        b"molecules",
        b"momentum",
        b"muscle",
        b"orbitals",
        b"oxygen",
        b"particle",
        b"patient",
        b"physical",
        b"population",
        b"pressure",
        b"probability",
        b"proportion",
        b"protein",
        b"reaction",
        b"reactions",
        b"rocks",
        b"sediment",
        b"simplify",
        b"soil",
        b"study",
        b"surface",
        b"temperature",
        b"theorem",
        b"theory",
        b"thesis",
        b"tissue",
        b"variables",
        b"vector",
        b"velocity",
        b"voltage",
        b"wave",
        b"waves",
    ]


class HyperscanHTMLKeywordLookupTagger(BaseTaggerWithMetadata):
    KEYWORDS: List[bytes]
    TYPE: str

    def __init__(self):
        assert HYPERSCAN_AVAILABLE, "Hyperscan is not available; please install with `pip install hyperscan`."

        self.db = Database()
        self.db.compile(
            expressions=self.KEYWORDS,
            ids=list(range(len(self.KEYWORDS))),
            elements=len(self.KEYWORDS),
            flags=[0 for _ in self.KEYWORDS],
        )

    def _get_content(self, doc: DocumentWithMetadata) -> bytes:
        html: Optional[bytes] = doc.metadata.get("html", None)
        if html is None:
            raise ValueError("Cannot find `html` key in metadata.")
        return html

    @staticmethod
    def _on_match(id_: int, from_: int, to: int, flags: int, context: Optional[Any] = None) -> None:
        if context is not None:
            context.append((id_, from_, to, flags))

    def predict(self, doc: DocumentWithMetadata) -> DocResult:  # type: ignore
        content = self._get_content(doc)

        context: List[tuple] = []
        self.db.scan(content, match_event_handler=self._on_match, context=context)
        if context:
            start, end = context[0][1], context[0][2]
            return DocResult(
                doc=doc, spans=[Span(start=start, end=end, type=self.TYPE, score=1, location="metadata.html")]
            )

        # if no match, return empty spans
        return DocResult(doc=doc, spans=[])


@TaggerRegistry.add("owm_math_v2")
class HyperscanOpenWebMathContainsMathTagger(HyperscanHTMLKeywordLookupTagger):
    TYPE = "math"
    KEYWORDS = [
        rb"MathJax",
        rb"mathjax",
        rb"<math",
        rb"math-container",
        rb"katex.min.css",
        rb"latex.php",
        rb"codecogs",
        rb"tex.cgi",
        rb'class="tex"',
        rb"class='tex'",
    ]


@TaggerRegistry.add("owm_latex_v2")
class HyperscanOpenWebMathContainsLatexTagger(HyperscanHTMLKeywordLookupTagger):
    TYPE = "latex"
    KEYWORDS = [
        rb"\\alpha",
        rb"\\bar",
        rb"\\be",
        rb"\\begin",
        rb"\\beta",
        rb"\\bf",
        rb"\\big",
        rb"\\cal",
        rb"\\cdot",
        rb"\\chi",
        rb"\\cos",
        rb"\\dagger",
        rb"\\def",
        rb"\\delta",
        rb"\\dot",
        rb"\\ee",
        rb"\\ell",
        rb"\\em",
        rb"\\emph",
        rb"\\end",
        rb"\\epsilon",
        rb"\\eqref",
        rb"\\equiv",
        rb"\\eta",
        rb"\\frac",
        rb"\\gamma",
        rb"\\hat",
        rb"\\hbox",
        rb"\\hline",
        rb"\\hspace",
        rb"\\in",
        rb"\\infty",
        rb"\\int",
        rb"\\it",
        rb"\\item",
        rb"\\kappa",
        rb"\\label",
        rb"\\lambda",
        rb"\\langle",
        rb"\\left",
        rb"\\leq",
        rb"\\ln",
        rb"\\mathbf",
        rb"\\mathcal",
        rb"\\mathrm",
        rb"\\mu",
        rb"\\noindent",
        rb"\\nonumber",
        rb"\\nu",
        rb"\\omega",
        rb"\\otimes",
        rb"\\over",
        rb"\\overline",
        rb"\\partial",
        rb"\\phi",
        rb"\\pi",
        rb"\\pm",
        rb"\\prime",
        rb"\\psi",
        rb"\\qquad",
        rb"\\quad",
        rb"\\rangle",
        rb"\\ref",
        rb"\\rho",
        rb"\\right",
        rb"\\rightarrow",
        rb"\\rm",
        rb"\\sigma",
        rb"\\sin",
        rb"\\sqrt",
        rb"\\sum",
        rb"\\tau",
        rb"\\text",
        rb"\\textit",
        rb"\\theta",
        rb"\\tilde",
        rb"\\times",
        rb"\\to",
        rb"\\varepsilon",
        rb"\\varphi",
        rb"\\vec",
        rb"\\vspace",
        rb"\\xi",
        rb"\\zeta",
    ]


@TaggerRegistry.add("science_kw_v2")
class HyperscanScienceKeywordsTagger(HyperscanHTMLKeywordLookupTagger):
    TYPE = "science"
    KEYWORDS = [
        rb"acceleration",
        rb"acid",
        rb"acids",
        rb"angular",
        rb"appendix",
        rb"atmosphere",
        rb"atoms",
        rb"blood",
        rb"bmatrix",
        rb"bond",
        rb"bonds",
        rb"carbon",
        rb"chemical",
        rb"circuit",
        rb"climate",
        rb"compound",
        rb"compounds",
        rb"computer",
        rb"concentration",
        rb"definition",
        rb"determine",
        rb"deviation",
        rb"disease",
        rb"distance",
        rb"earth",
        rb"earthquake",
        rb"electron",
        rb"electrons",
        rb"equation",
        rb"equations",
        rb"equilibrium",
        rb"exercise",
        rb"exercises",
        rb"financial",
        rb"forces",
        rb"frequency",
        rb"hydrogen",
        rb"hypothesis",
        rb"income",
        rb"learning",
        rb"levels",
        rb"located",
        rb"magnetic",
        rb"minerals",
        rb"molecular",
        rb"molecule",
        rb"molecules",
        rb"momentum",
        rb"muscle",
        rb"orbitals",
        rb"oxygen",
        rb"particle",
        rb"patient",
        rb"physical",
        rb"population",
        rb"pressure",
        rb"probability",
        rb"proportion",
        rb"protein",
        rb"reaction",
        rb"reactions",
        rb"rocks",
        rb"sediment",
        rb"simplify",
        rb"soil",
        rb"study",
        rb"surface",
        rb"temperature",
        rb"theorem",
        rb"theory",
        rb"thesis",
        rb"tissue",
        rb"variables",
        rb"vector",
        rb"velocity",
        rb"voltage",
        rb"wave",
        rb"waves",
    ]


@TaggerRegistry.add("ft_science_v1")
class FastTextScienceTagger(BaseFastTextTagger):
    MODEL_PATH = "https://dolma-artifacts.org/fasttext_models/scipile/model_exp_20000_0.3_owm_10000_syn_5000_wiki_5000_pretrained.bin"  # noqa: E501

    def __init__(self):
        BaseFastTextTagger.__init__(self, model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        preds = {
            label: float(score) for label, score in
            zip(*self.classifier.predict(text_slice.text.replace("\n", " ").strip(), k=-1))
        }
        return [Prediction(label="science", score=preds["__label__"])]


@TaggerRegistry.add("owm_math_latex_ft-science_combined")
class OwmMathLatexFtScienceCombined(HyperscanHTMLKeywordLookupTagger, BaseFastTextTagger):
    TYPE = "owm_math_latex"
    MODEL_PATH = FastTextScienceTagger.MODEL_PATH   # pyright: ignore
    KEYWORDS = (
        HyperscanOpenWebMathContainsMathTagger.KEYWORDS     # pyright: ignore
        + HyperscanOpenWebMathContainsLatexTagger.KEYWORDS  # pyright: ignore
    )

    def __init__(self):
        HyperscanHTMLKeywordLookupTagger.__init__(self)
        BaseFastTextTagger.__init__(self, model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def predict(self, doc: DocumentWithMetadata) -> DocResult:      # type: ignore
        keyword_result = HyperscanHTMLKeywordLookupTagger.predict(self, doc)

        if keyword_result.spans:
            return keyword_result

        return BaseFastTextTagger.predict(self, doc)
