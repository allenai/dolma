from typing import List, Optional

from acora import AcoraBuilder

from ..core.data_types import DocResult, DocumentWithMetadata, Span
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTaggerWithMetadata


class BaseHTMLKeywordLookupTagger(BaseTaggerWithMetadata):
    KEYWORDS: List[str]
    TYPE: str

    def __init__(self):
        builder = AcoraBuilder()
        builder.update(self.KEYWORDS)
        self.acora = builder.build()

    def _get_content(self, doc: DocumentWithMetadata) -> str:
        html: Optional[str] = doc.metadata.get("html", None)
        if html is None:
            raise ValueError("Cannot find `html` key in metadata.")
        return html

    def predict(self, doc: DocumentWithMetadata) -> DocResult:  # type: ignore
        content = self._get_content(doc)
        spans = [
            Span(
                start=(start := match[1]),
                end=(end := match[1] + len(match[0])),
                type=self.TYPE,
                score=(end - start),
            )
            for match in self.acora.finditer(content)
        ]

        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("owm_math_v1")
class OpenWebMathContainsMathTagger(BaseHTMLKeywordLookupTagger):
    TYPE = "math"
    KEYWORDS = [
        "MathJax",
        "mathjax",
        "<math",
        "math-container",
        "katex.min.css",
        "latex.php",
        "codecogs",
        "tex.cgi",
        'class="tex"',
        "class='tex'",
    ]


@TaggerRegistry.add("owm_latex_v1")
class OpenWebMathContainsLatexTagger(BaseHTMLKeywordLookupTagger):
    TYPE = "latex"
    KEYWORDS = [
        "\\end",
        "\\begin",
        "\\ref",
        "\\frac",
        "\\label",
        "\\bf",
        "\\right",
        "\\left",
        "\\rm",
        "\\alpha",
        "\\mu",
        "\\def",
        "\\it",
        "\\pi",
        "\\sigma",
        "\\sum",
        "\\lambda",
        "\\beta",
        "\\nu",
        "\\partial",
        "\\int",
        "\\delta",
        "\\rho",
        "\\phi",
        "\\gamma",
        "\\omega",
        "\\over",
        "\\nonumber",
        "\\bar",
        "\\sqrt",
        "\\theta",
        "\\tau",
        "\\em",
        "\\rangle",
        "\\hat",
        "\\tilde",
        "\\cal",
        "\\hline",
        "\\item",
        "\\psi",
        "\\vec",
        "\\langle",
        "\\epsilon",
        "\\eta",
        "\\cdot",
        "\\in",
        "\\xi",
        "\\infty",
        "\\quad",
        "\\mathcal",
        "\\times",
        "\\emph",
        "\\mathbf",
        "\\prime",
        "\\be",
        "\\mathrm",
        "\\ee",
        "\\vspace",
        "\\pm",
        "\\chi",
        "\\ell",
        "\\text",
        "\\qquad",
        "\\noindent",
        "\\to",
        "\\varphi",
        "\\hspace",
        "\\leq",
        "\\cos",
        "\\eqref",
        "\\overline",
        "\\sin",
        "\\kappa",
        "\\hbox",
        "\\rightarrow",
        "\\varepsilon",
        "\\textit",
        "\\dagger",
        "\\big",
        "\\otimes",
        "\\equiv",
        "\\zeta",
        "\\dot",
        "\\ln",
    ]


@TaggerRegistry.add("science_kw_v1")
class ScienceKeywordsTagger(BaseHTMLKeywordLookupTagger):
    TYPE = "science"
    KEYWORDS = [
        "bmatrix",
        "theorem",
        "orbitals",
        "equations",
        "electrons",
        "equation",
        "hypothesis",
        "equilibrium",
        "probability",
        "deviation",
        "atoms",
        "molecules",
        "theory",
        "acceleration",
        "molecule",
        "hydrogen",
        "molecular",
        "thesis",
        "proportion",
        "simplify",
        "velocity",
        "momentum",
        "concentration",
        "compounds",
        "voltage",
        "magnetic",
        "definition",
        "compound",
        "particle",
        "vector",
        "population",
        "determine",
        "forces",
        "acids",
        "study",
        "exercises",
        "circuit",
        "bonds",
        "variables",
        "temperature",
        "oxygen",
        "exercise",
        "physical",
        "angular",
        "frequency",
        "chemical",
        "appendix",
        "pressure",
        "atmosphere",
        "reaction",
        "sediment",
        "distance",
        "waves",
        "surface",
        "reactions",
        "computer",
        "learning",
        "located",
        "electron",
        "levels",
        "wave",
        "carbon",
        "earthquake",
        "bond",
        "protein",
        "earth",
        "soil",
        "income",
        "disease",
        "tissue",
        "blood",
        "patient",
        "climate",
        "muscle",
        "financial",
        "acid",
        "minerals",
        "rocks",
    ]
