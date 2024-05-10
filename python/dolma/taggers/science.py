import re
from typing import Optional

from ..core.data_types import DocResult, DocumentWithMetadata, Span
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTagger

MATH_KEYWORDS = [
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

LATEX_MATH_COMMANDS = [
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


SCIENCE_KEYWORDS = [
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
    "MathJax",
    "mathjax",
    "<math",
    "math-container",
    "katex.min.css",
    "latex.php",
    "codecogs",
    "tex.cgi",
    'class="tex"',
    'class="tex"',
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


@TaggerRegistry.add("owm_math_v1")
class OpenWebMathContainsMathTagger(BaseTagger):
    def __init__(self):
        self.expr = re.compile("|".join(MATH_KEYWORDS))

    def predict(self, doc: DocumentWithMetadata) -> DocResult:  # type: ignore
        html: Optional[str] = doc.metadata.get("html", None)
        if html is None:
            raise ValueError("Cannot find `html` key in metadata.")

        if match := self.expr.search(html):
            start, end = match.span()
            spans = [Span(start=start, end=end, type="math", score=end - start)]
        else:
            spans = []

        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("owm_latex_v1")
class OpenWebMathContainsLatexTagger(BaseTagger):
    def __init__(self):
        self.expr = re.compile("|".join(LATEX_MATH_COMMANDS))

    def predict(self, doc: DocumentWithMetadata) -> DocResult:  # type: ignore
        html: Optional[str] = doc.metadata.get("html", None)
        if html is None:
            raise ValueError("Cannot find `html` key in metadata.")

        if ("\\\\" in html) and (match := self.expr.search(html)):
            start, end = match.span()
            spans = [Span(start=start, end=end, type="latex", score=end - start)]
        else:
            spans = []

        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("science_kw_v1")
class ScienceKeywordsTagger(BaseTagger):
    def __init__(self):
        self.expr = re.compile("|".join(SCIENCE_KEYWORDS))

    def predict(self, doc: DocumentWithMetadata) -> DocResult:  # type: ignore
        text: Optional[str] = doc.metadata.get("html", None)
        if text is None:
            raise ValueError("Cannot find `html` key in metadata.")

        if match := self.expr.search(text):
            start, end = match.span()
            spans = [Span(start=start, end=end, type="science", score=end - start)]
        else:
            spans = []

        return DocResult(doc=doc, spans=spans)
