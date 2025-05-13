import regex

from ..core.data_types import DocResult, Document, Span
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTagger
from ..core.utils import split_paragraphs


@TaggerRegistry.add("not_alphanum_paragraph_v1")
class NotAlphanumParagraphV1(BaseTagger):
    def __init__(self) -> None:
        self.re_has_alphanum = regex.compile(r"[a-zA-Z0-9]", regex.UNICODE)
        self.re_all_punctuation = regex.compile(
            r"^("
            r"[[:punct:]]|"
            r"\s|"
            r"["
            "\U0001f300-\U0001f64f"
            "\U0001f680-\U0001f6ff"
            "\u2600-\u26ff\u2700-\u27bf"
            r"]+"
            r")+$",
            regex.UNICODE,
        )

    def predict(self, doc: Document) -> DocResult:
        spans = []

        for para in split_paragraphs(text=doc.text):
            if self.re_has_alphanum.search(para.text):
                continue

            if self.re_all_punctuation.search(para.text):
                spans.append(Span(start=para.start, end=para.end, type="all_punct", score=1))

        if not spans:
            spans.append(Span(start=0, end=len(doc.text), type="all_punct", score=0))

        return DocResult(doc=doc, spans=spans)


@TaggerRegistry.add("pipe_delimited_lines_v1")
class PipeDelimitedLinesTagger(BaseTagger):
    """
    Tags documents with a score indicating the percentage of lines that start and end with a '|' character.
    
    This tagger analyzes each line in a document and calculates what percentage of lines
    (after stripping whitespace) both start and end with the pipe symbol '|'.
    """
    
    def predict(self, doc: Document) -> DocResult:
        """
        Calculate the percentage of lines that start and end with '|'.
        
        Args:
            doc: The document to analyze.
            
        Returns:
            DocResult: Results containing a document-level score representing the
                     percentage of lines that start and end with '|'.
        """
        # Split the document into lines
        lines = doc.text.splitlines()
        
        # Count total non-empty lines
        non_empty_lines = [line for line in lines if line.strip()]
        total_lines = len(non_empty_lines)
        
        # Count lines that start and end with '|'
        pipe_delimited_count = 0
        for line in non_empty_lines:
            line = line.strip()
            if line and line.startswith('|') and line.endswith('|'):
                pipe_delimited_count += 1
        
        # Calculate percentage
        percentage = (pipe_delimited_count / total_lines) if total_lines > 0 else 0.0
        
        # Create a document-level span with the percentage score
        spans = [
            Span(
                start=0,
                end=len(doc.text),
                type="pipe_delimited_lines_ratio",
                score=percentage,
            )
        ]
        
        return DocResult(doc=doc, spans=spans)
    
@TaggerRegistry.add("avg_fraction_numbers_in_line_v1")
class AvgFractionNumbersInLineTagger(BaseTagger):
    """
    Tags a document based on the average fraction of characters in a line that are number, averaged across the whole doc

    """
    
    def predict(self, doc: Document) -> DocResult:
        """
        Calculate the percentage of lines that start and end with '|'.
        
        Args:
            doc: The document to analyze.
            
        Returns:
            DocResult: Results containing a document-level score representing the
                     percentage of lines that start and end with '|'.
        """
        # Split the document into lines
        lines = doc.text.splitlines()
        
        # Count total non-empty lines
        non_empty_lines = [line for line in lines if line.strip()]
        
        fraction_numbers = [len([c for c in line if c.isdigit()]) / len(line) for line in non_empty_lines]

        if len(fraction_numbers) > 0:
            avg_score = sum(fraction_numbers) / len(fraction_numbers)
        else:
            avg_score = 0.0
        
        # Create a document-level span with the percentage score
        spans = [
            Span(
                start=0,
                end=len(doc.text),
                type="avg_fraction_numbers_in_line_ratio",
                score=avg_score,
            )
        ]
        
        return DocResult(doc=doc, spans=spans)