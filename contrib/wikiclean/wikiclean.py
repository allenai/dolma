from typing import NamedTuple
from dolma.core.data_types import DocResult, Document, Span
from dolma import add_tagger, BaseTagger
import spacy
import regex as re
import html


class Section(NamedTuple):
    type: str
    text: str


@add_tagger("wikiclean")
class WikicleanTagger(BaseTagger):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.nlp = spacy.load("en_core_web_sm",  disable=["parser", "ner", "lemmatizer"])

    def clean_text(self, text: str) -> str:
        # replace &amp; with &
        text = html.unescape(text)

        # replace \xa0 with space
        text = text.replace("\xa0", " ")

        # remove space/punctuation right after opening parentheses
        text = re.sub(r'\([\p{P}|\s]+([^\p{P}])', r'\(\1', text)

        # remove space/punctuation right before closing parentheses
        text = re.sub(r'([^\p{P}])[\p{P}|\s]+\)', r'\1\)', text)

        # remove empty parentheses
        text = re.sub(r"\(\s+\)", "", text)

        # remove empty quotes
        text = text.replace('""', "").replace("''", "")

        # remove space before punctuation
        text = re.sub(r"(\w)\s([.;!?,])", r"\1\2", text)

        # remove multiple spaces
        text = re.sub(r"( )+", " ", text)

        return text.strip()

    def is_valid_page(self, title: str, text: str) -> bool:
        if not (text.endswith(".") or text.endswith("?") or text.endswith("!")):
            return False

        if text.count(" ") < 20:
            return False

        if len(text) < 500:
            return False

        if "List of" in title:
            return False

        if "(disambiguation)" in title:
            return False
        return True

    def predict(self, doc: Document) -> DocResult:
        title, text = doc.text.split("\n\n", 1)
        lines = text.split("\n")


        processed_lines: list[Section] = []

        if self.is_valid_page(title, text):
            processed_lines.append(Section(type="title", text=title))

            # split into sections
            for line in lines:
                line = self.clean_text(line)
                if line.count(" ") < 10 and line.endswith("."):
                    # not a possible section
                    # Use spacy to check if line contains verbs
                    doc_line = self.nlp(line)
                    has_verb = any(token.pos_ == "VERB" for token in doc_line)

                    if not has_verb:
                        # No verbs found, this is likely a title
                        processed_lines.append(Section(type="header", text=line.strip('.').strip()))
                        continue

                processed_lines.append(Section(type="text", text=line))

        # Remove adjacent headers, keeping only the last one
        filtered_lines: list[Section] = []
        i = 0
        while i < len(processed_lines):
            if processed_lines[i].type == "header":
                # Look ahead to see if there are more headers
                j = i
                while j < len(processed_lines) and processed_lines[j].type == "header":
                    j += 1
                # Add only the last header in the sequence
                filtered_lines.append(processed_lines[j - 1])
                i = j
            else:
                filtered_lines.append(processed_lines[i])
                i += 1

        processed_lines = filtered_lines



        print()
        for section in processed_lines:
            print(section)
        print()
        breakpoint()


        # we assign the random score to a span that
        # covers the entire document
        span = Span(
            start=0,
            end=len(doc.text),
            type="wikiclean",
            score="aaa"
        )

        # we return the span wrapped in a DocResult object
        return DocResult(doc=doc, spans=[span])
