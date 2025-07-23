from typing import NamedTuple
from dolma.core.data_types import DocResult, Document, Span
from dolma import add_tagger, BaseTagger
import spacy
import regex as re
import html
import openai


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
        text = re.sub(r'\([\p{Po}\p{Pc}\p{Pf}\p{Pd}\s]+([^\p{P}])', r'(\1', text)

        # remove space/punctuation right before closing parentheses
        text = re.sub(r'([^\p{P}])[\p{Po}\p{Pc}\p{Pf}\p{Pd}\s]+\)', r'\1)', text)

        # remove empty parentheses
        text = re.sub(r"\(\s*\)", "", text)

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

        total_page_length = sum(len(section.text) for section in processed_lines if section.type == "text")

        if total_page_length < 3000:
            span = Span(
                start=0,
                end=len(doc.text),
                type="wikiclean",
                score="\n".join(section.text if section.type == "text" else "" for section in processed_lines if section.type != "title").strip()
            )
            return DocResult(doc=doc, spans=[span])

        # count the length of the text sections until first header
        summary_text = ""
        for section in processed_lines:
            if section.type == "header":
                break
            if section.type == "text":
                summary_text += section.text + "\n"

        # if length of text is at least 500, we return it as is.
        if len(summary_text) >= 1000:
            span = Span(
                start=0,
                end=len(doc.text),
                type="wikiclean",
                score="\n".join(section.text for section in processed_lines if section.type == "text").strip()
            )
            return DocResult(doc=doc, spans=[span])

        # last case: if summary is too short, we use openai 4.1 nano api to expand the summary
        # and return the result

        prompt = f"""
        You are a helpful assistant that expands a summary of a Wikipedia page.

        The content of the page is:
        {doc.text}

        The existing summary is:
        {summary_text}...

        Continue the summary to be around 1000 characters. Keep the beginning of the summary as is.

        Return the expanded summary only, no other text.
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt.strip()}],
                max_tokens=200,
                temperature=0.8
            )
            content = response.choices[0].message.content
            assert content is not None and len(content) > 0
            span = Span(
                start=0,
                end=len(doc.text),
                type="wikiclean",
                score=content.strip()
            )
            # we return the span wrapped in a DocResult object
            return DocResult(doc=doc, spans=[span])

        except Exception as e:
            print(f"WARNING: Failed to expand summary: {e}")
            return DocResult(doc=doc, spans=[])
