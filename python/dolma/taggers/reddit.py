import re
from typing import List
from warnings import warn

import regex
import uniseg.wordbreak
from tokenizers import Regex, Tokenizer, pre_tokenizers

from dolma.core.data_types import DocResult, Document, Span, TextSlice
from dolma.core.registry import TaggerRegistry
from dolma.core.taggers import BaseTagger,BaseTaggerWithMetadata
from dolma.core.utils import split_paragraphs

"""
CANDIDATE TAGGERS

METADATA (hopefully)
num_comment
Score
Edited/deleted/hidden/pinned (submissions)
NSFW
Contains media
Contains URL
Edited, stickied, collapsed, self-comment, removed (comments)
over_18

EXISTING TAGGERS
Length in chars

NEW TAGGERS
Amount of length that is not URL
Length in BPE tokens
Len longest string without spaces
Whether starts with non-ascii char

NEW TAGGERS NEEDING LISTS
Known bot
Known non-English subreddit
Member of set of subreddits

NOT SURE YET
Depth in thread


"""

@TaggerRegistry.add("contains_url")
class ContainsURL(BaseTagger):

    def predict(self, doc: Document) -> DocResult:
        url_regex = re.compile(".*https?://[^\s]+",re.DOTALL)
        score = 1 if re.match(url_regex,doc.text) else 0
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="doc", score=score)])

@TaggerRegistry.add("len_wo_url")
class LenWithoutURL(BaseTagger):

    def predict(self, doc: Document) -> DocResult:
        url_regex = re.compile("https?://[^\s]+")
        # url_regex = re.compile("https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)")
        score = len(re.sub(url_regex,"",doc.text))
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])

@TaggerRegistry.add("bpe_tokenizer")
class BPETokenizer(BaseTagger):
    TOKENIZER_NAME_OR_PATH = "allenai/dolma2-tokenizer"

    def __init__(self) -> None:
        self.tokenizer = Tokenizer.from_pretrained(self.TOKENIZER_NAME_OR_PATH)
        super().__init__()

    def predict(self, doc: Document) -> DocResult:
        score = len(self.tokenizer.encode(text)) if (text := doc.text.strip()) else 0
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])

@TaggerRegistry.add("longest_string")
class LongestString(BaseTagger):
    # WHITESPACE_REGEX = regex.compile(r"\w+|[^\w\s]+")

    def predict(self, doc: Document) -> DocResult:
        score = max([len(x) for x in doc.text.split()])
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])

@TaggerRegistry.add("starts_nonascii")
class NonASCII(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        score = int(not doc.text[0].isascii())
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="doc", score=score)])

@TaggerRegistry.add("list_membership")
class ListMembership(BaseTaggerWithMetadata):
    
    def __init__(self) -> None:
        self.blocklist = set()
        with open(self.LOOKUP_LIST) as f:
            for line in f:
                self.blocklist.add(line.strip().lower())

@TaggerRegistry.add("banned_subreddit_membership")
class BannedSubs(ListMembership):
    LOOKUP_LIST="dolma/sources/reddit/reddit_blocklists/banned_subreddits.txt"

    def predict(self, doc: Document) -> DocResult:
        score = doc.metadata["subreddit"].lower() in self.blocklist
        if score:
            print(doc)
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="doc", score=score)])

@TaggerRegistry.add("non_english_subreddit")
class NonEnglishSubs(ListMembership):
    LOOKUP_LIST="dolma/sources/reddit/reddit_blocklists/non-english_subreddits.txt"

    def predict(self, doc: Document) -> DocResult:
        score = doc.metadata["subreddit"].lower() in self.blocklist
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="doc", score=score)])

@TaggerRegistry.add("bot_author")
class BotAuthor(ListMembership):
    LOOKUP_LIST="dolma/sources/reddit/reddit_blocklists/thresholded_botlist.txt"

    def predict(self, doc: Document) -> DocResult:
        score = doc.metadata["author"] in self.blocklist
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="doc", score=score)])
