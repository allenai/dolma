import re
from typing import List
from warnings import warn
from pathlib import Path

import regex
import uniseg.wordbreak
from tokenizers import Regex, Tokenizer, pre_tokenizers

from ..core.data_types import DocResult, Document, Span, TextSlice
from ..core.registry import TaggerRegistry
from ..core.taggers import BaseTagger,BaseTaggerWithMetadata
from ..core.utils import split_paragraphs


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
        score = len(re.sub(url_regex,"",doc.text))
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])

@TaggerRegistry.add("len_wo_url_v2")
class LenWithoutURL2(BaseTagger):

    def predict(self, doc: Document) -> DocResult:
        url_regex = re.compile("(\[[^\]]*])?\(?https?://[^\s]+\)?")
        score = len(re.sub(url_regex,"",doc.text))
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])

@TaggerRegistry.add("longest_tok_wo_url")
class LongestTokWithoutURL(BaseTagger):
    TOKENIZER_NAME_OR_PATH = "allenai/dolma2-tokenizer"

    def __init__(self) -> None:
        self.tokenizer = Tokenizer.from_pretrained(self.TOKENIZER_NAME_OR_PATH)
        super().__init__()

    def predict(self, doc: Document) -> DocResult:
        url_regex = re.compile("https?://[^\s]+\)?")
        toks = self.tokenizer.encode(text).tokens if (text := re.sub(url_regex,"",doc.text.strip())) else []
        score = max([len(x) for x in toks],default=0)
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])

@TaggerRegistry.add("longest_tok_wo_url_v2")
class LongestTokWithoutURL2(BaseTagger):
    TOKENIZER_NAME_OR_PATH = "allenai/dolma2-tokenizer"

    def __init__(self) -> None:
        self.tokenizer = Tokenizer.from_pretrained(self.TOKENIZER_NAME_OR_PATH)
        super().__init__()

    def predict(self, doc: Document) -> DocResult:
        url_regex = re.compile("(\[[^\]]*])?\(?https?://[^\s]+\)?")
        toks = self.tokenizer.encode(text).tokens if (text := re.sub(url_regex,"",doc.text.strip())) else []
        score = max([len(x) for x in toks],default=0)
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])

@TaggerRegistry.add("perc_len_wo_url")
class PercLenWithoutURL(BaseTagger):

    def predict(self, doc: Document) -> DocResult:
        url_regex = re.compile("https?://[^\s]+\)?")
        score = len(re.sub(url_regex,"",doc.text)) / len(doc.text) if doc.text else 0
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])

@TaggerRegistry.add("perc_len_wo_url_v2")
class PercLenWithoutURL2(BaseTagger):

    def predict(self, doc: Document) -> DocResult:
        url_regex = re.compile("(\[[^\]]*])?\(?https?://[^\s]+\)?")
        score = len(re.sub(url_regex,"",doc.text)) / len(doc.text) if doc.text else 0
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

    def predict(self, doc: Document) -> DocResult:
        score = max([len(x) for x in doc.text.split()],default=0)
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])

@TaggerRegistry.add("starts_nonascii")
class NonASCII(BaseTagger):
    def predict(self, doc: Document) -> DocResult:
        score = int(not doc.text[0].isascii()) if doc.text else 0
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="doc", score=score)])

@TaggerRegistry.add("removed_deleted")
class RemovedDeleted(BaseTaggerWithMetadata):
    LOOKUP_LIST = (Path(__file__).parent / "../data/reddit_blocklists/thresholded_botlist.txt")

    def predict(self, doc: Document) -> DocResult:
        # author = doc.metadata.get("author",None) in ("[deleted]","[removed]","[UNICODE ENCODE ERROR]")
        # body = doc.metadata.get("body",None) in ("[deleted]","[removed]","[UNICODE ENCODE ERROR]")
        present = ("[deleted]" in doc.text) or ("[removed]" in doc.text) or ("[UNICODE ENCODE ERROR]" in doc.text)
        # score = author or body or present
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="doc", score=present)])

@TaggerRegistry.add("list_membership")
class ListMembership(BaseTaggerWithMetadata):
    
    def __init__(self) -> None:
        self.blocklist = set()
        with open(self.LOOKUP_LIST) as f:
            for line in f:
                self.blocklist.add(line.strip().lower())

@TaggerRegistry.add("banned_subreddit_membership")
class BannedSubs(ListMembership):
    LOOKUP_LIST = (Path(__file__).parent / "../data/reddit_blocklists/banned_subreddits.txt")

    def predict(self, doc: Document) -> DocResult:
        score = doc.metadata["subreddit"].lower() in self.blocklist
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="doc", score=score)])

@TaggerRegistry.add("non_english_subreddit")
class NonEnglishSubs(ListMembership):
    LOOKUP_LIST = (Path(__file__).parent / "../data/reddit_blocklists/non-english_subreddits.txt")

    def predict(self, doc: Document) -> DocResult:
        score = doc.metadata["subreddit"].lower() in self.blocklist
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="doc", score=score)])
    
@TaggerRegistry.add("mmlu_subreddit")
class MMLUSubs(ListMembership):
    LOOKUP_LIST = (Path(__file__).parent / "../data/reddit_blocklists/mmlu_topic_subreddits.txt")

    def predict(self, doc: Document) -> DocResult:
        score = doc.metadata["subreddit"].lower() in self.blocklist
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="doc", score=score)])
    
@TaggerRegistry.add("science_subreddit")
class ScienceSubs(ListMembership):
    LOOKUP_LIST = (Path(__file__).parent / "../data/reddit_blocklists/sciencesubreddits.txt")

    def predict(self, doc: Document) -> DocResult:
        score = doc.metadata["subreddit"].lower() in self.blocklist
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="doc", score=score)])
    
@TaggerRegistry.add("history_subreddit")
class HistorySubs(ListMembership):
    LOOKUP_LIST = (Path(__file__).parent / "../data/reddit_blocklists/historysubreddits.txt")

    def predict(self, doc: Document) -> DocResult:
        score = doc.metadata["subreddit"].lower() in self.blocklist
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="doc", score=score)])
    
@TaggerRegistry.add("politics_subreddit")
class PoliticsSubs(ListMembership):
    LOOKUP_LIST = (Path(__file__).parent / "../data/reddit_blocklists/politicssubreddits.txt")

    def predict(self, doc: Document) -> DocResult:
        score = doc.metadata["subreddit"].lower() in self.blocklist
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="doc", score=score)])

@TaggerRegistry.add("bot_author")
class BotAuthor(ListMembership):
    LOOKUP_LIST = (Path(__file__).parent / "../data/reddit_blocklists/thresholded_botlist.txt")

    def predict(self, doc: Document) -> DocResult:
        score = doc.metadata["author"] in self.blocklist
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="doc", score=score)])
    
@TaggerRegistry.add("comment_bot_author")
class CommentBotAuthor(ListMembership):
    LOOKUP_LIST = (Path(__file__).parent / "../data/reddit_blocklists/thresholded_botlist.txt")

    def predict(self, doc: Document) -> DocResult:
        score = any(auth in self.blocklist for auth in doc.metadata["comment_authors"])
        return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="doc", score=score)])

# @TaggerRegistry.add("wildguard_classifier")
# class WildGuardClassifier(BaseTagger):

#     def __init__(self) -> None:
#         from wildguard import load_wildguard
#         self.wildguard = load_wildguard(use_vllm=False,ephemeral_model=False)
#         super().__init__()

#     def predict(self, doc: Document) -> DocResult:

#         results = self.wildguard.classify([{"prompt": doc.text.strip()}])
#         score = 1 if results[0]["prompt_harmfulness"] == "harmful" else 0
#         return DocResult(doc=doc, spans=[Span(start=0, end=len(doc.text), type="length", score=score)])