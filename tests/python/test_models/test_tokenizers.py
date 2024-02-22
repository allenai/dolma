from unittest import TestCase

from dolma.models.word_tokenizers import (
    FastTextPunctLowercaseTokenizer,
    FastTextPunctTokenizer,
    FastTextRemovePunctLowercaseTokenizer,
    FastTextRemovePunctTokenizer,
    FastTextWhitespaceLowercaseTokenizer,
    FastTextWhitespaceTokenizer,
    NoOpTokenizer,
    ParagraphFastTextRemovePunctLowercaseTokenizer,
    ParagraphFastTextWhitespaceLowercaseTokenizer,
)


class TestTokenizers(TestCase):
    def test_noop_tokenizer(self):
        tokenizer = NoOpTokenizer()

        tokenized, *_ = tokenizer("Hello, World")
        self.assertEqual(tokenized, "Hello, World")
        tokenized, *_ = tokenizer("Hello\nWorld!")
        self.assertEqual(tokenized, "Hello\nWorld!")
        tokenized, *_ = tokenizer("Hello\n\n\tWorld!")
        self.assertEqual(tokenized, "Hello\n\n\tWorld!")

    def test_fasttext_whitespace_tokenizer(self):
        tokenizer = FastTextWhitespaceTokenizer()

        tokenized, *_ = tokenizer("Hello, World")
        self.assertEqual(tokenized, "Hello, World")
        tokenized, *_ = tokenizer("Hello\nWorld!")
        self.assertEqual(tokenized, "Hello World!")
        tokenized, *_ = tokenizer("Hello\n\n\tWorld!")
        self.assertEqual(tokenized, "Hello World!")

    def test_fasttext_whitespace_lowercase_tokenizer(self):
        tokenizer = FastTextWhitespaceLowercaseTokenizer()

        tokenized, *_ = tokenizer("Hello, World")
        self.assertEqual(tokenized, "hello, world")
        tokenized, *_ = tokenizer("Hello\nWorld!")
        self.assertEqual(tokenized, "hello world!")
        tokenized, *_ = tokenizer("Hello\n\n\tWorld!")
        self.assertEqual(tokenized, "hello world!")

    def test_fasttext_punct_tokenizer(self):
        tokenizer = FastTextPunctTokenizer()

        tokenized, *_ = tokenizer("Hello, World")
        self.assertEqual(tokenized, "Hello , World")
        tokenized, *_ = tokenizer("Hello\nWorld!!")
        self.assertEqual(tokenized, "Hello World ! !")
        tokenized, *_ = tokenizer("Hello\n\n\tWorld!")
        self.assertEqual(tokenized, "Hello World !")

    def test_fasttext_punct_lowercase_tokenizer(self):
        tokenizer = FastTextPunctLowercaseTokenizer()

        tokenized, *_ = tokenizer("Hello, World")
        self.assertEqual(tokenized, "hello , world")
        tokenized, *_ = tokenizer("Hello\nWorld!!")
        self.assertEqual(tokenized, "hello world ! !")
        tokenized, *_ = tokenizer("Hello\n\n\tWorld!")
        self.assertEqual(tokenized, "hello world !")

    def test_fasttext_remove_punct_tokenizer(self):
        tokenizer = FastTextRemovePunctTokenizer()

        tokenized, *_ = tokenizer("hello, world")
        self.assertEqual(tokenized, "hello world")
        tokenized, *_ = tokenizer("Hello\nWorld!")
        self.assertEqual(tokenized, "Hello World")
        tokenized, *_ = tokenizer("Hello\n\n\tWorld!")
        self.assertEqual(tokenized, "Hello World")

    def test_fasttext_remove_punct_lowercase_tokenizer(self):
        tokenizer = FastTextRemovePunctLowercaseTokenizer()

        tokenized, *_ = tokenizer("Hello, World")
        self.assertEqual(tokenized, "hello world")
        tokenized, *_ = tokenizer("Hello\nWorld!")
        self.assertEqual(tokenized, "hello world")
        tokenized, *_ = tokenizer("Hello\n\n\tWorld!")
        self.assertEqual(tokenized, "hello world")

    def test_paragraph_fasttext_whitespace_tokenizer(self):
        tokenizer = ParagraphFastTextWhitespaceLowercaseTokenizer(min_length=3)

        sentences = [
            "This is a test of a paragraph tokenizer.",
            "The tokenizer–should–split on whitespace.",
            "And lowercase the text.\nIt will do this by paragraph.\n\nshort!",
        ]
        all_tokenized = list(tokenizer("\n".join(sentences)))
        self.assertEqual(len(all_tokenized), 4)
        self.assertEqual(all_tokenized[0], sentences[0].lower())
        self.assertEqual(all_tokenized[1], sentences[1].lower())
        self.assertEqual(all_tokenized[2], sentences[2].lower().split("\n")[0])
        self.assertEqual(all_tokenized[3], sentences[2].lower().split("\n")[1])

    def test_paragraph_fasttext_remove_punct_lowercase_tokenizer(self):
        tokenizer = ParagraphFastTextRemovePunctLowercaseTokenizer(min_length=3)

        sentences = [
            "This is a test of a paragraph tokenizer.",
            "The tokenizer–should–split on whitespace.",
            "And lowercase the text.\nIt will do this by paragraph.\n\nshort!",
        ]
        all_tokenized = list(tokenizer("\n".join(sentences)))
        self.assertEqual(len(all_tokenized), 4)
        self.assertEqual(all_tokenized[0], "this is a test of a paragraph tokenizer")
        self.assertEqual(all_tokenized[1], "the tokenizer should split on whitespace")
        self.assertEqual(all_tokenized[2], "and lowercase the text")
        self.assertEqual(all_tokenized[3], "it will do this by paragraph")
