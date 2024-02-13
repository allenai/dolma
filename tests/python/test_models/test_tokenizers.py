from unittest import TestCase

from dolma.models.word_tokenizers import (
    FastTextPunctLowercaseTokenizer,
    FastTextPunctTokenizer,
    FastTextRemovePunctLowercaseTokenizer,
    FastTextRemovePunctTokenizer,
    FastTextWhitespaceLowercaseTokenizer,
    FastTextWhitespaceTokenizer,
    NoOpTokenizer,
)


class TestTokenizers(TestCase):
    def test_noop_tokenizer(self):
        tokenizer = NoOpTokenizer()

        self.assertEqual(tokenizer("Hello, World"), "Hello, World")
        self.assertEqual(tokenizer("Hello\nWorld!"), "Hello\nWorld!")
        self.assertEqual(tokenizer("Hello\n\n\tWorld!"), "Hello\n\n\tWorld!")

    def test_fasttext_whitespace_tokenizer(self):
        tokenizer = FastTextWhitespaceTokenizer()

        self.assertEqual(tokenizer("Hello, World"), "Hello, World")
        self.assertEqual(tokenizer("Hello\nWorld!"), "Hello World!")
        self.assertEqual(tokenizer("Hello\n\n\tWorld!"), "Hello World!")

    def test_fasttext_whitespace_lowercase_tokenizer(self):
        tokenizer = FastTextWhitespaceLowercaseTokenizer()

        self.assertEqual(tokenizer("Hello, World"), "hello, world")
        self.assertEqual(tokenizer("Hello\nWorld!"), "hello world!")
        self.assertEqual(tokenizer("Hello\n\n\tWorld!"), "hello world!")

    def test_fasttext_punct_tokenizer(self):
        tokenizer = FastTextPunctTokenizer()

        self.assertEqual(tokenizer("Hello, World"), "Hello , World")
        self.assertEqual(tokenizer("Hello\nWorld!!"), "Hello World ! !")
        self.assertEqual(tokenizer("Hello\n\n\tWorld!"), "Hello World !")

    def test_fasttext_punct_lowercase_tokenizer(self):
        tokenizer = FastTextPunctLowercaseTokenizer()

        self.assertEqual(tokenizer("Hello, World"), "hello , world")
        self.assertEqual(tokenizer("Hello\nWorld!!"), "hello world ! !")
        self.assertEqual(tokenizer("Hello\n\n\tWorld!"), "hello world !")

    def test_fasttext_remove_punct_tokenizer(self):
        tokenizer = FastTextRemovePunctTokenizer()

        self.assertEqual(tokenizer("hello, world"), "hello world")
        self.assertEqual(tokenizer("Hello\nWorld!"), "Hello World")
        self.assertEqual(tokenizer("Hello\n\n\tWorld!"), "Hello World")

    def test_fasttext_remove_punct_lowercase_tokenizer(self):
        tokenizer = FastTextRemovePunctLowercaseTokenizer()

        self.assertEqual(tokenizer("Hello, World"), "hello world")
        self.assertEqual(tokenizer("Hello\nWorld!"), "hello world")
        self.assertEqual(tokenizer("Hello\n\n\tWorld!"), "hello world")
