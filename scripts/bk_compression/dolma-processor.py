#!/usr/bin/env python3
"""
Token Statistics Processor for Dolma
This processor calculates text length, token counts, and compression ratio for documents.
"""

import json
import gzip
from io import BytesIO
from typing import Dict, Any, List, Iterator, Optional, Union, Set

from dolma.core.parallel import BaseProcessor, Item, register_processor


@register_processor("token_stats")
class TokenStatsProcessor(BaseProcessor):
    """
    A processor that calculates token statistics for documents.
    Computes:
    - Token counts using specified tokenizer
    - Text length in characters
    - Characters per token ratio
    - Compression ratio of the text using gzip
    """
    
    def __init__(
        self,
        tokenizer_name: str = "EleutherAI/pythia-410m",
        compresslevel: int = 6,
        **kwargs
    ):
        """
        Initialize the TokenStatsProcessor.
        
        Args:
            tokenizer_name: Name or path of the HuggingFace tokenizer to use
            compresslevel: Compression level for gzip (1-9, where 9 is highest compression)
            **kwargs: Additional arguments passed to BaseProcessor
        """
        super().__init__(**kwargs)
        self.tokenizer_name = tokenizer_name
        self.compresslevel = compresslevel
        self._tokenizer = None
        
    @property
    def tokenizer(self):
        """Lazy-load the tokenizer to avoid loading it unless needed."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        return self._tokenizer
        
    def process_items(self, items: List[Item]) -> Iterator[Dict[str, Any]]:
        """
        Process a batch of documents to extract token statistics.
        
        Args:
            items: List of document items to process
            
        Yields:
            Dictionary with computed statistics for each document
        """
        for item in items:
            # Extract text from the document
            text = item.doc.get("text", "")
            if not text:
                continue
                
            # Calculate text length
            text_length = len(text)
            
            # Tokenize the text
            token_ids = self.tokenizer.encode(text)
            num_tokens = len(token_ids)
            
            # Calculate characters per token
            chars_per_token = text_length / num_tokens if num_tokens > 0 else 0
            
            # Calculate compression ratio of just the text field using gzip
            text_bytes = text.encode('utf-8')
            uncompressed_size = len(text_bytes)
            
            # Use BytesIO to avoid writing to disk
            compressed_bytes = BytesIO()
            with gzip.GzipFile(fileobj=compressed_bytes, mode='wb', compresslevel=self.compresslevel) as f:
                f.write(text_bytes)
            
            compressed_size = len(compressed_bytes.getvalue())
            text_compression_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 0
            
            # Return results with document ID and metadata
            yield {
                "doc_id": item.doc_id,
                "num_tokens": num_tokens,
                "text_length": text_length,
                "chars_per_token": chars_per_token,
                "text_compression_ratio": text_compression_ratio,
                # Include original document source info if available
                "source": item.doc.get("source", None),
                "url": item.doc.get("metadata", {}).get("url", None)
            }
    
    @classmethod
    def get_description(cls) -> str:
        """Return a description of this processor."""
        return "Calculates token statistics, text length, and compression ratio for documents."


if __name__ == "__main__":
    # This allows the processor to be run as a script with the Dolma CLI
    import sys
    from dolma.cli import main
    sys.exit(main())
