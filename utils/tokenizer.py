"""
Lightweight word-level tokenizer for Transformer training.

This module provides a simple whitespace-based tokenizer that splits text into words
and builds a vocabulary from the training corpus. It supports basic encoding and decoding
operations for converting between text and token indices.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WordTokenizer:
    """A simple word-level tokenizer with vocabulary management."""
    
    vocab: dict = field(default_factory=dict)  # word -> index mapping
    inv_vocab: dict = field(default_factory=dict)  # index -> word mapping
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    
    def __post_init__(self):
        if not self.vocab:
            # Initialize with special tokens
            self.vocab = {self.unk_token: 0, self.pad_token: 1}
            self.inv_vocab = {0: self.unk_token, 1: self.pad_token}
    
    def tokenize(self, text: str) -> list[str]:
        """Split text into tokens (words) using whitespace."""
        return text.split()
    
    def build_vocab(self, texts: list[str], max_size: Optional[int] = None):
        """Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings to build vocabulary from
            max_size: Maximum vocabulary size (excluding special tokens)
        """
        word_counts = {}
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
        
        # Sort by frequency (descending), then alphabetically for ties
        sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
        
        # Build vocabulary with frequency ordering
        idx = len(self.vocab)  # Start after special tokens
        for word, _ in sorted_words:
            if max_size and idx >= len(self.vocab) + max_size:
                break
            if word not in self.vocab:
                self.vocab[word] = idx
                self.inv_vocab[idx] = word
                idx += 1
    
    def encode(self, text: str) -> list[int]:
        """Convert text to token indices.
        
        Args:
            text: Input text string
            
        Returns:
            List of token indices
        """
        tokens = self.tokenize(text)
        unk_idx = self.vocab[self.unk_token]
        return [self.vocab.get(token, unk_idx) for token in tokens]
    
    def decode(self, indices: list[int]) -> str:
        """Convert token indices back to text.
        
        Args:
            indices: List of token indices
            
        Returns:
            Decoded text string
        """
        words = []
        for idx in indices:
            word = self.inv_vocab.get(int(idx), self.unk_token)
            if word not in [self.unk_token, self.pad_token]:
                words.append(word)
        return " ".join(words)
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def save(self, filepath: str):
        """Save tokenizer to file.
        
        Args:
            filepath: Path to save the tokenizer
        """
        import json
        data = {
            "vocab": self.vocab,
            "inv_vocab": {str(k): v for k, v in self.inv_vocab.items()},
            "unk_token": self.unk_token,
            "pad_token": self.pad_token
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'WordTokenizer':
        """Load tokenizer from file.
        
        Args:
            filepath: Path to load the tokenizer from
            
        Returns:
            Loaded WordTokenizer instance
        """
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(
            unk_token=data["unk_token"],
            pad_token=data["pad_token"]
        )
        tokenizer.vocab = data["vocab"]
        tokenizer.inv_vocab = {int(k): v for k, v in data["inv_vocab"].items()}
        return tokenizer
