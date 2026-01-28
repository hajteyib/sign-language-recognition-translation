import pickle
import os
from collections import Counter

class Vocabulary:
    """
    Vocabulary for text sequences with special tokens.
    """
    
    # Special tokens
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'
    
    def __init__(self, min_freq=1):
        """
        Args:
            min_freq: Minimum frequency for a word to be included in vocabulary
        """
        self.min_freq = min_freq
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.SOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.UNK_TOKEN: 3
        }
        self.idx2word = {
            0: self.PAD_TOKEN,
            1: self.SOS_TOKEN,
            2: self.EOS_TOKEN,
            3: self.UNK_TOKEN
        }
        self.word_freq = Counter()
    
    def build_from_texts(self, texts):
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of strings (sentences)
        """
        # Count word frequencies
        for text in texts:
            words = text.split()
            self.word_freq.update(words)
        
        # Add words above min_freq to vocabulary
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"Built vocabulary with {len(self.word2idx)} words (min_freq={self.min_freq})")
        print(f"  Special tokens: {len([self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN])}")
        print(f"  Regular words: {len(self.word2idx) - 4}")
    
    def encode(self, text, add_special_tokens=True):
        """
        Convert text to sequence of indices.
        
        Args:
            text: String to encode
            add_special_tokens: Whether to add SOS/EOS tokens
        
        Returns:
            indices: List of integer indices
        """
        words = text.split()
        indices = []
        
        if add_special_tokens:
            indices.append(self.word2idx[self.SOS_TOKEN])
        
        for word in words:
            indices.append(self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]))
        
        if add_special_tokens:
            indices.append(self.word2idx[self.EOS_TOKEN])
        
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        """
        Convert sequence of indices to text.
        
        Args:
            indices: List of integer indices or tensor
            skip_special_tokens: Whether to skip special tokens in output
        
        Returns:
            text: Decoded string
        """
        # Convert tensor to list if needed
        if hasattr(indices, 'tolist'):
            indices = indices.tolist()
        
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            
            if skip_special_tokens and word in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
                continue
            
            words.append(word)
        
        return ' '.join(words)
    
    def __len__(self):
        """Return vocabulary size."""
        return len(self.word2idx)
    
    @property
    def pad_idx(self):
        return self.word2idx[self.PAD_TOKEN]
    
    @property
    def sos_idx(self):
        return self.word2idx[self.SOS_TOKEN]
    
    @property
    def eos_idx(self):
        return self.word2idx[self.EOS_TOKEN]
    
    @property
    def unk_idx(self):
        return self.word2idx[self.UNK_TOKEN]
    
    def save(self, path):
        """Save vocabulary to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': self.word_freq,
            'min_freq': self.min_freq
        }
        with open(path, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {path}")
    
    @staticmethod
    def load(path):
        """
        Load vocabulary from file.
        
        Args:
            path: Path to vocabulary file
        
        Returns:
            vocab: Vocabulary instance
        """
        with open(path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        vocab = Vocabulary(min_freq=vocab_data.get('min_freq', 1))
        vocab.word2idx = vocab_data['word2idx']
        vocab.idx2word = vocab_data['idx2word']
        vocab.word_freq = vocab_data.get('word_freq', Counter())
        
        print(f"Vocabulary loaded from {path} ({len(vocab)} words)")
        return vocab
    
    def get_most_common(self, n=10):
        """Get n most common words."""
        return self.word_freq.most_common(n)
    
    def stats(self):
        """Print vocabulary statistics."""
        print(f"\nVocabulary Statistics:")
        print(f"  Total size: {len(self)}")
        print(f"  Special tokens: 4 ({self.PAD_TOKEN}, {self.SOS_TOKEN}, {self.EOS_TOKEN}, {self.UNK_TOKEN})")
        print(f"  Regular words: {len(self) - 4}")
        print(f"  Min frequency: {self.min_freq}")
        print(f"\n  Most common words:")
        for word, freq in self.get_most_common(10):
            print(f"    '{word}': {freq}")
