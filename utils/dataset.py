import torch
from torch.utils.data import Dataset
import numpy as np
import os
import gzip
import pickle
import glob
from .vocabulary import Vocabulary
from .preprocessing import LandmarkNormalizer
from .augmentation import LandmarkAugmentation

class PhoenixDataset(Dataset):
    def __init__(self, root_dir, annotation_file, vocab_path, split='train', 
                 normalizer_path=None, augment=False, max_len=300):
        """
        root_dir: path to processed landmarks (e.g., data/processed/landmarks)
        annotation_file: path to gzip pickle annotation file
        vocab_path: path to vocabulary file
        split: 'train', 'dev', or 'test'
        normalizer_path: path to normalization statistics file
        augment: whether to apply data augmentation (train only)
        max_len: maximum sequence length
        """
        self.root_dir = root_dir
        self.split = split
        self.max_len = max_len
        self.augment = augment and (split == 'train')  # Only augment training data
        
        # Load vocabulary
        self.vocab = Vocabulary.load(vocab_path)
        print(f"Loaded vocabulary with {len(self.vocab)} words")
        
        # Load normalizer if provided
        self.normalizer = None
        if normalizer_path and os.path.exists(normalizer_path):
            self.normalizer = LandmarkNormalizer(normalizer_path)
            print(f"Loaded normalizer from {normalizer_path}")
        
        # Initialize augmentation
        self.augmentation = None
        if self.augment:
            self.augmentation = LandmarkAugmentation(
                rotation_deg=10,
                scale_range=(0.95, 1.05),
                translation_range=0.03,
                temporal_mask_prob=0.05,
                apply_prob=0.5
            )
            print("Data augmentation enabled")
        
        # Load annotations
        with gzip.open(annotation_file, 'rb') as f:
            self.annotations = pickle.load(f)
            
        # Filter annotations to only include those that have corresponding files
        self.data = []
        for item in self.annotations:
            # item['name'] looks like 'train/11August_2010_Wednesday_tagesschau-1'
            # Our file path: root_dir/train/11August_2010_Wednesday_tagesschau-1.npy
            
            rel_path = item['name'] + '.npy'
            full_path = os.path.join(self.root_dir, rel_path)
            
            if os.path.exists(full_path):
                self.data.append({
                    'path': full_path,
                    'text': item['text'],
                    'gloss': item['gloss']
                })
                
        print(f"Loaded {len(self.data)} samples for split {split}")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load landmarks (dict of numpy arrays)
        landmarks = np.load(item['path'], allow_pickle=True).item()
        
        # Apply normalization
        if self.normalizer is not None:
            landmarks = self.normalizer.normalize(landmarks)
        
        # Apply augmentation (train only)
        if self.augmentation is not None:
            landmarks = self.augmentation(landmarks)
        
        # Convert to tensors
        sample = {
            'pose': torch.tensor(landmarks['pose'], dtype=torch.float32),
            'left_hand': torch.tensor(landmarks['left_hand'], dtype=torch.float32),
            'right_hand': torch.tensor(landmarks['right_hand'], dtype=torch.float32),
            'face': torch.tensor(landmarks['face'], dtype=torch.float32),
            'text': torch.tensor(self.vocab.encode(item['text']), dtype=torch.long),
            'text_raw': item['text']
        }
        
        return sample

def collate_fn(batch):
    # Pad sequences
    # We need to pad each feature separately
    
    keys = ['pose', 'left_hand', 'right_hand', 'face']
    batch_dict = {k: [] for k in keys}
    texts = []
    src_lengths = []  # Track original sequence lengths
    
    max_len = 0
    for item in batch:
        seq_len = item['pose'].shape[0]
        max_len = max(max_len, seq_len)
        src_lengths.append(seq_len)
        texts.append(item['text'])
        
    # Pad features
    for k in keys:
        for item in batch:
            tensor = item[k]
            pad_len = max_len - tensor.shape[0]
            if pad_len > 0:
                # Pad with zeros
                padding = torch.zeros((pad_len, *tensor.shape[1:]))
                padded = torch.cat([tensor, padding], dim=0)
            else:
                padded = tensor
            batch_dict[k].append(padded)
        batch_dict[k] = torch.stack(batch_dict[k])
        
    # Pad text
    texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=False, padding_value=0) # (T, B)
    
    # Convert lengths to tensor
    src_lengths = torch.tensor(src_lengths, dtype=torch.long)
    
    return batch_dict, texts, src_lengths
