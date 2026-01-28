import numpy as np
import pickle
import os
from pathlib import Path

class LandmarkNormalizer:
    """
    Normalizes landmarks to zero mean and unit variance.
    Computes statistics from training data and applies them consistently.
    """
    def __init__(self, stats_path=None):
        """
        Args:
            stats_path: Path to saved normalization statistics (mean, std)
        """
        self.stats = None
        if stats_path and os.path.exists(stats_path):
            self.load_stats(stats_path)
    
    def compute_stats(self, landmarks_list):
        """
        Compute normalization statistics from a list of landmark dicts.
        
        Args:
            landmarks_list: List of dicts with keys ['pose', 'left_hand', 'right_hand', 'face']
                           Each value is a numpy array of shape (T, N, C)
        
        Returns:
            stats: Dict with mean and std for each body part
        """
        # Accumulate all landmarks
        accumulated = {
            'pose': [],
            'left_hand': [],
            'right_hand': [],
            'face': []
        }
        
        for landmarks in landmarks_list:
            for key in accumulated.keys():
                if key in landmarks:
                    accumulated[key].append(landmarks[key])
        
        # Compute mean and std for each part
        stats = {}
        for key, data_list in accumulated.items():
            if len(data_list) > 0:
                # Concatenate along time dimension: (Total_T, N, C)
                all_data = np.concatenate(data_list, axis=0)
                
                # Compute mean and std across time and landmarks: shape (C,)
                # We want per-coordinate statistics (x, y, z, visibility)
                mean = np.mean(all_data, axis=(0, 1))  # (C,)
                std = np.std(all_data, axis=(0, 1)) + 1e-8  # (C,) + epsilon for stability
                
                stats[key] = {
                    'mean': mean,
                    'std': std
                }
            else:
                # Default to identity normalization
                stats[key] = {
                    'mean': np.array([0.0]),
                    'std': np.array([1.0])
                }
        
        self.stats = stats
        return stats
    
    def normalize(self, landmarks):
        """
        Normalize a single landmarks dict using stored statistics.
        
        Args:
            landmarks: Dict with keys ['pose', 'left_hand', 'right_hand', 'face']
                      Each value is numpy array of shape (T, N, C)
        
        Returns:
            normalized: Dict with same structure, normalized values
        """
        if self.stats is None:
            raise ValueError("Stats not computed or loaded. Call compute_stats() or load_stats() first.")
        
        normalized = {}
        for key in ['pose', 'left_hand', 'right_hand', 'face']:
            if key in landmarks:
                data = landmarks[key]
                mean = self.stats[key]['mean']
                std = self.stats[key]['std']
                
                # Broadcasting: (T, N, C) - (C,) elementwise
                normalized[key] = (data - mean) / std
            else:
                normalized[key] = landmarks[key]
        
        return normalized
    
    def save_stats(self, path):
        """Save normalization statistics to file."""
        if self.stats is None:
            raise ValueError("No stats to save. Call compute_stats() first.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.stats, f)
        print(f"Normalization stats saved to {path}")
    
    def load_stats(self, path):
        """Load normalization statistics from file."""
        with open(path, 'rb') as f:
            self.stats = pickle.load(f)
        print(f"Normalization stats loaded from {path}")


def compute_normalization_stats_from_dataset(dataset, output_path='data/norm_stats.pkl', max_samples=None):
    """
    Utility function to compute normalization stats from a dataset.
    
    Args:
        dataset: PhoenixDataset or similar with __getitem__ returning landmarks
        output_path: Where to save the computed statistics
        max_samples: Limit number of samples to process (for speed)
    """
    normalizer = LandmarkNormalizer()
    
    landmarks_list = []
    n_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    
    print(f"Computing normalization statistics from {n_samples} samples...")
    for i in range(n_samples):
        item = dataset.data[i]  # Access raw data path
        landmarks = np.load(item['path'], allow_pickle=True).item()
        landmarks_list.append(landmarks)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_samples} samples")
    
    stats = normalizer.compute_stats(landmarks_list)
    normalizer.save_stats(output_path)
    
    # Print statistics summary
    print("\nNormalization Statistics Summary:")
    for key, stat in stats.items():
        print(f"  {key}:")
        print(f"    Mean: {stat['mean']}")
        print(f"    Std:  {stat['std']}")
    
    return normalizer
