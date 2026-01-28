import argparse
import sys
import os
import gzip
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import compute_normalization_stats_from_dataset
from utils.dataset import PhoenixDataset
from utils.vocabulary import Vocabulary

def main():
    parser = argparse.ArgumentParser(description='Compute normalization statistics from training data')
    parser.add_argument('--landmarks_dir', type=str, 
                       default='data/processed/landmarks',
                       help='Directory containing landmark files')
    parser.add_argument('--annotation_file', type=str, 
                       default='data/raw/phoenix14t.pami0.train.annotations_only.gzip',
                       help='Path to training annotation file')
    parser.add_argument('--vocab_path', type=str, 
                       default='data/vocab.pkl',
                       help='Path to vocabulary file')
    parser.add_argument('--output_path', type=str, 
                       default='data/norm_stats.pkl',
                       help='Path to save normalization statistics')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use (for speed)')
    
    args = parser.parse_args()
    
    print("Creating temporary dataset (without normalization)...")
    
    # Create a temporary dataset without normalization
    # We need a minimal dataset just to access the file paths
    dataset = PhoenixDataset(
        root_dir=args.landmarks_dir,
        annotation_file=args.annotation_file,
        vocab_path=args.vocab_path,
        split='train',
        normalizer_path=None,  # No normalization yet!
        augment=False
    )
    
    print(f"\nComputing normalization statistics from {len(dataset)} samples...")
    if args.max_samples:
        print(f"(Limited to {args.max_samples} samples)")
    
    # Compute and save statistics
    compute_normalization_stats_from_dataset(
        dataset, 
        output_path=args.output_path,
        max_samples=args.max_samples
    )
    
    print(f"\n✓ Normalization statistics successfully computed and saved to {args.output_path}")

if __name__ == "__main__":
    main()
