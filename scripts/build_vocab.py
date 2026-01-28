import argparse
import sys
import os
import gzip
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.vocabulary import Vocabulary

def main():
    parser = argparse.ArgumentParser(description='Build vocabulary from training annotations')
    parser.add_argument('--annotation_file', type=str, 
                       default='data/raw/phoenix14t.pami0.train.annotations_only.gzip',
                       help='Path to training annotation file')
    parser.add_argument('--output_path', type=str, 
                       default='data/vocab.pkl',
                       help='Path to save vocabulary')
    parser.add_argument('--min_freq', type=int, default=1,
                       help='Minimum word frequency to include in vocabulary')
    
    args = parser.parse_args()
    
    print(f"Loading annotations from {args.annotation_file}...")
    
    # Load training annotations
    with gzip.open(args.annotation_file, 'rb') as f:
        annotations = pickle.load(f)
    
    print(f"Loaded {len(annotations)} annotations")
    
    # Extract all text
    texts = [item['text'] for item in annotations]
    
    print(f"\nBuilding vocabulary (min_freq={args.min_freq})...")
    
    # Build vocabulary
    vocab = Vocabulary(min_freq=args.min_freq)
    vocab.build_from_texts(texts)
    
    # Save vocabulary
    vocab.save(args.output_path)
    
    # Print statistics
    vocab.stats()
    
    print(f"\n✓ Vocabulary successfully built and saved to {args.output_path}")

if __name__ == "__main__":
    main()
