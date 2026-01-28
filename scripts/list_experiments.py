#!/usr/bin/env python
"""
Script to list and manage experiments.
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.experiment_tracker import list_experiments

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='List experiments')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory containing experiments')
    
    args = parser.parse_args()
    
    list_experiments(args.checkpoint_dir)
