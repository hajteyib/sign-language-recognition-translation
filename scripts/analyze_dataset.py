"""
Analyse du dataset Phoenix-2014T pour comprendre les patterns et la diversité.
"""

import sys
import os
import json
import gzip
from collections import Counter, defaultdict
import numpy as np

sys.path.append('.')
from utils.vocabulary import Vocabulary

def load_annotations(annotation_file):
    """Load annotations from gzip file."""
    import pickle
    with gzip.open(annotation_file, 'rb') as f:
        data = pickle.load(f)
    return data

def analyze_dataset(split='train', annotations_file=None):
    """Analyze dataset statistics."""
    
    print(f"\n{'='*70}")
    print(f"📊 ANALYSE DU DATASET - {split.upper()}")
    print(f"{'='*70}\n")
    
    # Load data
    if annotations_file is None:
        annotations_file = f'data/raw/phoenix14t.pami0.{split}.annotations_only.gzip'
    
    data = load_annotations(annotations_file)
    
    # Basic stats
    num_samples = len(data)
    print(f"📌 Statistiques de base:")
    print(f"  - Nombre d'échantillons: {num_samples}")
    
    # Text analysis
    all_texts = [item['translation'][0] for item in data]
    all_words = []
    sentence_lengths = []
    
    for text in all_texts:
        words = text.split()
        all_words.extend(words)
        sentence_lengths.append(len(words))
    
    # Word statistics
    word_freq = Counter(all_words)
    unique_words = len(word_freq)
    total_words = len(all_words)
    
    print(f"\n📝 Statistiques textuelles:")
    print(f"  - Mots totaux: {total_words}")
    print(f"  - Mots uniques: {unique_words}")
    print(f"  - Vocabulaire coverage: {unique_words/total_words*100:.1f}%")
    print(f"  - Longueur moyenne des phrases: {np.mean(sentence_lengths):.1f} mots")
    print(f"  - Longueur min/max: {min(sentence_lengths)} / {max(sentence_lengths)} mots")
    
    # Most common words
    print(f"\n🔤 Top 20 mots les plus fréquents:")
    for word, freq in word_freq.most_common(20):
        print(f"  - '{word}': {freq} ({freq/total_words*100:.1f}%)")
    
    # Sentence diversity
    unique_sentences = len(set(all_texts))
    duplicate_rate = (1 - unique_sentences/num_samples) * 100
    
    print(f"\n🎨 Diversité des phrases:")
    print(f"  - Phrases uniques: {unique_sentences}/{num_samples}")
    print(f"  - Taux de duplication: {duplicate_rate:.1f}%")
    
    # Find duplicates
    sentence_counts = Counter(all_texts)
    duplicates = [(sent, count) for sent, count in sentence_counts.items() if count > 1]
    duplicates.sort(key=lambda x: x[1], reverse=True)
    
    if duplicates:
        print(f"\n🔁 Top 10 phrases répétées:")
        for sent, count in duplicates[:10]:
            print(f"  - ({count}×) {sent[:80]}...")
    
    # N-gram analysis
    print(f"\n📊 Analyse des patterns communs:")
    
    # Bigrams
    bigrams = []
    for text in all_texts:
        words = text.split()
        for i in range(len(words)-1):
            bigrams.append(f"{words[i]} {words[i+1]}")
    
    bigram_freq = Counter(bigrams)
    print(f"  Top 10 bigrammes:")
    for bigram, freq in bigram_freq.most_common(10):
        print(f"    - '{bigram}': {freq} fois")
    
    # Trigrams
    trigrams = []
    for text in all_texts:
        words = text.split()
        for i in range(len(words)-2):
            trigrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    
    trigram_freq = Counter(trigrams)
    print(f"\n  Top 10 trigrammes:")
    for trigram, freq in trigram_freq.most_common(10):
        print(f"    - '{trigram}': {freq} fois")
    
    # Sentence starters
    starters = [text.split()[0] if text.split() else '' for text in all_texts]
    starter_freq = Counter(starters)
    print(f"\n🚀 Débuts de phrases:")
    for starter, freq in starter_freq.most_common(10):
        print(f"  - '{starter}': {freq} ({freq/num_samples*100:.1f}%)")
    
    # Generic phrases detection
    print(f"\n⚠️ Détection de phrases génériques:")
    generic_patterns = [
        "am tag",
        "am sonntag",
        "im norden und südosten",
        "teilweise kräftige regenfälle",
        "noch längere zeit freundlich",
        "minus",
        "grad"
    ]
    
    for pattern in generic_patterns:
        count = sum(1 for text in all_texts if pattern in text.lower())
        if count > 0:
            print(f"  - '{pattern}': {count}/{num_samples} ({count/num_samples*100:.1f}%)")
    
    # Save detailed analysis
    output_file = f'data/dataset_analysis_{split}.json'
    analysis = {
        'split': split,
        'num_samples': num_samples,
        'unique_words': unique_words,
        'total_words': total_words,
        'unique_sentences': unique_sentences,
        'duplicate_rate': duplicate_rate,
        'avg_sentence_length': float(np.mean(sentence_lengths)),
        'top_words': word_freq.most_common(50),
        'top_bigrams': bigram_freq.most_common(20),
        'top_trigrams': trigram_freq.most_common(20),
        'duplicates': duplicates[:20]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Analyse détaillée sauvegardée dans: {output_file}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Phoenix dataset')
    parser.add_argument('--split', type=str, default='train', 
                       choices=['train', 'dev', 'test'],
                       help='Dataset split to analyze')
    
    args = parser.parse_args()
    
    analyze_dataset(args.split)
