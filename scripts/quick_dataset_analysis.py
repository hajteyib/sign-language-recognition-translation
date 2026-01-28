"""
Analyse rapide du dataset en utilisant les données déjà chargées par le DataLoader.
"""

import sys
import os
import json
from collections import Counter
import numpy as np

sys.path.append('.')
from utils.vocabulary import Vocabulary
from utils.dataset import PhoenixDataset

def quick_analysis():
    """Quick dataset analysis using existing data."""
    
    print(f"\n{'='*70}")
    print(f"📊 ANALYSE RAPIDE DU DATASET PHOENIX-2014T")
    print(f"{'='*70}\n")
    
    # Load vocabulary
    vocab = Vocabulary.load('data/vocab.pkl')
    
    print(f"📚 Vocabulaire:")
    print(f"  - Taille totale: {len(vocab)} mots")
    print(f"  - Mots réguliers: {len(vocab) - 4}")
    
    # Most common words
    print(f"\n🔤 Top 30 mots les plus fréquents:")
    for word, freq in vocab.get_most_common(30):
        print(f"  - '{word}': {freq} occurrences")
    
    # Load datasets
    print(f"\n📦 Chargement des datasets...")
    
    train_dataset = PhoenixDataset(
        'data/processed/landmarks',
        'data/raw/phoenix14t.pami0.train.annotations_only.gzip',
        'data/vocab.pkl',
        split='train',
        normalizer_path='data/norm_stats.pkl',
        augment=False
    )
    
    dev_dataset = PhoenixDataset(
        'data/processed/landmarks',
        'data/raw/phoenix14t.pami0.dev.annotations_only.gzip',
        'data/vocab.pkl',
        split='dev',
        normalizer_path='data/norm_stats.pkl',
        augment=False
    )
    
    test_dataset = PhoenixDataset(
        'data/processed/landmarks',
        'data/raw/phoenix14t.pami0.test.annotations_only.gzip',
        'data/vocab.pkl',
        split='test',
        normalizer_path='data/norm_stats.pkl',
        augment=False
    )
    
    print(f"\n📊 Tailles des datasets:")
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Dev: {len(dev_dataset)} samples")
    print(f"  - Test: {len(test_dataset)} samples")
    print(f"  - Total: {len(train_dataset) + len(dev_dataset) + len(test_dataset)} samples")
    
    # Analyze sentence lengths from train
    print(f"\n📏 Analyse de {len(train_dataset)} phrases train:")
    
    sentence_lengths = []
    all_words = []
    all_sentences = []
    
    for i in range(len(train_dataset)):
        sample = train_dataset.samples[i]
        text = sample['translation']
        all_sentences.append(text)
        words = text.split()
        sentence_lengths.append(len(words))
        all_words.extend(words)
    
    print(f"  - Longueur moyenne: {np.mean(sentence_lengths):.1f} mots")
    print(f"  - Longueur médiane: {np.median(sentence_lengths):.0f} mots")
    print(f"  - Longueur min/max: {min(sentence_lengths)} / {max(sentence_lengths)} mots")
    print(f"  - Écart-type: {np.std(sentence_lengths):.1f}")
    
    # Sentence diversity
    unique_sentences = len(set(all_sentences))
    duplicate_rate = (1 - unique_sentences/len(all_sentences)) * 100
    
    print(f"\n🎨 Diversité:")
    print(f"  - Phrases uniques: {unique_sentences}/{len(all_sentences)}")
    print(f"  - Taux de duplication: {duplicate_rate:.1f}%")
    
    # Find most common phrases
    sentence_counts = Counter(all_sentences)
    duplicates = [(sent, count) for sent, count in sentence_counts.items() if count > 1]
    duplicates.sort(key=lambda x: x[1], reverse=True)
    
    if duplicates:
        print(f"\n🔁 Top 10 phrases les plus répétées:")
        for sent, count in duplicates[:10]:
            print(f"  - ({count}×) {sent[:70]}...")
    
    # Word frequency in actual corpus
    word_freq = Counter(all_words)
    print(f"\n📈 Distribution des mots (corpus réel):")
    print(f"  - Mots totaux utilisés: {len(all_words)}")
    print(f"  - Mots uniques utilisés: {len(word_freq)}")
    
    # Analyze common starting patterns
    starters = Counter([s.split()[0] if s.split() else '' for s in all_sentences])
    print(f"\n🚀 Débuts de phrases communs:")
    for starter, count in starters.most_common(15):
        print(f"  - '{starter}': {count} ({count/len(all_sentences)*100:.1f}%)")
    
    # Check for weather-specific patterns
    print(f"\n🌤️ Patterns météo communs:")
    weather_patterns = [
        ("am tag", "Temps du jour"),
        ("am sonntag", "Dimanche"),
        ("im norden", "Au nord"),
        ("im süden", "Au sud"),
        ("grad", "Degrés"),
        ("regen", "Pluie"),
        ("sonne", "Soleil"),
        ("wolken", "Nuages"),
        ("wind", "Vent"),
        ("schnee", "Neige")
    ]
    
    for pattern, desc in weather_patterns:
        count = sum(1 for s in all_sentences if pattern in s.lower())
        if count > 0:
            print(f"  - '{pattern}' ({desc}): {count} ({count/len(all_sentences)*100:.1f}%)")
    
    # Save analysis
    analysis = {
        'vocab_size': len(vocab),
        'train_samples': len(train_dataset),
        'dev_samples': len(dev_dataset),
        'test_samples': len(test_dataset),
        'avg_sentence_length': float(np.mean(sentence_lengths)),
        'median_sentence_length': float(np.median(sentence_lengths)),
        'unique_sentences': unique_sentences,
        'duplicate_rate': duplicate_rate,
        'top_starters': starters.most_common(20),
        'top_duplicates': duplicates[:20],
        'top_words': word_freq.most_common(50)
    }
    
    with open('data/quick_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Analyse sauvegardée dans: data/quick_analysis.json")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    quick_analysis()
