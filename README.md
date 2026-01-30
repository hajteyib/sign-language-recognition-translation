# Sign Language Recognition and Translation (SLRT)

Deep Learning-based German Sign Language (DGS) to German text translation using Transformer architecture and MediaPipe landmarks.

## 📋 Overview

This project implements an end-to-end pipeline for recognizing and translating German Sign Language (Deutsche Gebärdensprache - DGS) into written German text. Using the Phoenix-2014T dataset of weather broadcast sign language videos, the system extracts spatial landmarks via MediaPipe and translates them to text using a Transformer model.

### Key Features

- **Landmark-based approach**: Efficient sign language representation using MediaPipe (543 body landmarks)
- **Transformer architecture**: State-of-the-art sequence-to-sequence model
- **Data augmentation**: Spatial and temporal augmentations for improved generalization
- **Repetition penalty**: Eliminates infinite repetitions in generated text
- **Mac MPS support**: Optimized for Apple Silicon GPUs

## 🎯 Results

| Configuration | Val Loss | Test Loss | BLEU | Training Time |
|--------------|----------|-----------|------|---------------|
| Baseline (1300 samples, 384-dim) | 4.62 | 3.46 | 23.1 | 2h58 |
| Large model (1300 samples, 512-dim) | 4.62 | 3.54 | 19.5 | 2h42 |
| **Final (2000 samples, 448-dim, aug)** | **4.42** | **3.32** | **27.4** | **2h04** |

**Improvements**: -4.3% validation loss, -4.0% test loss, +18.6% BLEU score, 60% reduction in mode collapse

## 🏗️ Architecture

```
Vidéo LSA → MediaPipe → Landmarks (543 pts) → Normalization
                                ↓
                        Spatial Embedding (448-dim)
                                ↓
                        Positional Encoding
                                ↓
                Transformer Encoder (4 layers, 8 heads)
                                ↓
                Transformer Decoder (4 layers, 8 heads)
                                ↓
                    Linear Layer (448 → 2892)
                                ↓
                            Softmax
                                ↓
                        Texte Allemand
```

**Model Parameters**:
- d_model: 448
- Encoder/Decoder layers: 4 each
- Attention heads: 8
- Total parameters: 27.1M
- Vocabulary size: 2892

## 📦 Project Structure

```
.
├── models/
│   └── transformer.py          # Transformer architecture
├── utils/
│   ├── preprocessing.py        # MediaPipe landmark extraction
│   ├── dataset.py             # Phoenix-2014T data loader
│   ├── augmentation.py        # Data augmentation
│   ├── vocabulary.py          # Vocabulary management
│   ├── decoder.py             # Beam search + repetition penalty
│   └── experiment_tracker.py  # Training tracking
├── scripts/
│   ├── extract_landmarks.py   # Extract landmarks from videos
│   ├── build_vocab.py         # Build vocabulary
│   ├── compute_norm_stats.py  # Compute normalization stats
│   ├── train.py               # Training script
│   └── evaluate.py            # Evaluation script
├── run_extraction.sh          # Landmark extraction launcher
├── run_training.sh            # Training launcher
├── run_evaluation.sh          # Evaluation launcher
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PyTorch 2.x (with MPS support for Mac)
- MediaPipe
- Phoenix-2014T dataset

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sign-language-translation.git
cd sign-language-translation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install mediapipe opencv-python numpy pandas tqdm
```

### Dataset Setup

Download the Phoenix-2014T dataset and place it in the following structure:

```
data/
├── raw/
│   ├── phoenix14t.pami0.train.annotations_only.gzip
│   ├── phoenix14t.pami0.dev.annotations_only.gzip
│   └── phoenix14t.pami0.test.annotations_only.gzip
└── videos/
    └── [video files]
```

### Pipeline Execution

**1. Extract Landmarks**
```bash
./run_extraction.sh --split train --max_samples 2000
./run_extraction.sh --split dev --max_samples 200
```

**2. Build Vocabulary**
```bash
python scripts/build_vocab.py
```

**3. Compute Normalization Statistics**
```bash
python scripts/compute_norm_stats.py --max_samples 2000
```

**4. Train Model**
```bash
./run_training.sh \
  --exp_name my_experiment \
  --d_model 448 \
  --nhead 8 \
  --num_encoder_layers 4 \
  --num_decoder_layers 4 \
  --batch_size 8 \
  --epochs 50 \
  --lr 3e-4 \
  --dropout 0.25
```

**5. Evaluate**
```bash
./run_evaluation.sh \
  --checkpoint checkpoints/my_experiment/models/best_model.pt \
  --split test
```

## 🔬 Key Components

### Data Augmentation

The pipeline includes several augmentation techniques:
- **Spatial**: Rotation (±10°), scaling (95-105%), translation (±3%)
- **Temporal**: Frame masking (10% probability)
- Applied with 50% probability during training

### Repetition Penalty

Custom n-gram repetition penalty prevents infinite loops:
```python
score_adjusted = score_original - λ × count(n-gram)
```
where λ = 0.15 (tuned experimentally)

### Training Details

- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.01)
- **Scheduler**: Linear warmup (800 steps) + cosine annealing
- **Loss**: Cross-entropy with label smoothing (0.15)
- **Early stopping**: Patience of 8 epochs
- **Gradient clipping**: Max norm 1.0

## 📊 Dataset

**Phoenix-2014T** (Weather broadcast sign language):
- Training: 2000 samples (28% of full dataset)
- Validation: 200 samples
- Test: 200 samples
- Vocabulary: 2892 unique German words
- Domain: Weather forecasts (inherently repetitive)

**Limitations**: The weather domain leads to naturally repetitive vocabulary and phrases, making complete elimination of mode collapse challenging.

## 📈 Experimental Results

### Ablation Studies

| Configuration | Impact |
|--------------|--------|
| +700 samples (1300→2000) | -2% val loss, +20% diversity |
| Data augmentation | -2% val loss, **+80% diversity** |
| 448-dim architecture | -0.5% val loss, +10% diversity |
| Weight decay 0.01 | -0.3% val loss |

**Key finding**: Data augmentation is the most impactful optimization for diversity.

## 🛠️ Hardware

All experiments conducted on:
- **MacBook** with Apple Silicon (M1)
- **Backend**: PyTorch MPS (Metal Performance Shaders)




## 📚 References

- [Phoenix-2014T Dataset](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [MediaPipe](https://google.github.io/mediapipe/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**El haj Samitt Ebou**  
M2 Vision et Machine Intelligente  
Université Paris Cité

---


