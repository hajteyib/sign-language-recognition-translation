import torch
import argparse
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import SignLanguageTransformer
from utils.vocabulary import Vocabulary
from utils.preprocessing import LandmarkNormalizer

def greedy_decode(model, src_dict, src_lengths, vocab, max_len=100, device='cpu'):
    """Greedy decoding for inference."""
    model.eval()
    
    batch_size = src_dict['pose'].shape[0]
    
    # Start with SOS token
    ys = torch.full((1, batch_size), vocab.sos_idx, dtype=torch.long, device=device)
    
    with torch.no_grad():
        for i in range(max_len):
            # Create masks
            tgt_mask = model.generate_square_subsequent_mask(ys.size(0)).to(device)
            src_key_padding_mask = model.create_src_padding_mask(src_dict, src_lengths)
            
            # Forward pass
            out = model(src_dict, ys,
                       src_key_padding_mask=src_key_padding_mask,
                       tgt_mask=tgt_mask)
            
            # Get next token (greedy)
            prob = out[-1, :, :]  # (Batch, Vocab)
            next_word = torch.argmax(prob, dim=-1).unsqueeze(0)  # (1, Batch)
            
            ys = torch.cat([ys, next_word], dim=0)
            
            # Stop if all sequences have generated EOS
            if (next_word == vocab.eos_idx).all():
                break
    
    return ys  # (Time, Batch)

def translate_landmarks(landmarks_path, model, vocab, normalizer, device):
    """Translate a single landmarks file."""
    
    # Load landmarks
    landmarks = np.load(landmarks_path, allow_pickle=True).item()
    
    # Normalize
    if normalizer is not None:
        landmarks = normalizer.normalize(landmarks)
    
    # Convert to tensors and add batch dimension
    src_dict = {
        'pose': torch.tensor(landmarks['pose'], dtype=torch.float32).unsqueeze(0),  # (1, T, N, C)
        'left_hand': torch.tensor(landmarks['left_hand'], dtype=torch.float32).unsqueeze(0),
        'right_hand': torch.tensor(landmarks['right_hand'], dtype=torch.float32).unsqueeze(0),
        'face': torch.tensor(landmarks['face'], dtype=torch.float32).unsqueeze(0)
    }
    
    # Move to device and flatten
    for k in src_dict:
        src_dict[k] = src_dict[k].to(device)
        B, T, N, C = src_dict[k].shape
        src_dict[k] = src_dict[k].view(B, T, -1)
    
    # Sequence length
    src_lengths = torch.tensor([landmarks['pose'].shape[0]], dtype=torch.long, device=device)
    
    # Decode
    output = greedy_decode(model, src_dict, src_lengths, vocab, max_len=100, device=device)
    
    # Decode to text
    translation = vocab.decode(output[:, 0].cpu(), skip_special_tokens=True)
    
    return translation

def main():
    parser = argparse.ArgumentParser(description='Demo: Translate sign language landmarks to text')
    
    # Paths
    parser.add_argument('--landmarks', type=str, required=True,
                       help='Path to landmarks .npy file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl')
    parser.add_argument('--norm_stats_path', type=str, default='data/norm_stats.pkl')
    
    # Model config
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    args = parser.parse_args()
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}\n")
    
    # Load vocabulary
    print(f"Loading vocabulary from {args.vocab_path}...")
    vocab = Vocabulary.load(args.vocab_path)
    
    # Load normalizer
    normalizer = None
    if os.path.exists(args.norm_stats_path):
        print(f"Loading normalizer from {args.norm_stats_path}...")
        normalizer = LandmarkNormalizer(args.norm_stats_path)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    
    input_dims = {
        'pose': 33 * 4,
        'left_hand': 21 * 3,
        'right_hand': 21 * 3,
        'face': 468 * 3
    }
    
    model = SignLanguageTransformer(
        input_dims=input_dims,
        vocab_size=len(vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!\n")
    
    # Translate
    print(f"Translating {args.landmarks}...")
    translation = translate_landmarks(args.landmarks, model, vocab, normalizer, device)
    
    print("\n" + "=" * 60)
    print("Translation Result:")
    print("=" * 60)
    print(f"\n{translation}\n")
    print("=" * 60)

if __name__ == "__main__":
    main()
