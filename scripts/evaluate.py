import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import sys
import os
import json
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import SignLanguageTransformer
from utils.dataset import PhoenixDataset, collate_fn
from utils.decoder import beam_search_decode
from utils.vocabulary import Vocabulary

def calculate_bleu(references, hypotheses):
    """
    Simple BLEU score calculation.
    For production, use libraries like sacrebleu or nltk.bleu
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        
        # Format for BLEU: references = [[[ref1_tokens], [ref2_tokens]], ...]
        # hypotheses = [[hyp1_tokens], [hyp2_tokens], ...]
        refs = [[ref.split()] for ref in references]
        hyps = [hyp.split() for hyp in hypotheses]
        
        smoothing = SmoothingFunction().method1
        bleu = corpus_bleu(refs, hyps, smoothing_function=smoothing)
        return bleu
    except ImportError:
        print("Warning: NLTK not installed. BLEU score not computed.")
        print("Install with: pip install nltk")
        return None

def calculate_wer(references, hypotheses):
    """
    Calculate Word Error Rate.
    """
    try:
        from jiwer import wer
        return wer(references, hypotheses)
    except ImportError:
        print("Warning: jiwer not installed. WER not computed.")
        print("Install with: pip install jiwer")
        return None

def greedy_decode(model, src_dict, src_lengths, vocab, max_len=100, device='cpu'):
    """
    Greedy decoding for inference.
    """
    model.eval()
    
    batch_size = src_dict['pose'].shape[0]
    
    # Start with SOS token
    ys = torch.full((1, batch_size), vocab.sos_idx, dtype=torch.long, device=device)
    
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

def evaluate(args):
    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Load vocabulary
    vocab = Vocabulary.load(args.vocab_path)
    vocab_size = len(vocab)
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = PhoenixDataset(
        args.landmarks_dir,
        args.annotation_file,
        args.vocab_path,
        split=args.split,
        normalizer_path=args.norm_stats_path,
        augment=False
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Loaded {len(dataset)} samples")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Model dimensions
    input_dims = {
        'pose': 33 * 4,
        'left_hand': 21 * 3,
        'right_hand': 21 * 3,
        'face': 468 * 3
    }
    
    model = SignLanguageTransformer(
        input_dims=input_dims,
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded (trained for {checkpoint.get('epoch', '?')} epochs)")
    
    # Evaluation
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    total_loss = 0
    all_references = []
    all_hypotheses = []
    num_batches = 0
    
    print(f"\nEvaluating on {args.split} set...")
    
    with torch.no_grad():
        for src_dict, tgt, src_lengths in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            for k in src_dict:
                src_dict[k] = src_dict[k].to(device)
                B, T, N, C = src_dict[k].shape
                src_dict[k] = src_dict[k].view(B, T, -1)
            
            tgt = tgt.to(device)
            src_lengths = src_lengths.to(device)
            
            # Teacher forcing for loss calculation
            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]
            
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(0)).to(device)
            src_key_padding_mask = model.create_src_padding_mask(src_dict, src_lengths)
            tgt_key_padding_mask = (tgt_input == 0).transpose(0, 1)
            
            output = model(src_dict, tgt_input,
                          src_key_padding_mask=src_key_padding_mask,
                          tgt_mask=tgt_mask,
                          tgt_key_padding_mask=tgt_key_padding_mask)
            
            loss = criterion(output.view(-1, vocab_size), tgt_output.reshape(-1))
            total_loss += loss.item()
            num_batches += 1
            
            
            # Beam search decoding for BLEU/WER (better quality than greedy)
            batch_size = tgt.shape[1]
            for b in range(batch_size):
                # Extract single sample
                src_single = {k: v[b:b+1] for k, v in src_dict.items()}
                src_len_single = src_lengths[b:b+1]
                
                # Beam search decode
                pred_indices = beam_search_decode(
                    model, src_single, src_len_single, vocab,
                    max_len=100, beam_size=5, 
                    length_penalty=0.6, repetition_penalty=1.2,
                    device=device
                )
                
                ref_indices = tgt[:, b].cpu()
                
                pred_text = vocab.decode(torch.tensor(pred_indices), skip_special_tokens=True)
                ref_text = vocab.decode(ref_indices, skip_special_tokens=True)
                
                all_hypotheses.append(pred_text)
                all_references.append(ref_text)
    
    # Calculate metrics
    avg_loss = total_loss / num_batches
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Average Loss: {avg_loss:.4f}")
    
    # BLEU score
    bleu = calculate_bleu(all_references, all_hypotheses)
    if bleu is not None:
        print(f"BLEU Score:   {bleu:.4f}")
    
    # WER
    wer_score = calculate_wer(all_references, all_hypotheses)
    if wer_score is not None:
        print(f"WER:          {wer_score:.4f}")
    
    print("=" * 60)
    
    # Show some examples
    print("\nExample Predictions:")
    print("-" * 60)
    for i in range(min(5, len(all_references))):
        print(f"\nExample {i+1}:")
        print(f"  Reference:  {all_references[i]}")
        print(f"  Hypothesis: {all_hypotheses[i]}")
    print("-" * 60)
    
    # Save results
    if args.output_file:
        results = {
            'split': args.split,
            'checkpoint': args.checkpoint,
            'num_samples': len(dataset),
            'loss': avg_loss,
            'bleu': bleu,
            'wer': wer_score,
            'examples': [
                {'reference': ref, 'hypothesis': hyp}
                for ref, hyp in zip(all_references[:20], all_hypotheses[:20])
            ]
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Sign Language Translation Model')
    
    # Data paths
    parser.add_argument('--landmarks_dir', type=str, default='data/processed/landmarks')
    parser.add_argument('--annotation_file', type=str,
                       default='data/raw/phoenix14t.pami0.dev.annotations_only.gzip')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl')
    parser.add_argument('--norm_stats_path', type=str, default='data/norm_stats.pkl')
    parser.add_argument('--split', type=str, default='dev', choices=['train', 'dev', 'test'])
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # Evaluation
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save evaluation results (JSON)')
    
    args = parser.parse_args()
    
    evaluate(args)
