"""
Decoder utilities for improved sequence generation.
Includes beam search, repetition penalty, and other decoding strategies.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


def compute_repetition_penalty(logits: torch.Tensor, 
                                generated_tokens: torch.Tensor,
                                penalty: float = 1.2) -> torch.Tensor:
    """
    Apply repetition penalty to logits to discourage repeating tokens.
    
    Args:
        logits: (vocab_size,) unnormalized logits for next token
        generated_tokens: (seq_len,) previously generated tokens
        penalty: penalty factor (> 1.0 discourages repetition)
    
    Returns:
        Modified logits with penalty applied
    """
    if len(generated_tokens) == 0:
        return logits
    
    # Get unique tokens that have been generated
    unique_tokens = generated_tokens.unique()
    
    # Apply penalty to previously generated tokens
    logits[unique_tokens] = logits[unique_tokens] / penalty
    
    return logits


def compute_ngram_repetition_loss(output: torch.Tensor, 
                                   n: int = 3,
                                   penalty_weight: float = 0.1) -> torch.Tensor:
    """
    Compute loss penalty for repeated n-grams in generated sequence.
    Used during training to discourage repetitive patterns.
    
    Args:
        output: (seq_len, batch, vocab_size) model output logits
        n: n-gram size (default 3 for trigrams)
        penalty_weight: weight of penalty in total loss
    
    Returns:
        Scalar penalty loss
    """
    seq_len, batch_size, vocab_size = output.shape
    
    if seq_len < n:
        return torch.tensor(0.0, device=output.device)
    
    # Get predicted tokens (greedy)
    pred_tokens = output.argmax(dim=-1)  # (seq_len, batch)
    
    penalties = []
    
    for b in range(batch_size):
        tokens = pred_tokens[:, b]
        
        # Extract all n-grams
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n].tolist())
            ngrams.append(ngram)
        
        # Count repetitions
        if len(ngrams) > 0:
            unique_ngrams = len(set(ngrams))
            total_ngrams = len(ngrams)
            repetition_ratio = 1.0 - (unique_ngrams / total_ngrams)
            penalties.append(repetition_ratio)
    
    if penalties:
        return torch.tensor(sum(penalties) / len(penalties), 
                           device=output.device) * penalty_weight
    return torch.tensor(0.0, device=output.device)


def beam_search_decode(model, 
                       src_dict: dict,
                       src_lengths: torch.Tensor,
                       vocab,
                       max_len: int = 100,
                       beam_size: int = 5,
                       length_penalty: float = 0.6,
                       repetition_penalty: float = 1.2,
                       device: str = 'cpu') -> List[int]:
    """
    Beam search decoding with repetition penalty and length normalization.
    
    Args:
        model: Trained transformer model
        src_dict: Source features dictionary
        src_lengths: Source sequence lengths
        vocab: Vocabulary object
        max_len: Maximum generation length
        beam_size: Number of beams
        length_penalty: Length normalization factor (0.6 typical)
        repetition_penalty: Penalty for repeated tokens (>1.0)
        device: Device to run on
    
    Returns:
        List of predicted token IDs
    """
    model.eval()
    
    # Move source to device
    for k in src_dict:
        src_dict[k] = src_dict[k].to(device)
    src_lengths = src_lengths.to(device)
    
    # Special tokens
    sos_idx = vocab.sos_idx
    eos_idx = vocab.eos_idx
    pad_idx = vocab.pad_idx
    
    # Initialize beams: (beam_size, seq_len)
    beams = [[sos_idx] for _ in range(beam_size)]
    beam_scores = torch.zeros(beam_size, device=device)
    finished_beams = []
    
    with torch.no_grad():
        for step in range(max_len):
            all_candidates = []
            
            for beam_idx, beam in enumerate(beams):
                if beam[-1] == eos_idx:
                    # Beam already finished
                    finished_beams.append((beam, beam_scores[beam_idx]))
                    continue
                
                # Prepare input
                tgt_input = torch.tensor(beam, device=device).unsqueeze(1)  # (len, 1)
                
                # Generate tgt_mask
                tgt_mask = model.generate_square_subsequent_mask(len(beam)).to(device)
                
                # Forward pass
                output = model(src_dict, tgt_input, tgt_mask=tgt_mask)
                
                # Get logits for last position
                logits = output[-1, 0, :]  # (vocab_size,)
                
                # Apply repetition penalty
                generated = torch.tensor(beam[1:], device=device)  # Exclude SOS
                logits = compute_repetition_penalty(logits, generated, repetition_penalty)
                
                # Get log probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get top-k candidates
                top_log_probs, top_indices = log_probs.topk(beam_size)
                
                for i in range(beam_size):
                    candidate_seq = beam + [top_indices[i].item()]
                    candidate_score = beam_scores[beam_idx] + top_log_probs[i]
                    all_candidates.append((candidate_seq, candidate_score))
            
            if not all_candidates:
                break
            
            # Sort all candidates by score with length normalization
            all_candidates.sort(key=lambda x: x[1] / (len(x[0]) ** length_penalty), 
                              reverse=True)
            
            # Keep top beam_size
            beams = [c[0] for c in all_candidates[:beam_size]]
            beam_scores = torch.tensor([c[1] for c in all_candidates[:beam_size]], 
                                      device=device)
            
            # Check if all beams finished
            if all(b[-1] == eos_idx for b in beams):
                break
    
    # Add remaining beams to finished
    for beam, score in zip(beams, beam_scores):
        finished_beams.append((beam, score))
    
    # Return best beam (with length normalization)
    if finished_beams:
        best_beam = max(finished_beams, 
                       key=lambda x: x[1] / (len(x[0]) ** length_penalty))
        return best_beam[0][1:]  # Remove SOS
    
    return beams[0][1:]  # Fallback to first beam without SOS
