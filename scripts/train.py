import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import SignLanguageTransformer
from utils.dataset import PhoenixDataset, collate_fn
from utils.experiment_tracker import ExperimentTracker
from utils.decoder import compute_ngram_repetition_loss

def validate(model, val_loader, criterion, device, vocab_size):
    """Validation function"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for src_dict, tgt, src_lengths in val_loader:
            # Move to device
            for k in src_dict:
                src_dict[k] = src_dict[k].to(device)
                B, T, N, C = src_dict[k].shape
                src_dict[k] = src_dict[k].view(B, T, -1)
            
            tgt = tgt.to(device)
            src_lengths = src_lengths.to(device)
            
            # Target input/output
            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]
            
            # Masks
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(0)).to(device)
            src_key_padding_mask = model.create_src_padding_mask(src_dict, src_lengths)
            tgt_key_padding_mask = (tgt_input == 0).transpose(0, 1)
            
            # Forward pass
            output = model(src_dict, tgt_input, 
                          src_key_padding_mask=src_key_padding_mask,
                          tgt_mask=tgt_mask, 
                          tgt_key_padding_mask=tgt_key_padding_mask)
            
            loss = criterion(output.view(-1, vocab_size), tgt_output.reshape(-1))
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / max(num_batches, 1)


def train(args):
    # Start time tracking
    start_time = time.time()
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(
        base_dir=args.checkpoint_dir,
        experiment_name=args.exp_name
    )
    
    # Device configuration
    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # Hyperparameters
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs
    WARMUP_STEPS = args.warmup_steps
    
    # Paths
    LANDMARKS_DIR = args.landmarks_dir
    VOCAB_PATH = args.vocab_path
    NORM_STATS_PATH = args.norm_stats_path
    
    # Dataset
    print("Loading training dataset...")
    train_dataset = PhoenixDataset(
        LANDMARKS_DIR, 
        args.train_annotation_file, 
        VOCAB_PATH,
        split='train',
        normalizer_path=NORM_STATS_PATH,
        augment=True  # Enable augmentation for training
    )
    
    print("Loading validation dataset...")
    val_dataset = PhoenixDataset(
        LANDMARKS_DIR,
        args.dev_annotation_file,
        VOCAB_PATH,
        split='dev',
        normalizer_path=NORM_STATS_PATH,
        augment=False  # No augmentation for validation
    )
    
    vocab_size = len(train_dataset.vocab)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
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
    
    print(f"\nModel configuration:")
    print(f"  d_model: {args.d_model}")
    print(f"  nhead: {args.nhead}")
    print(f"  encoder layers: {args.num_encoder_layers}")
    print(f"  decoder layers: {args.num_decoder_layers}")
    print(f"  dropout: {args.dropout}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Save experiment configuration
    config = {
        'experiment_name': args.exp_name,
        'device': str(device),
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_encoder_layers': args.num_encoder_layers,
        'num_decoder_layers': args.num_decoder_layers,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'warmup_steps': args.warmup_steps,
        'grad_clip': args.grad_clip,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'vocab_size': vocab_size
    }
    tracker.save_config(config)
    
    # Optimization
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing)  # 0 is <pad>
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    # Improved Scheduler: Warmup + Cosine Annealing
    total_steps = EPOCHS * len(train_loader)
    
    # Warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=WARMUP_STEPS
    )
    
    # Cosine annealing scheduler
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - WARMUP_STEPS,
        eta_min=LEARNING_RATE * 0.01
    )
    
    # Sequential scheduler
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_STEPS]
    )
    
    print(f"\nTraining configuration:")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Warmup steps: {WARMUP_STEPS}")
    print(f"  Total steps: {total_steps}")
    print(f"  Gradient clip: {args.grad_clip}")
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    global_step = 0  # Track total training steps for LR logging
    
    model.train()
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        num_batches = 0
        epoch_start = time.time()
        
        for i, (src_dict, tgt, src_lengths) in enumerate(train_loader):
            # Move to device
            for k in src_dict:
                src_dict[k] = src_dict[k].to(device)
                # Flatten features: (Batch, Time, N, C) -> (Batch, Time, N*C)
                B, T, N, C = src_dict[k].shape
                src_dict[k] = src_dict[k].view(B, T, -1)
            
            tgt = tgt.to(device)  # (Time, Batch)
            src_lengths = src_lengths.to(device)
            
            # Target input (exclude last token) and output (exclude first token)
            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :]
            
            # Masks
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(0)).to(device)
            src_key_padding_mask = model.create_src_padding_mask(src_dict, src_lengths)  # ✅ Fixed!
            tgt_key_padding_mask = (tgt_input == 0).transpose(0, 1)  # (Batch, TgtTime)
            
            optimizer.zero_grad()
            
            output = model(src_dict, tgt_input, 
                          src_key_padding_mask=src_key_padding_mask,  # ✅ Now provided!
                          tgt_mask=tgt_mask, 
                          tgt_key_padding_mask=tgt_key_padding_mask)
            
            # Output: (Time, Batch, Vocab)
            # Target: (Time, Batch)
            ce_loss = criterion(output.view(-1, vocab_size), tgt_output.reshape(-1))
            
            # Add repetition penalty to discourage repetitive patterns
            rep_penalty = compute_ngram_repetition_loss(output, n=3, penalty_weight=args.repetition_penalty)
            
            loss = ce_loss + rep_penalty
            loss.backward()
            
            # Gradient clipping ✅
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            
            optimizer.step()
            scheduler.step()
            global_step += 1
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if i % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        epoch_time = time.time() - epoch_start
        
        # Compute average training loss
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation ✅
        print(f"\nRunning validation...")
        val_loss = validate(model, val_loader, criterion, device, vocab_size)
        val_losses.append(val_loss)
        
        # Log to tracker
        current_lr = optimizer.param_groups[0]['lr']
        tracker.log_epoch(epoch + 1, avg_train_loss, val_loss, current_lr)
        
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Epoch Time: {epoch_time:.2f}s")
        print(f"  LR: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'vocab_size': vocab_size,
            'config': config
        }
        
        # Determine if best
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= args.patience:
            print(f"\n⚠ Early stopping triggered after {patience_counter} epochs without improvement")
            print(f"Best val loss: {best_val_loss:.4f} at epoch {epoch + 1 - patience_counter}")
            break
        
        # Save to experiment tracker
        tracker.save_checkpoint(checkpoint, epoch + 1, is_best=is_best)
        
        print("-" * 60)
    
    # Training complete
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)
    
    # Finalize experiment
    tracker.finalize(EPOCHS, total_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Sign Language Translation Model')
    
    # Data paths
    parser.add_argument('--landmarks_dir', type=str, default='data/processed/landmarks')
    parser.add_argument('--train_annotation_file', type=str, 
                       default='data/raw/phoenix14t.pami0.train.annotations_only.gzip')
    parser.add_argument('--dev_annotation_file', type=str,
                       default='data/raw/phoenix14t.pami0.dev.annotations_only.gzip')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl')
    parser.add_argument('--norm_stats_path', type=str, default='data/norm_stats.pkl')
    
    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (increased for better generalization)')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--label_smoothing', type=float, default=0.15, help='Label smoothing factor')
    parser.add_argument('--repetition_penalty', type=float, default=0.1, help='N-gram repetition penalty weight')
    
    # Experiment tracking
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Base directory for experiments')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    
    args = parser.parse_args()
    
    # Verify required files exist
    if not os.path.exists(args.vocab_path):
        print(f"ERROR: Vocabulary file not found: {args.vocab_path}")
        print("Please run: python scripts/build_vocab.py")
        sys.exit(1)
    
    if not os.path.exists(args.norm_stats_path):
        print(f"WARNING: Normalization stats not found: {args.norm_stats_path}")
        print("Training will proceed without normalization.")
        print("For better results, run: python scripts/compute_norm_stats.py")
        args.norm_stats_path = None
    
    train(args)
