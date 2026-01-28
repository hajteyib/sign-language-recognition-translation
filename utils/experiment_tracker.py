import os
import json
from datetime import datetime
from pathlib import Path
import shutil

# Optional matplotlib import (for plotting)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plots will not be generated.")

class ExperimentTracker:
    """
    Manages experiment tracking with organized directory structure.
    Each experiment gets its own folder with all artifacts.
    """
    
    def __init__(self, base_dir='checkpoints', experiment_name=None):
        """
        Args:
            base_dir: Base directory for all experiments
            experiment_name: Name of experiment (auto-generated if None)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create experiment directory
        if experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"exp_{timestamp}"
        
        self.exp_name = experiment_name
        self.exp_dir = self.base_dir / experiment_name
        self.exp_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.models_dir = self.exp_dir / 'models'
        self.plots_dir = self.exp_dir / 'plots'
        self.logs_dir = self.exp_dir / 'logs'
        
        self.models_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Tracking data
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.config = {}
        self.best_metrics = {
            'epoch': 0,
            'val_loss': float('inf'),
            'train_loss': float('inf')
        }
        
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"Directory: {self.exp_dir}")
        print(f"{'='*60}\n")
    
    def save_config(self, config_dict):
        """Save experiment configuration."""
        self.config = config_dict
        config_path = self.exp_dir / 'config.json'
        
        # Convert non-serializable objects
        config_serializable = {}
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_serializable[key] = str(value)
            else:
                config_serializable[key] = value
        
        with open(config_path, 'w') as f:
            json.dump(config_serializable, f, indent=2)
        
        print(f"✓ Config saved to {config_path}")
    
    def log_epoch(self, epoch, train_loss, val_loss, lr=None):
        """Log metrics for an epoch."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if lr is not None:
            self.learning_rates.append(lr)
        
        # Update best metrics
        if val_loss < self.best_metrics['val_loss']:
            self.best_metrics['epoch'] = epoch
            self.best_metrics['val_loss'] = val_loss
            self.best_metrics['train_loss'] = train_loss
        
        # Generate plots after each epoch
        try:
            self.plot_losses()
            if len(self.learning_rates) > 0:
                self.plot_learning_rate()
        except Exception as e:
            print(f"  ⚠ Warning: Could not generate plots ({e})")
    
    def save_checkpoint(self, checkpoint_dict, epoch, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            checkpoint_dict: Full checkpoint dictionary
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        import torch
        
        # Save ONLY best model (not latest)
        if is_best:
            best_path = self.models_dir / 'best_model.pt'
            torch.save(checkpoint_dict, best_path)
            print(f"  ✓ Best model saved (epoch {epoch}, val_loss={checkpoint_dict.get('val_loss', 0):.4f})")
    
    def plot_losses(self):
        """Generate and save loss curves."""
        if not MATPLOTLIB_AVAILABLE:
            print("  ⚠ Skipping plots (matplotlib not available)")
            return
        
        if len(self.train_losses) == 0:
            return
        
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax.axvline(x=self.best_metrics['epoch'], color='g', linestyle='--', 
                   label=f"Best (epoch {self.best_metrics['epoch']})")
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Training History - {self.exp_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save figure
        plot_path = self.plots_dir / 'loss_curve.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Loss curve saved to {plot_path}")
    
    def plot_learning_rate(self):
        """Generate and save learning rate curve."""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if len(self.learning_rates) == 0:
            return
        
        steps = range(1, len(self.learning_rates) + 1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, self.learning_rates, 'g-', linewidth=2)
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title(f'Learning Rate Schedule - {self.exp_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plot_path = self.plots_dir / 'learning_rate.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Learning rate curve saved to {plot_path}")
    
    def save_metrics_log(self):
        """Save detailed metrics log."""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_metrics': self.best_metrics
        }
        
        log_path = self.logs_dir / 'metrics.json'
        with open(log_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Metrics log saved to {log_path}")
    
    def save_summary(self, total_epochs, total_time=None):
        """Save experiment summary."""
        summary = {
            'experiment_name': self.exp_name,
            'total_epochs': total_epochs,
            'total_time_seconds': total_time,
            'best_epoch': self.best_metrics['epoch'],
            'best_val_loss': self.best_metrics['val_loss'],
            'best_train_loss': self.best_metrics['train_loss'],
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'config': self.config
        }
        
        summary_path = self.exp_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Experiment Summary")
        print(f"{'='*60}")
        print(f"Name: {self.exp_name}")
        print(f"Total Epochs: {total_epochs}")
        if total_time:
            print(f"Total Time: {total_time/3600:.2f} hours")
        print(f"\nBest Results (Epoch {self.best_metrics['epoch']}):")
        print(f"  Train Loss: {self.best_metrics['train_loss']:.4f}")
        print(f"  Val Loss:   {self.best_metrics['val_loss']:.4f}")
        print(f"\nFinal Results (Epoch {total_epochs}):")
        print(f"  Train Loss: {self.train_losses[-1]:.4f}")
        print(f"  Val Loss:   {self.val_losses[-1]:.4f}")
        print(f"\nAll results saved to: {self.exp_dir}")
        print(f"{'='*60}\n")
    
    def finalize(self, total_epochs, total_time=None):
        """
        Finalize experiment: save all plots, logs, and summary.
        """
        print("\n" + "="*60)
        print("Finalizing Experiment")
        print("="*60 + "\n")
        
        self.plot_losses()
        self.plot_learning_rate()
        self.save_metrics_log()
        self.save_summary(total_epochs, total_time)
        
        # Create README
        self._create_readme()
    
    def _create_readme(self):
        """Create README file for the experiment."""
        readme_content = f"""# Experiment: {self.exp_name}

## Overview
This directory contains all artifacts from the training experiment.

## Directory Structure
```
{self.exp_name}/
├── config.json          # Experiment configuration
├── summary.json         # Final results summary
├── models/              # Model checkpoints
│   ├── best_model.pt    # Best model (lowest val loss)
│   ├── latest.pt        # Latest checkpoint (for resuming)
│   └── checkpoint_epoch_*.pt  # Per-epoch checkpoints
├── plots/               # Visualization plots
│   ├── loss_curve.png   # Training/validation loss curves
│   └── learning_rate.png # Learning rate schedule
└── logs/                # Detailed logs
    └── metrics.json     # Per-epoch metrics

```

## Best Results
- **Best Epoch:** {self.best_metrics['epoch']}
- **Best Val Loss:** {self.best_metrics['val_loss']:.4f}
- **Best Train Loss:** {self.best_metrics['train_loss']:.4f}

## Configuration
See `config.json` for full hyperparameters.

## How to Use

### Resume Training
```bash
python scripts/train.py --resume {self.exp_dir}/models/latest.pt
```

### Evaluate
```bash
python scripts/evaluate.py --checkpoint {self.exp_dir}/models/best_model.pt --split dev
```

### Inference
```bash
python scripts/demo.py --checkpoint {self.exp_dir}/models/best_model.pt --landmarks path/to/file.npy
```
"""
        
        readme_path = self.exp_dir / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"✓ README created at {readme_path}")


def list_experiments(base_dir='checkpoints'):
    """List all experiments with their summaries."""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"No experiments found in {base_dir}")
        return
    
    experiments = sorted(base_path.iterdir(), key=lambda x: x.name, reverse=True)
    
    print("\n" + "="*80)
    print("Available Experiments")
    print("="*80)
    
    for exp_dir in experiments:
        if not exp_dir.is_dir():
            continue
        
        summary_path = exp_dir / 'summary.json'
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            
            print(f"\n{exp_dir.name}")
            print(f"  Best Val Loss: {summary.get('best_val_loss', 'N/A'):.4f} (epoch {summary.get('best_epoch', 'N/A')})")
            print(f"  Total Epochs: {summary.get('total_epochs', 'N/A')}")
            print(f"  Config: d_model={summary['config'].get('d_model', 'N/A')}, "
                  f"batch_size={summary['config'].get('batch_size', 'N/A')}")
        else:
            print(f"\n{exp_dir.name} (incomplete)")
    
    print("\n" + "="*80 + "\n")
