import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from modules.branch_paths import branch_dir, safe_branch_name, get_current_branch

# Select training log by index: 0=latest, 1=previous, 2=two before, ...
log_index = 0
if len(sys.argv) > 1:
    try:
        log_index = int(sys.argv[1])
    except ValueError:
        print(f"Error: log index must be an integer, got '{sys.argv[1]}'")
        raise SystemExit(1)

if log_index < 0:
    print(f"Error: log index must be >= 0, got {log_index}")
    raise SystemExit(1)

branch_name = safe_branch_name(get_current_branch())
log_dir = branch_dir('docs', create=False)
if not log_dir.exists():
    print(f"Warning: No training log files found in docs/{branch_name}/")
    raise SystemExit(0)

logs = sorted(log_dir.glob('training_log_*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
if not logs:
    print(f"Warning: No training log files found in docs/{branch_name}/")
    raise SystemExit(0)

if log_index >= len(logs):
    print(f"Error: requested log index {log_index}, but only {len(logs)} log file(s) found in docs/{branch_name}/")
    raise SystemExit(1)

selected_log = logs[log_index]
print(f'Loading training log (index {log_index}): {selected_log}')

# Load the CSV file
df = pd.read_csv(str(selected_log))
print(f'Total epochs: {len(df)}')
print(df.head())

# Create a figure with multiple subplots
fig, axes = plt.subplots(4, 2, figsize=(15, 13))
fig.suptitle(f'Training Metrics - {os.path.basename(selected_log)}', fontsize=16)

# Plot 1: Total Loss (Train vs Val)
ax = axes[0, 0]
ax.plot(df['epoch'], df['total_loss'], label='Train Loss', linewidth=2)
if 'val_total_loss' in df.columns and df['val_total_loss'].notna().any():
    ax.plot(df['epoch'], df['val_total_loss'], label='Val Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Total Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Reconstruction Loss (Train vs Val)
ax = axes[0, 1]
ax.plot(df['epoch'], df['reconstruction_loss'], label='Train Rec Loss', linewidth=2)
if 'val_reconstruction_loss' in df.columns and df['val_reconstruction_loss'].notna().any():
    ax.plot(df['epoch'], df['val_reconstruction_loss'], label='Val Rec Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Reconstruction Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: KL Loss (Train vs Val)
ax = axes[1, 0]
ax.plot(df['epoch'], df['kl_loss'], label='Train KL Loss', linewidth=2)
if 'val_kl_loss' in df.columns and df['val_kl_loss'].notna().any():
    ax.plot(df['epoch'], df['val_kl_loss'], label='Val KL Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('KL Divergence Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Perplexity (Train vs Val)
ax = axes[1, 1]
ax.plot(df['epoch'], df['perplexity'], label='Train Perplexity', linewidth=2)
if 'val_perplexity' in df.columns and df['val_perplexity'].notna().any():
    ax.plot(df['epoch'], df['val_perplexity'], label='Val Perplexity', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Perplexity')
ax.set_title('Model Perplexity')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Learning Rate and Beta
ax = axes[2, 0]
ax1 = ax
ax2 = ax1.twinx()
ax1.plot(df['epoch'], df['learning_rate'], label='Learning Rate', color='blue', linewidth=2)
ax2.plot(df['epoch'], df['beta'], label='Beta', color='orange', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Learning Rate', color='blue')
ax2.set_ylabel('Beta', color='orange')
ax1.set_title('Learning Rate and Beta')
ax1.tick_params(axis='y', labelcolor='blue')
ax2.tick_params(axis='y', labelcolor='orange')
ax1.grid(True, alpha=0.3)

# Plot 6: Annealing Factor
ax = axes[2, 1]
ax.plot(df['epoch'], df['annealing_factor'], label='Annealing Factor', linewidth=2, color='green')
ax.set_xlabel('Epoch')
ax.set_ylabel('Annealing Factor')
ax.set_title('KL Annealing Factor')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 7: Cross entropy per compound field (step, pitch, velocity, deviation)
ax = axes[3, 0]
field_cols = [c for c in df.columns if c.startswith('ce_')]
for col in field_cols:
    line, = ax.plot(df['epoch'], df[col], label=f'Train {col[3:]}', linewidth=2)
    val_col = 'val_' + col
    if val_col in df.columns and df[val_col].notna().any():
        ax.plot(df['epoch'], df[val_col], label=f'Val {col[3:]}', linewidth=1.5, linestyle='--', color=line.get_color())
ax.set_xlabel('Epoch')
ax.set_ylabel('Cross entropy')
ax.set_title('Reconstruction error per field' if field_cols else 'Per field error (not logged)')
if field_cols:
    ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 8: Frames (or tokens of older logs) per sequence
ax = axes[3, 1]
len_col = 'avg_frames_per_sequence' if 'avg_frames_per_sequence' in df.columns else 'avg_tokens_per_sequence'
if len_col in df.columns:
    ax.plot(df['epoch'], df[len_col], linewidth=2, color='purple')
ax.set_xlabel('Epoch')
ax.set_ylabel(len_col.replace('_', ' '))
ax.set_title('Sequence length')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)
print(f"Final Train Loss: {df['total_loss'].iloc[-1]:.4f}")
if 'val_total_loss' in df.columns and df['val_total_loss'].notna().any():
    print(f"Final Val Loss: {df['val_total_loss'].iloc[-1]:.4f}")
print(f"Best Train Loss: {df['total_loss'].min():.4f} (Epoch {df.loc[df['total_loss'].idxmin(), 'epoch']:.0f})")
if 'val_total_loss' in df.columns and df['val_total_loss'].notna().any():
    print(f"Best Val Loss: {df['val_total_loss'].min():.4f} (Epoch {df.loc[df['val_total_loss'].idxmin(), 'epoch']:.0f})")
print(f"Final Perplexity: {df['perplexity'].iloc[-1]:.2f}")
print(f"Total Training Time: {df['elapsed_time_seconds'].iloc[-1]/60:.2f} minutes")
print("="*50)
