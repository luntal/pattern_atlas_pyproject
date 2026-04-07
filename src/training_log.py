import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Find the latest training log in docs directory
log_files = glob.glob('docs/training_log_*.csv')
if not log_files:
    raise FileNotFoundError("No training log files found in ../docs/ directory")

latest_log = max(log_files, key=os.path.getmtime)
print(f'Loading training log: {latest_log}')

# Load the CSV file
df = pd.read_csv(latest_log)
print(f'Total epochs: {len(df)}')
print(df.head())

# Create a figure with multiple subplots
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle(f'Training Metrics - {os.path.basename(latest_log)}', fontsize=16)

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
