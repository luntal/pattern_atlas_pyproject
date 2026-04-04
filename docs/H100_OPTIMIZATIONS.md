# H100 GPU Training Optimizations

## Summary of Optimizations

This document describes the performance optimizations implemented for NVIDIA H100 GPU training while maintaining full compatibility with MacBook (CPU/MPS) training.

## Implemented Optimizations

### 1. **Automatic Mixed Precision (AMP) Training**
- **What**: Uses FP16 (half precision) for most operations, FP32 for critical parts
- **Benefit**: ~2-3x speedup on H100, reduced memory usage
- **H100 Advantage**: H100 has 1979 TFLOPS FP16 vs 989 TFLOPS FP32
- **Implementation**: `torch.cuda.amp.autocast()` with `GradScaler`
- **Compatibility**: Only enabled on CUDA devices, MacBook uses FP32

### 2. **TF32 Tensor Cores**
- **What**: TensorFloat-32 format for matrix multiplications
- **Benefit**: ~8x speedup for matmul operations vs FP32
- **H100 Advantage**: 3958 TFLOPS TF32 (4x better than A100)
- **Implementation**: `torch.backends.cuda.matmul.allow_tf32 = True`
- **Compatibility**: Only enabled on compute capability 8.0+ (A100/H100)

### 3. **cuDNN Auto-tuning**
- **What**: Automatically selects optimal cuDNN kernels
- **Benefit**: 10-30% speedup after initial warm-up
- **Implementation**: `torch.backends.cudnn.benchmark = True`
- **Compatibility**: CUDA only, no effect on MacBook

### 4. **torch.compile() Optimization**
- **What**: PyTorch 2.0+ graph compilation and kernel fusion
- **Benefit**: 20-50% additional speedup on modern GPUs
- **Implementation**: `torch.compile(model, mode='reduce-overhead')`
- **Compatibility**: Optional, falls back gracefully if unavailable

### 5. **Device-Aware Configuration**
- **What**: Automatic detection and configuration per device
- **Devices Supported**:
  - CUDA (NVIDIA GPUs) - Full optimizations
  - MPS (Apple Silicon) - FP32, optimized kernels
  - CPU - Standard FP32, no special optimizations

## Performance Estimates

### H100 GPU (with all optimizations):
- **Expected speedup vs baseline**: 3-5x
- **Batch size 32, 40K samples, 40 epochs**: ~1.5-2 hours
- **Batch size 64, 40K samples, 30 epochs**: ~1-1.5 hours
- **Memory usage**: ~40-50% less than FP32

### MacBook M2/M3 (MPS):
- **No regression**: Same performance as before
- **Batch size 32**: ~8-10 hours for 40 epochs
- **FP32 precision maintained**: No loss in quality

### MacBook CPU:
- **No regression**: Same performance as before
- **Slower but stable**: ~20-30 hours for 40 epochs

## H100-Specific Features Used

| Feature | H100 Capability | Implementation Status |
|---------|----------------|----------------------|
| FP16 (AMP) | 1979 TFLOPS | ✅ Enabled |
| TF32 | 3958 TFLOPS | ✅ Enabled |
| FP8 | 7958 TFLOPS | ❌ Future work |
| Transformer Engine | Native support | ❌ Future work |
| CUDA Graphs | Supported | ❌ Future work |

## Compatibility Matrix

| Platform | Precision | AMP | TF32 | torch.compile | cuDNN Benchmark |
|----------|-----------|-----|------|---------------|-----------------|
| H100 GPU | FP16/TF32 | ✅ | ✅ | ✅ | ✅ |
| A100 GPU | FP16/TF32 | ✅ | ✅ | ✅ | ✅ |
| V100 GPU | FP16 | ✅ | ❌ | ✅ | ✅ |
| RTX 3090 | FP16 | ✅ | ❌ | ✅ | ✅ |
| M2/M3 MPS | FP32 | ❌ | ❌ | ❌ | ❌ |
| CPU | FP32 | ❌ | ❌ | ❌ | ❌ |

## Code Changes

### train.py
- Added device detection with priority: CUDA > MPS > CPU
- Displays device name and type on startup

### model.py (train_vae function)
- Auto-detects GPU compute capability
- Enables TF32 on capable devices (compute ≥ 8.0)
- Enables AMP with GradScaler for CUDA
- Wraps forward/loss in autocast context
- Uses scaled backward pass for mixed precision
- Falls back to FP32 on non-CUDA devices

## Usage

No code changes needed! The optimizations are **automatically applied** based on detected hardware:

```bash
# On H100 server
python src/train.py
# Output: "Using CUDA device: NVIDIA H100"
# Output: "TF32 enabled for faster training on H100/A100"
# Output: "Automatic Mixed Precision (AMP) enabled"
# Output: "torch.compile enabled for additional optimization"

# On MacBook with Apple Silicon
python src/train.py
# Output: "Using MPS device (Apple Silicon)"
# Output: "MPS optimizations enabled for Apple Silicon"

# On MacBook CPU or older machine
python src/train.py
# Output: "Using CPU device"
# Output: "CPU mode - no hardware-specific optimizations"
```

## Monitoring Performance

To verify optimizations are working:

```python
# Check if AMP is reducing memory
nvidia-smi  # Monitor GPU memory usage

# Check training speed
# Look at "step_time" in training output
# H100 with optimizations: ~0.1-0.3s per step (batch_size=32)
# H100 without optimizations: ~0.5-0.8s per step
# MPS: ~0.8-1.2s per step
```

## Future Optimizations (Not Yet Implemented)

1. **FP8 Training** - H100 supports FP8 (8000 TFLOPS!)
   - Requires TransformerEngine or PyTorch 2.1+ FP8 support
   - Potential 2x additional speedup

2. **CUDA Graphs** - Capture and replay computation graphs
   - Reduces kernel launch overhead
   - ~10-20% additional speedup

3. **Flash Attention** - Optimized attention for transformers
   - Already good with current implementation
   - Could use `torch.nn.functional.scaled_dot_product_attention`

4. **Gradient Accumulation** - For even larger effective batch sizes
   - Simulate batch_size=256+ with smaller batches
   - Better gradient estimates

5. **Distributed Training** - Multi-GPU support
   - Use `torch.nn.parallel.DistributedDataParallel`
   - Linear scaling with number of GPUs

## Troubleshooting

### Issue: "RuntimeError: Expected all tensors to be on the same device"
- **Cause**: Model and data on different devices
- **Fix**: Automatically handled in the code

### Issue: "CUDA out of memory"
- **Cause**: Batch size too large
- **Fix**: Reduce `batch_size` in hparams.py

### Issue: "torch.compile not available"
- **Cause**: PyTorch version < 2.0
- **Fix**: Upgrade PyTorch or ignore (graceful fallback)

### Issue: Slower on MacBook
- **Cause**: AMP not beneficial on MPS/CPU
- **Fix**: None needed, code automatically uses FP32

## Verification

To verify all optimizations are enabled on H100:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU name:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
print("cuDNN version:", torch.backends.cudnn.version())
print("TF32 matmul:", torch.backends.cuda.matmul.allow_tf32)
print("TF32 cuDNN:", torch.backends.cudnn.allow_tf32)
```

Expected output on H100:
```
PyTorch version: 2.x.x
CUDA available: True
CUDA version: 12.x
GPU name: NVIDIA H100 PCIe
Compute capability: (9, 0)
cuDNN version: 8xxx
TF32 matmul: True
TF32 cuDNN: True
```

## Performance Comparison

Estimated training time for 40K samples, 40 epochs:

| Hardware | Batch Size | Optimizations | Time | Speedup |
|----------|-----------|---------------|------|---------|
| H100 | 32 | All | 1.5h | 5.3x |
| H100 | 32 | None (FP32) | 8h | 1x |
| A100 | 32 | All | 2.5h | 3.2x |
| V100 | 32 | AMP only | 5h | 1.6x |
| M2 Max | 32 | MPS | 10h | 0.8x |
| CPU (16-core) | 32 | None | 25h | 0.3x |

These are estimates and will vary based on model size and complexity.
