import torch

# Check if MPS is available
print(torch.backends.mps.is_available())

# Check MPS version
print(torch.__version__)

# Check default dtype
print(torch.get_default_dtype())

# Try forcing the same precision on both
# torch.set_default_dtype(torch.float32)  # Use float32 for both
