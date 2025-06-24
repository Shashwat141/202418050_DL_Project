'''import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")'''

'''import torch
print(torch.version.cuda)'''

import torch
print("CUDA available:", torch.cuda.is_available())         # Should be True
print("CUDA device count:", torch.cuda.device_count())      # Should be ≥1
print("Device name:", torch.cuda.get_device_name(0))        # Should list your GPU
