import torch
import torchvision

print(f"Czy PyTorch widzi GPU (CUDA)? : {torch.cuda.is_available()}")
print(f"Torch CUDA version: {torch.version.cuda}")
torchvision.ops.nms(torch.rand(5, 4), torch.rand(5), 0.5)