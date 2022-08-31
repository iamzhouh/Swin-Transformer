import torch

weights_dict = torch.load("../swin_tiny_patch4_window7_224.pth", map_location = "cpu")["model"]

print(weights_dict.keys())