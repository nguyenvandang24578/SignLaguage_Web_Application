# Trong môi trường training
import torch
ckpt = torch.load('stagcn_tiny_supcon_50cls_best.pt', map_location='cpu')
state_dict = ckpt.get('state_dict', ckpt)  # tùy format
total = sum(v.numel() for v in state_dict.values() if torch.is_floating_point(v))
print(f"Params: {total:,}")
print(f"Size MB: {total * 4 / 1024**2:.2f}")