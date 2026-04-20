"""Inspect existing best_model.pt to determine its format."""
import torch

path = r"experiments\element_transformer_test_v4\best_model.pt"
ckpt = torch.load(path, map_location='cpu', weights_only=False)
print(type(ckpt))
if isinstance(ckpt, dict):
    print("Keys:", list(ckpt.keys()))
    if 'config' in ckpt:
        print("config:", ckpt['config'])
    if 'elements' in ckpt:
        print("elements:", ckpt['elements'])
    if 'wl_min' in ckpt:
        print("wl_min:", ckpt['wl_min'], "wl_max:", ckpt['wl_max'])
else:
    print("Not a dict — raw state_dict")
    k = list(ckpt.keys())[:3]
    print("First 3 keys:", k)
