import torch

ckpt = torch.load("output/best.pth", map_location="cpu")

preds = ckpt["preds"]   # this is your list of tensors



print(preds[0].shape, preds[0].min(), preds[0].max())
print(preds[1].shape, preds[1].min(), preds[1].max())
