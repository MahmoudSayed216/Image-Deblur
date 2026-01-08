import torch
from torchmetrics import StructuralSimilarityIndexMeasure



class SSIM:
    def __init__(self, device):
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def calculate(self, im1: torch.Tensor, im2: torch.Tensor):
        ssim_val = self.ssim(im1, im2)
        return ssim_val
