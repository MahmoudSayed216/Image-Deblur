from torchmetrics import StructuralSimilarityIndexMeasure


def SSIM(im1, im2):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim_val = ssim(im1, im2)