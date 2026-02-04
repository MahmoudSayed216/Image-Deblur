# from PIL import Image
# import random
# from torchvision.transforms import functional as F
# import torchvision.transforms as T
# import torch



# def transform_pil_images(b_img, s_img):
#     # Spatial transforms
#     angle = random.choice([0, 90, 180, 270])
#     b_img = b_img.rotate(angle)
#     s_img = s_img.rotate(angle)
#     if random.random() < 0.5:
#         b_img = b_img.transpose(Image.FLIP_LEFT_RIGHT)
#         s_img = s_img.transpose(Image.FLIP_LEFT_RIGHT)
#     if random.random() < 0.5:
#         b_img = b_img.transpose(Image.FLIP_TOP_BOTTOM)
#         s_img = s_img.transpose(Image.FLIP_TOP_BOTTOM)
    
#     # Channel permutation (PIL-only)
#     r, g, b = b_img.split()
#     channels = [r, g, b]
#     random.shuffle(channels)
#     b_img = Image.merge("RGB", channels)
    
#     # Saturation in PIL using HSV
#     b_img = T.ColorJitter(saturation=(0.5, 1.5))(b_img)
    
#     return F.to_tensor(b_img), F.to_tensor(s_img)


# def add_gaussian_noise(img: torch.Tensor):
#     noise_std = max(0, random.gauss(0, (2/255)))
#     noise = torch.randn_like(img)*noise_std
#     img = img+noise
#     img = torch.clamp(img, min=0, max=1)

#     return img




# def apply_transforms(b_img: Image, s_img: Image):
#     b_img , s_img = transform_pil_images(b_img, s_img)
#     b_img = add_gaussian_noise(b_img)

#     return b_img, s_img

import torch
import random
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np

def augment_patch(input_img: Image, target_img: Image):
    """
    input_img: PIL.Image (blurry)
    target_img: PIL.Image (sharp)
    returns: tuple of torch tensors (input, target)
    """

    # Convert to tensor first for fast processing
    input_patch = F.to_tensor(input_img)
    target_patch = F.to_tensor(target_img)

    # --- target_input: replace input with target sometimes ---
    if random.randint(1, 10) == 1:
        input_patch = target_patch.clone()

    # --- flip horizontal ---
    if random.randint(0, 1) == 1:
        input_patch = torch.flip(input_patch, dims=[2])
        target_patch = torch.flip(target_patch, dims=[2])

    # --- rotate 0,90,180,270 ---
    rotate = random.randint(0, 3)
    if rotate > 0:
        input_patch = torch.rot90(input_patch, k=rotate, dims=[1, 2])
        target_patch = torch.rot90(target_patch, k=rotate, dims=[1, 2])

    # --- shuffle color channels ---
    perm = torch.randperm(3)
    input_patch = input_patch[perm]
    target_patch = target_patch[perm]

    # --- change saturation ---
    if random.randint(1, 10) == 1:
        amp_factor = 1 + random.uniform(-0.5, 0.5)

        # Convert RGB -> HSV
        def rgb_to_hsv(img):
            img = img.permute(1, 2, 0).numpy()  # H,W,C
            hsv = np.zeros_like(img)
            maxc = img.max(axis=2)
            minc = img.min(axis=2)
            v = maxc
            s = (maxc - minc) / (maxc + 1e-8)
            # Avoid hue computation for simplicity; we only need to scale S
            hsv[:, :, 1] = s
            hsv[:, :, 2] = v
            return hsv, minc, maxc, img

        def hsv_to_rgb(hsv, minc, maxc, original_rgb):
            # Approximate conversion: scale only S channel
            s = hsv[:, :, 1] * amp_factor
            s = np.clip(s, 0, 1)
            # Simple approximate scaling: rescale colors
            rgb = original_rgb * (s[:, :, None] / (hsv[:, :, 1][:, :, None] + 1e-8))
            rgb = np.clip(rgb, 0, 1)
            return torch.from_numpy(rgb.transpose(2,0,1)).float()

        # Input
        hsv, minc, maxc, orig_rgb = rgb_to_hsv(input_patch)
        input_patch = hsv_to_rgb(hsv, minc, maxc, orig_rgb)
        # Target
        hsv, minc, maxc, orig_rgb = rgb_to_hsv(target_patch)
        target_patch = hsv_to_rgb(hsv, minc, maxc, orig_rgb)

    # --- add noise to input only ---
    sigma_sigma = 2/255
    sigma = random.gauss(0, sigma_sigma)
    noise = torch.randn_like(input_patch) * sigma
    input_patch = input_patch + noise

    # --- clamp both images ---
    input_patch = torch.clamp(input_patch, 0, 1)
    target_patch = torch.clamp(target_patch, 0, 1)

    return input_patch, target_patch
