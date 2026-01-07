from PIL import Image
import random
from torchvision.transforms import functional as F
import torchvision.transforms as T
import torch



def transform_pil_images(b_img, s_img):
    # Spatial transforms
    angle = random.choice([0, 90, 180, 270])
    b_img = b_img.rotate(angle)
    s_img = s_img.rotate(angle)
    if random.random() < 0.5:
        b_img = b_img.transpose(Image.FLIP_LEFT_RIGHT)
        s_img = s_img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        b_img = b_img.transpose(Image.FLIP_TOP_BOTTOM)
        s_img = s_img.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Channel permutation (PIL-only)
    r, g, b = b_img.split()
    channels = [r, g, b]
    random.shuffle(channels)
    b_img = Image.merge("RGB", channels)
    
    # Saturation in PIL using HSV
    b_img = T.ColorJitter(saturation=(0.5, 1.5))(b_img)
    
    return F.to_tensor(b_img), F.to_tensor(s_img)


def add_gaussian_noise(img: torch.Tensor):
    noise_std = max(0, random.gauss(0, (2/255)))
    noise = torch.randn_like(img)*noise_std
    img = img+noise
    img = torch.clamp(img, min=0, max=1)

    return img




def apply_transforms(b_img: Image, s_img: Image):
    b_img , s_img = transform_pil_images(b_img, s_img)
    b_img = add_gaussian_noise(b_img)

    return b_img, s_img

