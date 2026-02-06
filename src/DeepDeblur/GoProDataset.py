from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as F
from PIL import Image
from typing import Literal
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch

class GoProDataset(Dataset):
    def __init__(self, data_path: str, split:str=Literal["train", "test"], crops = False,transforms = None):
        super().__init__()
        self.go_pro_resolution = (1280, 720)
        self.cropped_region_side_length = 256
        self.scaling_factor = 0.5
        self.sharp_base = os.path.join(data_path, split, "sharp")
        self.blur_base = os.path.join(data_path, split, "blur")
        self.mode = split        
        self.transforms = transforms
        self.sharp_dir_content = os.listdir(self.sharp_base)
        self.blur_dir_content = os.listdir(self.blur_base)
        self.crops = crops

    def get_cropped_regions_points(self):
        
        left = random.randint(0, self.go_pro_resolution[0]-self.cropped_region_side_length-1)
        right = left+self.cropped_region_side_length
        upper = random.randint(0, self.go_pro_resolution[1]-self.cropped_region_side_length-1)
        lower = upper + self.cropped_region_side_length

        return left, upper, right, lower

    def tl_br_to_xywh(self, left, upper, right, lower):
        w = (right - left)
        h = (lower - upper)
        x, y = (left, upper)

        return (x, y), w, h


    def __len__(self):
        return len(self.blur_dir_content)

    def __getitem__(self, index):
        sharp_image_path = os.path.join(self.sharp_base, self.sharp_dir_content[index])
        blur_image_path = os.path.join(self.blur_base, self.blur_dir_content[index])

        sharp_image = Image.open(sharp_image_path)
        blur_image = Image.open(blur_image_path)


        fine_scale_sharp = sharp_image
        fine_scale_blur = blur_image

        if self.crops:
            region = self.get_cropped_regions_points()
            fine_scale_sharp = sharp_image.crop(region)
            fine_scale_blur = blur_image.crop(region)


        

        if self.transforms is not None:
            # _256_scale_sharp = self.transforms(_256_scale_sharp)
            # _256_scale_blur = self.transforms(_256_scale_blur)
            fine_scale_blur, fine_scale_sharp = self.transforms(fine_scale_blur,fine_scale_sharp )
        else:
            fine_scale_blur, fine_scale_sharp = F.to_tensor(fine_scale_blur) ,F.to_tensor(fine_scale_sharp)

        medium_scale_blur = F.resize(fine_scale_blur, size=
                                    [
                                        round(self.scaling_factor*self.cropped_region_side_length), 
                                        round(self.scaling_factor*self.cropped_region_side_length), 
                                    ])
        
        coarse_scale_blur  = F.resize(fine_scale_blur, size=
                                    [
                                        round(self.scaling_factor*self.scaling_factor*self.cropped_region_side_length), 
                                        round(self.scaling_factor*self.scaling_factor*self.cropped_region_side_length), 
                                    ])
    

        medium_scale_sharp = F.resize(fine_scale_sharp, size=
                                    [
                                        round(self.scaling_factor*self.cropped_region_side_length), 
                                        round(self.scaling_factor*self.cropped_region_side_length), 
                                    ])
        
        coarse_scale_sharp  = F.resize(fine_scale_sharp, size=
                                    [
                                        round(self.scaling_factor*self.scaling_factor*self.cropped_region_side_length), 
                                        round(self.scaling_factor*self.scaling_factor*self.cropped_region_side_length), 
                                    ])
        #! ARE THE TENSORS BEING SCALED?
        # print()
        return (fine_scale_blur, medium_scale_blur, coarse_scale_blur), (fine_scale_sharp, medium_scale_sharp, coarse_scale_sharp)
