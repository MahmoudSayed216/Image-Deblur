import torch
import torch.nn as nn

#! MISSING: PIXELSHUFFLE FOR UPSAMPLING
#! MISSING: MEAN SHIFT

class UpConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, padding='same')
        self.shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, padding='same', kernel_size=5)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, padding='same', kernel_size=5)
        

    def forward(self, x):
        _x = self.conv1(x)
        _x = self.activation(_x)
        _x = self.conv2(_x)

        return _x+x

class Network(nn.Module):
    #! change the name of the finer network param
    def __init__(self,  training_configs, finer_network = True, upscale = False):
        super().__init__()
        self.finer_network = finer_network
        self.upscale = upscale
        self.conv1 = nn.Conv2d(6 if finer_network else 3, out_channels=training_configs["model"]["n_channels"], padding='same', kernel_size=5)
        self.ResBlocks = nn.ModuleList([ResBlock(64) for i in range(19)])
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding='same')
        # self.upconv = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=2, stride=2)
        self.upconv = UpConv()
    def forward(self, blur_tensor, lower_level_tensor= None):
        
        if lower_level_tensor is not None:
            blur_tensor=torch.cat([blur_tensor, lower_level_tensor], dim=1)
        
        blur_tensor = self.conv1(blur_tensor)
        
        for block in self.ResBlocks:
            blur_tensor = block(blur_tensor)

        blur_tensor = self.conv2(blur_tensor)
        
        upscaled_tensor = None

        if self.upscale:
            upscaled_tensor = self.upconv(blur_tensor)

        return blur_tensor, upscaled_tensor
        



class DeepDeblur(nn.Module):
    def __init__(self, train_configs, shared_configs):
        super().__init__()
        #! PROBABLY NEEDS TO BE REVIEWED, THEY ARE SIMPLY SUBTRACTING 0.5 FROM EACH CHANNEL, NOT A PER CHANNEL AVERAGE
        per_channel_mean = train_configs["training"]["per_channel_mean"]
        device = shared_configs["device"]

        self.per_channel_mean = torch.tensor(per_channel_mean).view(1, 3, 1, 1).to(device=device)
        self.coarse_level_network = Network(training_configs=train_configs, finer_network=False, upscale=True)
        self.intermediate_level_network = Network(training_configs=train_configs,finer_network=True, upscale=True)
        self.fine_level_network = Network(training_configs=train_configs,finer_network=True, upscale=False)


    def forward(self, blur_tensors):
        blur_tensors[-1] = blur_tensors[-1] - self.per_channel_mean
        blur_tensors[-2] = blur_tensors[-2] - self.per_channel_mean
        blur_tensors[-3] = blur_tensors[-3] - self.per_channel_mean


        b3, u3 = self.coarse_level_network(blur_tensors[-1])
        b2, u2 = self.intermediate_level_network(blur_tensors[-2], u3)
        b1, _ = self.fine_level_network(blur_tensors[-3], u2)
        
        b3 = b3 + self.per_channel_mean
        b2 = b2 + self.per_channel_mean
        b1 = b1 + self.per_channel_mean

        return b1, b2, b3