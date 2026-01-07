from BlurDataset import GoProDataset
from DeepDeblur import DeepDeblur
import torchvision.transforms as T

train_configs = {
    "training": {"per_channel_mean": [0.1,0.2,0.3]}
}

shared_configs = {
    "device": "cpu"
}

train_transforms = T.Compose([    
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
ds = GoProDataset("/home/mahmoud-sayed/Desktop/Datasets/Gopro Dataset", mode="train", transforms=train_transforms)

(x, y, z), (a, b, c) = ds.__getitem__(0)
print("x.shape: ", x.shape)
print("y.shape: ", y.shape)
print("z.shape: ", z.shape)
print("a.shape: ", a.shape)
print("b.shape: ", b.shape)
print("c.shape: ", c.shape)

model = DeepDeblur(train_configs=train_configs, shared_configs=shared_configs)
nparams = sum(p.numel() for p in model.parameters())

print(nparams)

x = x.unsqueeze(0)
y= y.unsqueeze(0)
z = z.unsqueeze(0)

_1, _2, _3 = model([x, y, z])

print(_1.shape)
print(_2.shape)
print(_3.shape)