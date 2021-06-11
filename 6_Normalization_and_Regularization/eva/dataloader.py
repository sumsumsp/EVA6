import torch
from torchvision import datasets, transforms

def getMnistDataLoader(batch_size=128, shuffle=True, **kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        #transforms.RandomRotation((-6.9, 6.9), fill=(1,)),                                        
                        #transforms.RandomAffine(degees=10, shear=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                        #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader