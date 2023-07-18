import os
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10

class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None, download=False, aux_dataset=None):          
        super(MyCIFAR10, self).__init__(root, train, transform, download)
        self.transform = transform
        self.aux_dataset = aux_dataset
        
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        if self.aux_dataset is not None:                   
            aux_img = self.aux_dataset.random_data()
        else:
            aux_img = img.copy()
            
        if self.transform is not None:
            img = self.transform(img)
            aux_img = self.transform(aux_img)  
        
        return img, target, aux_img