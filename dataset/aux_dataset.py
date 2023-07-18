# # Copyright 2022-present, Tao Zhuo
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import glob
import numpy as np
import pickle
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps

# load images at initialization
def get_files(target_dir, extension='.jpg'):
    
    item_list = os.listdir(target_dir)

    file_list = list()
    for item in item_list:
        item_dir = os.path.join(target_dir,item)
        if os.path.isdir(item_dir):
            file_list += get_files(item_dir)
        else:
            if os.path.splitext(item_dir)[1] == extension:
                file_list.append(item_dir)
    return file_list

class AuxDataset:
    """
    External Dataset to assist model training
    """
    def __init__(self, root='/root/data/dataset', name='caltech256', transform=None):
        self.name = name
        self.images = []
        
        if transform is None:
            transform = transforms.Resize(64)        
        
        files = []
        if self.name == 'caltech256':
            files = get_files(os.path.join(root, 'Caltech256'))
        elif self.name == 'flowers102':
            files = get_files(os.path.join(root, 'Flowers102'))
        elif self.name == 'xpie_s':
            files = get_files(os.path.join(root, 'XPIE', 'Salient'))   
        elif self.name == 'xpie_n':
            files = get_files(os.path.join(root, 'XPIE', 'Non-Salient'))  
        elif self.name == 'xpie':
            files = get_files(os.path.join(root, 'XPIE'))                

        random.shuffle(files)
        
        files = files[:8000]        

        for i in range(len(files)):
            img = Image.open(files[i])            
            if len(img.split()) < 3: 
                img = img.convert('RGB')     
            img = transform(img)
            img = transforms.functional.to_pil_image(img)
            self.images.append(img)      
            
        self.length = len(self.images)
        
    def random_data(self):            
        index = np.random.randint(0, self.length)      
        img = self.images[index]

        return img
    