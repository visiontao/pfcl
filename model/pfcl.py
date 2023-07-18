# Copyright 2023-present, Tao Zhuo
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
import torchvision.transforms.functional as ttf
from utils.data_utils import get_transform
import torchvision.transforms as transforms


def tf_tensor(xs, transforms):
    device = xs.device

    xs = torch.cat([transforms(x).unsqueeze_(0) 
                              for x in xs.cpu()], dim=0)    
    
    return xs.to(device=device)

def kl_div(yn, yo, tau=2):    
    yn = F.log_softmax(yn / tau, dim=-1)
    yo = F.softmax(yo / tau, dim=-1)    
    loss = F.kl_div(yn, yo, reduction='batchmean')
    loss = loss * tau**2

    return loss 

class PFCL:
    def __init__(self, net, args):
        super(PFCL, self).__init__()                
        self.net = net
        self.net_old = None
        self.optim = None        
            
        self.args = args
        
    def end_task(self):          
        self.net_old = deepcopy(self.net)
        self.net_old.eval()        
        
    def observe(self, img, label, aux):      
        self.optim.zero_grad()              

        n = img.size(0)

        if self.net_old is not None:            
            img = torch.cat((img, aux), dim=0)             
        
        img = tf_tensor(img, self.args.transform)     
        
        predict = self.net(img) 
        loss = F.cross_entropy(predict[:n], label)           

        if self.net_old is not None:    
            with torch.no_grad():
                predict_old = self.net_old(img)

            dist = (predict - predict_old).abs().mean(dim=1)
            ind = dist.topk(dim=0, k=n, largest=True)[1]        
            
            if not self.args.stop_kd:
                loss += self.args.alpha * kl_div(predict[ind,:], predict_old[ind,:])            
#                loss += self.args.alpha * F.mse_loss(predict[ind,:], predict_old[ind,:])            

    
        loss.backward()
        self.optim.step()
                        
        return loss      
