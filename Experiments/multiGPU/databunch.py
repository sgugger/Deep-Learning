from PIL import Image
from torch.utils.data import Dataset
import pickle, gzip, torch, math, random, numpy as np, torch.nn.functional as F
from pathlib import Path
from IPython.core.debugger import set_trace
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader as DataLoader1
import torchvision
from collections import Iterable
from functools import reduce,partial
from tqdm import tqdm, tqdm_notebook, trange, tnrange
from sampler import DistributedSampler

def find_classes(folder):
    classes = [d for d in folder.iterdir()
               if d.is_dir() and not d.name.startswith('.')]
    classes.sort(key=lambda d: d.name)
    return classes

def get_image_files(c):
    return [o for o in list(c.iterdir())
            if not o.name.startswith('.') and not o.is_dir()]

class FilesDataset1(Dataset):#Renamed to avoid conflict with fastai FilesDataset
    def __init__(self, folder, tfms):
        cls_dirs = find_classes(folder)
        self.fns, self.y = [], []
        self.classes = [cls.name for cls in cls_dirs]
        for i, cls_dir in enumerate(cls_dirs):
            fnames = get_image_files(cls_dir)
            self.fns += fnames
            self.y += [i] * len(fnames)
        self.tfms = tfms
        
    def __len__(self): return len(self.fns)

    def __getitem__(self,i):
        x = Image.open(self.fns[i])
        x = torch.tensor(np.array(x, dtype=np.float32).transpose(2,0,1)).div_(255.)
        if self.tfms is not None: x = self.tfms(x)[0]
        return x,self.y[i]


def get_dataloader(ds, bs, shuffle, device, stats, sampler):
    return DeviceDataLoader(DataLoader1(ds, batch_size=bs, shuffle=shuffle,num_workers=8, sampler=sampler), device, stats)

class DeviceDataLoader():
    def __init__(self, dl, device, stats):
        self.dl,self.device = dl,device
        self.m, self.s = map(lambda x:torch.tensor(x, dtype=torch.float32, device=device), stats)
        
    def __iter__(self):
        for b in self.dl:
            x, y = b[0].to(self.device),b[1].to(self.device)
            x = (x - self.m[None,:,None,None]) / self.s[None,:,None,None]
            yield x,y
    
    def __len__(self): return (len(self.dl))

class DataBunch():
    def __init__(self, trn_ds, val_ds, stats, device, trn_sampler=None, bs=64):
        if hasattr(trn_ds, 'classes'): self.classes = trn_ds.classes
        if trn_sampler is not None:
            self.trn_dl = get_dataloader(trn_ds, bs, shuffle=False, device=device, stats=stats, sampler=trn_sampler)
        else: self.trn_dl = get_dataloader(trn_ds, bs, shuffle=True, device=device, stats=stats, sampler=None)
        self.val_dl = get_dataloader(val_ds, bs*2, shuffle=False, device=device, stats=stats, sampler=None)

    @classmethod
    def from_files(cls, Path, trn_tfms, val_tfms, stats, device, distrib=False, trn_name='train', val_name='valid', bs=64):
        trn_ds, val_ds = FilesDataset1(Path/trn_name, trn_tfms), FilesDataset1(Path/val_name, val_tfms)
        trn_sampler = DistributedSampler(trn_ds) if distrib else None
        return cls(trn_ds, val_ds, stats, device, trn_sampler, bs)

def interpolate(x, coords, padding='reflect'):
    if padding=='reflect':#Reflect padding isn't implemented in grid_sample yet
        coords[coords < -1] = coords[coords < -1].mul_(-1).add_(-2)
        coords[coords > 1] = coords[coords > 1].mul_(-1).add_(2)
        padding='zeros'
    return F.grid_sample(x, coords, padding_mode=padding)

def affine_transform(img, matrix, interpol=True, padding='reflect'):
    """
    Applies an affine transformation to an image.
    
    Optional: only computes the new coordinates without doing the interpolation to create the new images.
    Args:
    x: a batch of images
    matrix: a matrix of size 2 by 3 describing the transformation.
            if the transformation is Ax + b, the matrix is (A|b)
    interpol: if False, returns only the new coordinates
    padding: padding to apply during the interpolation. Supports zeros, border, reflect
    
    """
    coords = F.affine_grid(matrix[None], img[None].size())
    return interpolate(img[None],coords,padding) if interpol else coords

def get_random_rot_matrix(degrees):
    theta = random.uniform(-degrees,degrees) * math.pi / 180
    return torch.tensor([[math.cos(theta), -math.sin(theta), 0],
                         [math.sin(theta), math.cos(theta),  0],
                         [0,               0,                1]])

def get_random_scale_matrix(zoom_range):
    scale = random.uniform(*zoom_range)
    return torch.tensor([[scale, 0, 0],
                         [0, scale, 0],
                         [0,  0,    1]])

def get_random_flip(prob):
    if np.random.rand() < prob:
        return torch.tensor([[-1, 0, 0],
                             [0,  1, 0],
                             [0,  0, 1]]).float()
    else: return torch.eye(3)

class CustomTfm():
    
    def __init__(self, p_flip, pad, size, size_mult):
        self.p_flip,self.pad,self.size,self.size_mult = p_flip,pad,size,size_mult
        
    def __call__(self, x):
        _, h, w = x.size()
        #Resize the image so that the lower dimension is size * size_mult
        ratio = (self.size * self.size_mult) / min(h,w)
        h,w = int(h * ratio), int(w*ratio)
        #Pads
        x = F.pad(x[None], (self.pad,self.pad,self.pad,self.pad), 'reflect') #Symmetric not implemented in F.pad
        #Affine transforms
        matrix = get_random_flip(self.p_flip)
        matrix = matrix[:2,:]
        img_size = torch.Size([1,3,h+2*self.pad,w+2*self.pad])
        coords = F.affine_grid(matrix[None], img_size)
        #Coords transforms then crop
        a = random.randint(0, h+2*self.pad-self.size) if h + 2*self.pad>= self.size else 0
        b = random.randint(0, w+2*self.pad-self.size) if w + 2*self.pad>= self.size else 0
        coords = coords[:,a:a+self.size,b:b+self.size,:]
        #Interpolation
        return interpolate(x, coords)