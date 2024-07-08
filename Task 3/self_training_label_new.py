import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt

import gzip
import pickle

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import albumentations as albu

import segmentation_models_pytorch as smp



def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    label_transform = [albu.Resize(320, 320, interpolation=2)]
    return albu.Compose(label_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
    
ENCODER = "efficientnet-b3"#'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    

    
    
    
    

    
    
    
    







with gzip.open("task3data/train.pkl", 'rb') as f:
    trainX= pickle.load(f)

with gzip.open("task3data/sample.pkl", 'rb') as f:
    samples = pickle.load(f)

# load best saved checkpoint
best_model = torch.load('./100epochsfpn_efficient-netb3.pth')

CLASSES=["valve"]
DEVICE = "cuda"

class Dataset3(BaseDataset):
    """mitral valve Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['valve']
    
    def __init__(
            self, 
            data,
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.data = data
        self.images_fps = [pair for pair in self.data]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        image = self.images_fps[i]    
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=np.ones(image.shape))
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing

        if self.preprocessing:
            sample = self.preprocessing(image=image[:,:,np.newaxis], mask=np.ones(image[:,:,np.newaxis].shape))
            image, mask = sample['image'], sample['mask']

        return image
        
    def __len__(self):
        return len(self.data)

train_data = []
for i in range(len(trainX)):
    video = trainX[i]["video"]
    for j in range(video.shape[2]):
        train_data.append(video[:,:,j])

train_dataset = Dataset3(
    train_data, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)


count =0        
predictions = []
for i in range(65):
    n_frames = trainX[i]["video"].shape[2]
    already = trainX[i]["frames"]
    #prediction = np.array(np.zeros_like(trainX[i]["video"]), dtype=np.bool)
    print(f"GOING INTO {i}th video")
    for j in range(n_frames):
        if j in already:
            continue
        image = train_dataset[count]
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = pr_mask.squeeze().cpu().numpy()
        pr_mask_resized = cv2.resize(pr_mask[:,:,np.newaxis], dsize=trainX[i]["video"].shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
        count+=1
        trainX[i]["label"][:,:,j] = pr_mask_resized.round()
    

with gzip.open('labelled_train.pkl', 'wb') as f:
    pickle.dump(trainX, f, 2)