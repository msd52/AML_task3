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
    test_transform = [albu.Resize(320, 320, interpolation=2)]
    return albu.Compose(test_transform)


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
    
ENCODER = 'se_resnext50_32x4d'#"efficientnet-b3"
ENCODER_WEIGHTS = 'imagenet'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    
    
    
    
    
    
    
    
    
    
    







with gzip.open("task3data/test.pkl", 'rb') as f:
    testX= pickle.load(f)

with gzip.open("task3data/sample.pkl", 'rb') as f:
    samples = pickle.load(f)

# load best saved checkpoint
best_model = torch.load('./192epochsfpn_se_resnext50_32x4dfullds.pth')

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

test_data = []
for i in range(len(testX)):
    video = testX[i]["video"]
    for j in range(video.shape[2]):
        test_data.append(video[:,:,j])

test_dataset = Dataset3(
    test_data, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)


count =0        
predictions = []
for i in range(20):
    n_frames = testX[i]["video"].shape[2]
    prediction = np.array(np.zeros_like(testX[i]["video"]), dtype=np.bool)
    print(f"GOING INTO {i}th video")
    for j in range(n_frames):
        
        image = test_dataset[count]
       
        #plt.imshow(image[0,:,:])
        #plt.show()
        #print(image[0,int(image.shape[1]/2),:])
        #print(image.dtype)
        #print(np.amax(image))

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = pr_mask.squeeze().cpu().numpy()
        pr_mask_resized = cv2.resize(pr_mask[:,:,np.newaxis], dsize=testX[i]["video"].shape[:2][::-1], interpolation=2)#cv2.INTER_LANCZOS4)
        # if np.sum(pr_mask_resized)<600:
            # ind = np.unravel_index(np.argsort(pr_mask_resized, axis=None), pr_mask_resized.shape)
            # prediction[:,:,j][ind[0][-600:],ind[1][-600:]] = 1.0
        # else:
            # prediction[:,:,j] = np.where(pr_mask_resized > 0.08, 1, 0)#pr_mask_resized.round()
        
        prediction[:,:,j] = pr_mask_resized.round()

        count+=1
        
    predictions.append({
        'name': testX[i]['name'],
        'prediction': prediction
        }
    )
    
    
with gzip.open('top600_thres=008_192epochsfpn_se_resnext50_32x4dfullds.pkl', 'wb') as f:
    pickle.dump(predictions, f, 2)