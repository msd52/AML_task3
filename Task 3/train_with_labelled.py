import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import gzip
import numpy as np
import os
import albumentations as albu

print("start")
def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
		
# load data
trainX = load_zipped_pickle("labelled_train.pkl")

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

trainX_extr = []
n_frames=0
for i in range(len(trainX)):
    video = trainX[i]["video"]
    labels = trainX[i]["label"]
    frames = trainX[i]["frames"]

    for j in range(trainX[i]["video"].shape[2]//10):
        frame = video[:,:,10*j]
        label = labels[:,:,10*j]
        trainX_extr.append((frame,label))
        n_frames+=1
    for j in frames:
        frame = video[:,:,j]
        label = labels[:,:,j]
        trainX_extr.append((frame,label))
        n_frames+=1
	
class Dataset2(BaseDataset):
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
        self.images_fps = [pair[0] for pair in self.data]
        self.masks_fps = [pair[1] for pair in self.data]
        
        # convert str names to class values on masks
        self.class_values = [1.0]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#        mask = cv2.imread(self.masks_fps[i], 0)
        
        image = self.images_fps[i][:,:,np.newaxis]
        mask = self.masks_fps[i]

        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        print(np.array(image).shape)
        print(np.array(mask).shape)
        # apply augmentations
        if self.augmentation:
            print("doing augmentations")
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            print("doing preprocessing")
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        print(np.array(image).shape)
        print(np.array(mask).shape) 

        return image, mask
        
    def __len__(self):
        return len(self.data)
		
def get_training_augmentation():
    train_transform = [

        albu.ShiftScaleRotate(scale_limit = [-0.06,0.06],rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.Resize(320, 320, interpolation=2),
        #albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        #albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
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
	
	
	
	
	
import torch
import numpy as np
import segmentation_models_pytorch as smp



print("before critical point")
ENCODER = "se_resnext50_32x4d"#'efficientnet-b3'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['valve']
ACTIVATION = 'sigmoid' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder
# model = smp.FPN(
    # encoder_name=ENCODER, 
    # encoder_weights=ENCODER_WEIGHTS, 
    # classes=len(CLASSES), 
    # activation=ACTIVATION,
    # in_channels = 3
# )
model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
    in_channels = 3
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    

train_dataset = Dataset2(
     trainX_extr[:(95*n_frames//100)], 
    classes=['valve'],
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
)


valid_dataset = Dataset2(
     trainX_extr[(95*n_frames//100):], 
    classes=['valve'],
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
)


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)#12
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)#4


# Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
# IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0001),
])



# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

print("way after critical point")
#model.load_state_dict(torch.load("100epochsfpn.pth"))


# train model for 40 epochs

max_score = 0
best=0
for i in range(0, 100):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')
        best = i
    print(f"CURRENT BEST IS FROM ITERATION {best}")
    print(f"CURRENT MAX SCORE IS {max_score}")
    print(f"CURRENT VAL SCORE IS {valid_logs['iou_score']}")



    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')