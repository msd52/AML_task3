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



#EXTRACT BOXES:

with gzip.open("192epochsfpn_se_resnext50_32x4dfullds.pkl", 'rb') as f:
    preds = pickle.load(f)    
    
for i in range(20):
    shapecur = preds[i]["prediction"].shape
    n_frames = preds[i]["prediction"].shape[2]
    newind = int(shapecur[1]*0.4)
    for j in range(n_frames):
        print(f"NEW FRAME {j}")
        preds[i]["prediction"][:,:newind,j] = 0.0
        frame = preds[i]["prediction"][:,:,j] 
        if frame.any():
            horizs = np.sort(np.nonzero(frame)[1])
            verts = np.sort(np.nonzero(frame)[0])
#             print(verts)
#             print(horizs)
            idxh = np.argmax(np.diff(horizs))
            idxv = np.argmax(np.diff(verts))
            hthres = horizs[idxh]
            vthres = verts[idxv]
            print(vthres)
            print(hthres)
            if np.max(np.diff(horizs))>60:
                print(f"not skipped h for {j}")
                if hthres<0.5*shapecur[1]:
                    hthres =  horizs[idxh+1]
                    preds[i]["prediction"][:,:hthres,j] = 0.0
                else:
                    hthres = horizs[idxh]
                    preds[i]["prediction"][:,hthres:,j] = 0.0
            if np.max(np.diff(verts))>60:
                print(f"not skipped v for {j}")
                if vthres<0.5*shapecur[0]:
                    vthres = verts[idxv+1]
                    preds[i]["prediction"][:vthres,:,j] = 0.0
                else:
                    vthres = verts[idxv]
                    preds[i]["prediction"][vthres:,:,j] = 0.0


BOXES = []
for i in range(20):
    maxleft = 1000
    maxright = 0
    maxup = 1000
    maxdown = 0
    nframes = preds[i]["prediction"].shape[2]
    vid = preds[i]["prediction"]
    print(nframes)
    print(preds[i]["prediction"].shape)
    for j in range(nframes):
        if vid[:,:,j].any() and np.sum(vid[:,:,j])>400 :
            horizontal = np.nonzero(vid[:,:,j])[1]
            vertical = np.nonzero(vid[:,:,j])[0]
            left,right,up,down = np.amin(horizontal),np.amax(horizontal),np.amin(vertical),np.amax(vertical)
            if left<maxleft:
                maxleft = left
            if right>maxright:
                maxright = right
            if up<maxup:
                maxup = up
            if down>maxdown:
                maxdown = down
#         else:
#              print("skipped")
    print(f" the maxes are {maxleft}, {maxright}, {maxup}, {maxdown} ")
    BOXES.append((maxleft,maxright,maxup,maxdown))
    
print(f"{len(BOXES)} is length of boxes")
for i in range(20):
    print(f"{i} video, gives {BOXES[i][0]} and {BOXES[i][1]} and {BOXES[i][2]} and {BOXES[i][3]}")
    

    
    




with gzip.open("task3data/test.pkl", 'rb') as f:
    testX= pickle.load(f)



test_data = []
BOXES2 = []
for i in range(len(testX)):
    video = testX[i]["video"]
    maxleft,maxright,maxup,maxdown = BOXES[i][0],BOXES[i][1],BOXES[i][2],BOXES[i][3]
    h = maxright-maxleft
    v = maxdown-maxup
    if 1.3*v>h:
        maxright += (1.3*v-h)/2
        maxleft -= (1.3*v-h)/2
    if v>0.35*vid[:,:,0].shape[0]:
        #maxright -= 0.1*h
        maxleft += 0.1*h
        maxdown -= 0.15*v
        print("jhereee")
    for j in range(video.shape[2]):
        print(f"maxright is {maxright}")
        print(f"maxleft is {maxleft}")
        print(f"maxup is {maxup}")
        print(f"maxdown is {maxdown}")
        test_data.append(video[:,:,j][int(maxup):int(maxdown),int(maxleft):int(maxright)])
    BOXES2.append((int(maxleft),int(maxright),int(maxup),int(maxdown)))
    
    
############################################### AS USUAL###################################
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
    maxleft,maxright,maxup,maxdown = BOXES2[i]
    
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
        print(f"EL SIZE IS {(pr_mask_resized>0.0).sum()}")
        print(f"EL SIZEEEEEEE IS {(pr_mask_resized[maxup:maxdown,maxleft:maxright]>0.0).sum()}")

        if np.sum(pr_mask_resized)<2000:
            ind = np.unravel_index(np.argsort(pr_mask_resized, axis=None), pr_mask_resized.shape)
            prediction[:,:,j][ind[0][-2000:],ind[1][-2000:]] = 1.0
            prediction[:,:,j][:,:maxleft] = 0.0
            prediction[:,:,j][:,maxright:] = 0.0
            prediction[:,:,j][:maxup,:] = 0.0
            prediction[:,:,j][maxdown:,:] = 0.0

        else:
            prediction[:,:,j] = np.where(pr_mask_resized > 0.08, 1, 0)#pr_mask_resized.round()
        count+=1
        
    predictions.append({
        'name': testX[i]['name'],
        'prediction': prediction
        }
    )
    
    
with gzip.open('boxed_top1200_thres=008_192epochsfpn_se_resnext50_32x4dfullds.pkl', 'wb') as f:
    pickle.dump(predictions, f, 2)