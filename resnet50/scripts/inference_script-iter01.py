#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import PIL
from PIL import Image

import cv2

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import random

import albumentations
from albumentations.pytorch.transforms import ToTensorV2


# In[2]:


def get_instance_objectdetection_model(num_classes,path_weight):
    # load an instance segmentation model pre-trained on COCO
    create_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=False)

    # get the number of input features for the classifier
    in_features = create_model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    create_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    create_model.load_state_dict(torch.load(path_weight,map_location=torch.device('cpu')))

    return create_model


# In[3]:


#path_weight = "/kaggle/input/global_wheat_detection/fasterrcnn_resnet50_fpn_best.pth"
path_weight = "/kaggle/input/gwd-augmentation-training-weights-ver01/customtrained_fasterrcnn_resnet50_fpn_augementation (1).pth"


# In[4]:


num_classes = 2
# Why 2 classes - background and wheat-heads
trained_model = get_instance_objectdetection_model(num_classes,path_weight)


# In[5]:


def get_test_transform():
    return albumentations.Compose([
        # A.Resize(512, 512),
        ToTensorV2(p=1.0)
    ])


# In[6]:


class GlobalWheatDetectionTestDataset(torch.utils.data.Dataset):
    # first lets start with __init__ and initialize any objects
    def __init__(self,input_df,input_dir,transforms=None):
        
        self.df=input_df
        
        self.list_images = list(self.df['image_id'].unique())
        
        self.image_dir=input_dir
        
        self.transforms = transforms
    
    # next lets define __getitem__
    # very important to note what it returns for EACH image:
    # I. image - a PIL image of size (H,W) for ResNet50 FPN image should be scaled
    
    # II. image_id 
    
    def __getitem__(self,idx):
        
        # II. image_id
        img_id = self.list_images[idx]
        # I. Input image
        # Specifications: A.RGB format B. scaled (0,1) C. size (H,W) D. PIL format
        
        img = cv2.imread(self.image_dir+"/"+img_id+".jpg")
        img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_scaled = img_RGB/255.0
        img_final = img_scaled
        ret_image = {}
        ret_image['image']=img_final
        if self.transforms is not None:
            res = self.transforms(**ret_image)
            img_final = res['image']
            #img_final = torch.tensor(img_final, dtype=torch.float32)
        
        return img_final, img_id
    
    # next lets define __len__    
    def __len__(self):
        
        return len(self.df['image_id'].unique())


# In[7]:


def collate_fn(batch):
    return tuple(zip(*batch))


# In[8]:


df_test=pd.read_csv("/kaggle/input/global-wheat-detection/sample_submission.csv")


# In[9]:


test_dir="/kaggle/input/global-wheat-detection/test/"


# In[10]:


df_test.head()


# In[11]:


test_dataset = GlobalWheatDetectionTestDataset(df_test,test_dir,get_test_transform())


# In[12]:


test_dataloader = DataLoader(test_dataset, batch_size=8,shuffle=False, num_workers=1,collate_fn=collate_fn)


# In[13]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


# In[14]:


detection_threshold = 0.45


# In[15]:


def format_prediction_string(boxes, scores): ## Define the formate for storing prediction results
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


# In[16]:


images, img_ids = next(iter(test_dataloader))


# In[17]:


## Lets make the prediction
results=[]
trained_model.to(device)
trained_model.eval()
images = []
outputs =[]
for images_, image_ids in test_dataloader:    

    images = list(image.to(device,dtype=torch.float) for image in images_)
    outputs = trained_model(images)

    for i, image in enumerate(images):

        boxes = outputs[i]['boxes'].data.cpu().numpy()    ##Formate of the output's box is [Xmin,Ymin,Xmax,Ymax]
        scores = outputs[i]['scores'].data.cpu().numpy()
        
        boxes = boxes[scores >= detection_threshold].astype(np.int32) #Compare the score of output with the threshold and
        scores = scores[scores >= detection_threshold]                    #slelect only those boxes whose score is greater
                                                                          # than threshold value
        image_id = image_ids[i]
        
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]         
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]         #Convert the box formate to [Xmin,Ymin,W,H]
        
        
            
        result = {                                     #Store the image id and boxes and scores in result dict.
            'image_id': image_id,
            'PredictionString': format_prediction_string(boxes, scores)
        }

        
        results.append(result)              #Append the result dict to Results list

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.head()


# In[18]:


test_df.to_csv('submission.csv', index=False)

