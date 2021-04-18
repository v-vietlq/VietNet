import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import transforms

import cv2

def preprocess(img, height = 1024, width = 512):
    img = transforms.Resize((height, width),interpolation=Image.NEAREST)(img)
    return img

def segmentation(img,origin_height, origin_width, model, device):
    img = T.ToTensor()(img).unsqueeze(dim=0).to(device) 
    
    with torch.no_grad():
        output = model(img)
        

    output = torch.nn.Upsample(size=((origin_height, origin_width)),
                               mode='bilinear',
                               align_corners=True
                               )(output)
    
    output = output.squeeze(0)
    output = output.cpu().detach().numpy()
    
    seg_map = np.argmax(output, axis=0)
       
    return seg_map


def visualize(seg_map, img):
    # Class 0-background: black; Class 1-road: Green; Class 2-person: Red; Class 3-Vehicle: Blue
    COLOR_CODE = [[0,0,0],[0,1,0],[0,0,1],[0,0,255]]
    
    segmap_rgb = np.zeros(img.shape)
    
    for k in np.unique(seg_map):
        segmap_rgb[seg_map == k] = COLOR_CODE[k]
    segmap_rgb = (segmap_rgb * 255).astype('uint8')
    
    overlaid_img = cv2.addWeighted(img,1,segmap_rgb,0.5, 0)
    
    return overlaid_img
    
    
        