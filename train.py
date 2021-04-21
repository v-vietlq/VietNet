import torch
import torch.nn as nn
import torchvision
import os
import torch.nn.functional as F
from PIL import Image
from utils.utils import iou, AverageMeter
from tqdm import tqdm


def train_one_epoch(model, criterion, optimizer, data_loader, device):
    
    top1 = AverageMeter('Acc@1', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')
    jacc1 = AverageMeter('Jacc_sim@1', ':6.2f')
    train_loss = 0.0
    model.train()
    for data in tqdm(data_loader):
        
      model.to(device) 
      
      image , target = data['image'].to(device, dtype=torch.float), data['label'].to(device)
      
      output = model(image)
      
      loss = criterion(output, target)
      
      optimizer.zero_grad()
      
      loss.backward()
      
      optimizer.step()
      
      train_loss += loss.cpu().detach().numpy()    
      
      acc1, jacc = iou(output, target)
      
      top1.update(acc1, image.size(0))
      avgloss.update(loss, image.size(0))
      jacc1.update(jacc, image.size(0)) 
            
    return avgloss.avg, top1.avg, jacc1.avg


def validate_model(model, criterion, valid_loader, device):

  top1 = AverageMeter('Acc@1', ':6.2f')
  jacc1 = AverageMeter('Jacc_sim@1', ':6.2f')
  avgloss = AverageMeter('Loss', '1.5f')
  val_loss = 0.0

  model.eval()
  with torch.no_grad():
    for data in valid_loader:
      image , target = data['image'].to(device, dtype=torch.float), data['label'].to(device)
      output = model(image)
    
      loss = criterion(output, target)
    
    
      val_loss += loss.cpu().detach().numpy()    
    
      acc1, jacc = iou(output, target)
      
      top1.update(acc1, image.size(0))
      avgloss.update(loss, image.size(0)) 
      jacc1.update(jacc, image.size(0))                             
              
  return avgloss.avg , top1.avg, jacc1.avg