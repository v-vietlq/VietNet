import torch
from utils.utils import iou, AverageMeter, pixel_accuracy, dice_coefficient
from tqdm import tqdm


def train_one_epoch(model, criterion, optimizer, data_loader, device):
    
    accuracy = AverageMeter('Acc@1', ':6.2f')
    avgloss = AverageMeter('Loss', '1.5f')
    avgIoU = AverageMeter('IoU@1', ':6.2f')
    Dice_coeff = AverageMeter('Dice@1', '6.2f')
    
    
    model.train()
    for data in tqdm(data_loader):
        
      model.to(device) 
      
      image , target = data['image'].to(device, dtype=torch.float), data['label'].to(device)
      
      output = model(image)
      
      loss = criterion(output, target)
      
      optimizer.zero_grad()
      
      loss.backward()
      
      optimizer.step()
      
      
      miou = iou(output, target)
      
      acc = pixel_accuracy(output, target)
      
      dice_coeff = dice_coefficient(output, target)
      
      accuracy.update(acc, image.size(0))
      avgloss.update(loss, image.size(0))
      avgIoU.update(miou, image.size(0)) 
      Dice_coeff.update(dice_coeff, image.size(0))
            
    return avgloss.avg, avgIoU.avg, accuracy.avg, Dice_coeff.avg


def validate_model(model, criterion, valid_loader, device):

  accuracy = AverageMeter('Acc@1', ':6.2f')
  avgloss = AverageMeter('Loss', '1.5f')
  avgIoU = AverageMeter('Jacc_sim@1', ':6.2f')
  Dice_coeff = AverageMeter('Dice@1', '6.2f')

  model.eval()
  with torch.no_grad():
    for data in valid_loader:
      image , target = data['image'].to(device, dtype=torch.float), data['label'].to(device)
      output = model(image)
    
      loss = criterion(output, target)
     
      miou = iou(output, target)
      
      acc = pixel_accuracy(output, target)
      
      dice_coeff = dice_coefficient(output, target)
      
      accuracy.update(acc, image.size(0))
      avgloss.update(loss, image.size(0))
      avgIoU.update(miou, image.size(0)) 
      Dice_coeff.update(dice_coeff, image.size(0))                           
              
  return avgloss.avg , accuracy.avg, avgIoU.avg, Dice_coeff.avg