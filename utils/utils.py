from numpy.lib.function_base import average
import torch
import numpy as np

from sklearn.metrics import jaccard_score as jsc




class AverageMeter(object):
    def __init__(self, name, fmt=':f') -> None:
        super(AverageMeter, self).__init__()
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self) -> str:
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)   


def get_score(output, target):
    intersection = np.logical_and(target, output)
    union = np.logical_or(target, output)
    iou_score = np.sum(intersection)/ np.sum(union)
    y_true = target.reshape(-1)
    y_pred = output.reshape(-1)
    jacc_sim = jsc(y_true, y_pred, average=None)
    mean_jacc_sim = np.mean(jacc_sim)
    return iou_score, mean_jacc_sim

def iou(output, target):
    output = output
    # output = output.max(dim=1)[1]
    # output = output.float().unsqueeze(1)
    output = torch.argmax(output, 1)
    output = output.cpu().numpy()
    return get_score(output, target.cpu().numpy())


    