import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from utils.loss import OhemCrossEntropy2d
import scipy.ndimage as nd


class CriterionOhemDSN(nn.Module):
    def __init__(self, ignore_index = 0, threshold = 0.7, min_kept = 100000) -> None:
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2d(ignore_index, threshold, min_kept)
        self.criterion2 = nn.CrossEntropyLoss(ignore_index, reduce=True)
        
    def forward(self, preds, target):
        h,w = target.size(1), target.size(2)
        scale_pred = F.interpolate(input= preds[0], size=(h, w), mode='bilinear', align_corners=True)
        
        loss1 = self.criterion1(scale_pred, target)
        loss2 = self.criterion2(scale_pred, target)
        
        return loss1 + loss2 *0.4
        