import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from utils.loss import OhemCrossEntropy2d, CrossEntropyLoss2d
import scipy.ndimage as nd


class CriterionOhemDSN(nn.Module):
    def __init__(self, ignore_index = 0, threshold = 0.7, min_kept = 100000) -> None:
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2d(ignore_index, threshold, min_kept)
        self.criterion2 = nn.CrossEntropyLoss(ignore_index=ignore_index,reduction='mean')
        
    def forward(self, preds, target):
        loss1 = self.criterion1(preds, target)
        loss2 = self.criterion2(preds, target)
        
        return loss1 + loss2 *0.4
        