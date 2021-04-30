import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import scipy.ndimage as nd

class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label= 0, thresh=0.7, min_kept=100000, factor=8) -> None:
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = thresh
        self.min_kept = int(min_kept)
        self.factor = factor
        self.criterion = torch.nn.CrossEntropyLoss()
        
    
    def find_threshold(self, np_predict, np_target):
        factor = self.factor
        predict = nd.zoom(np_predict,(1.0, 1.0, 1.0/factor, 1.0/factor), order=1)
        target = nd.zoom(np_target, (1.0, 1.0/factor, 1.0/factor), order=0)
        
        n,c,h,w = predict.shape
        
        min_kept = self.min_kept // (factor * factor)
        
        input_label = target.ravel().astype(np.int32)
        input_prob = np.rollaxis(predict,1). reshape((c, -1))
        
        valid_flag = input_label != self.ignore_label
        
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        
        if min_kept >=num_valid:
            threshold = 1.0
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if min_kept > 0:
                k_th = min(len(pred), min_kept) - 1
                new_arr = np.partition(pred, k_th)
                new_th = new_arr[k_th]
                if new_th > self.thresh:
                    threshold = new_th
        
        return threshold
    
    
    def generate_new_target(self, predict, target):
        np_predict = predict.data.cpu().numpy()
        np_target = target.data.cpu().numpy()
        
        n, c, h, w = np_predict.shape
        
        threshold = self.find_threshold(np_predict, np_target)
        
        input_label = np_target.ravel().astype(np.int32)
        input_prob = np.rollaxis(np_predict, -1).reshape((c, -1))
        
        valid_flag = input_label != self.ignore_label
        
        valid_inds = np.where(valid_flag)[0]
        
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        
        if num_valid > 0 :
            prob = input_prob[:,valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            print('Labels: {} {}'.format(len(valid_inds), threshold))
        
        
        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        
        new_target = torch.from_numpy(input_label.reshape(target.size())).long().cuda(target.get_device())
        
        return new_target
    
    def forward(self, predict, target):
        
        input_prob = F.softmax(predict, 1)
        target = self.generate_new_target(input_prob, target)
        
        return self.criterion(predict, target)

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None) -> None:
        super().__init__()
        self.loss = nn.NLLLoss(weight)
        
    def forward(self, outputs, targets):
        outputs = outputs
        return self.loss(F.log_softmax(outputs, dim=1), targets)
            
        
        