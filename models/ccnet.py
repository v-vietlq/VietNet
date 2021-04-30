import torch
import torch.nn as nn
import torchvision

def INF(B,H,W):
    return -torch.diag(torch.tensor(float('inf')).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

class CrissCrossAttention(nn.Module):
    def __init__(self, in_ch) -> None:
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels= in_ch, out_channels= in_ch//8, kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels=in_ch, out_channels= in_ch//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels= in_ch, out_channels= in_ch//8, kernel_size= 1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.INF = INF
        
    
    
    def forward(self, x):
        bs,_, height, width = x.size()
        
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(bs*width,-1, height).permute(0,2,1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(bs*height,-1, width).permute(0,2,1)
        
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(bs*width, -1, height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(bs*height,-1, width)
        
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(bs*width, -1, height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(bs*height,-1, width)
        
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(bs, height, width)).view(bs, width, height, height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(bs, height, width, width)
        
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(bs*width, height, height)
        