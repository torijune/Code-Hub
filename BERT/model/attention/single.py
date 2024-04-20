import torch
import torch.nn.functional as F
import torch.nn as nn
import math


# Scaled Dot Product Attention

class Attention(nn.Module) :
    
    def forward(self, query, key, value, mask = None, dropout = None) :
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        
        if mask is not None :
            scores = scores.masked_fill(mask == 0, -1e9)
            
        p_attn = F.softmax(scores, dim = 1)
        
        if dropout is not None :
            p_attn = dropout(p_attn)
            
        return torch.matmul(p_attn, value), p_attn