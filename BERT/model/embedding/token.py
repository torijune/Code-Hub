import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size = 512):
        super().__init__(vocab_size, embed_size, padding_idx = 0)