from torch import nn
from blocks.encoder_block import EncoderLayer
from embedding.transformer_embedding import TransformerEmbedding

class Encoder():
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super(Encoder).__init__()
        self.emb = TransformerEmbedding(d_model = d_model,
                                        max_len = max_len,
                                        drop_prob = drop_prob,
                                        device = device)
        self.layers = nn.Module([EncoderLayer(d_model = d_model,
                                              ffn_hidden = ffn_hidden,
                                              n_head = n_head,
                                              drop_prob = drop_prob)
                                for _ in range(n_layers)])
        
    def forawrd(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
