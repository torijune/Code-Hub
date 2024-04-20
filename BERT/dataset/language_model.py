import torch.nn as nn

from .bert import BERT

class BERTLM(nn.Module):
    def __init__(self, bert : BERT, vocab_size):
        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentecePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentece(x), self.mask_lm(x)
    

class  NextSentecePrediction(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftMax(dim = -1)

    def forward(self, x):
        return self.softmax(self.linear(x[:,0]))
    
class MaskedLanguageModel(nn.Module):
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftMax(dim = -1)

    def forward(self, x):
        return self.softmax(self.linear(x))
