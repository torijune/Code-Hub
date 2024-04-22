# model.py
# 베이스라인 코드로 제공되는 CNN과 LSTM 모델 모두 마지막 출력이 logit의 형태이며,
# 이를 sigmoid 함수를 통과시켜 확률값으로 변환하는 과정이 모델에서 이루어지지 않고,
# 손실 함수 중 BCEWithLogitsLoss를 사용하여 이 과정이 손실 함수 내부에서 이루어지도록 설정되어 있습니다.
# (BCEWithLogitsLoss = sigmoid + BCELoss)
# 따라서, 모델을 추가 및 수정할 때에도 마지막 출력값이 logit의 형태가 되도록 설계해야 합니다.

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self):
        """
        Forward pass logic
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)



class CNN(BaseModel):
    ''' Initialize the model '''
    def __init__(self, input_size=15, seq_len=200, output_size=1, 
                 cnn_hidden_list=[16,32], fc_hidden_list=[], dropout_p=0, batch_norm=False):
        '''
        input_size: 입력 데이터의 feature 수
        seq_len: 입력 데이터의 시퀀스 길이
        output_size: 출력 데이터의 차원 (=1)
        cnn_hidden_list: CNN layer의 hidden 차원 리스트
        fc_hidden_list: FC layer의 hidden 차원 리스트 ([]일 경우, 1차원으로 요약하는 layer 하나만 적용)
        dropout_p: dropout 확률 (0~1, 0이면 dropout을 사용하지 않음)
        batch_norm: batch normalization 사용 여부
        '''
        super(CNN, self).__init__()

        def conv(inp, oup, kernel, stride, pad, batch_norm, dropout_p):
            return nn.Sequential(
                nn.Conv1d(inp, oup, kernel, stride, pad, bias=True),
                nn.BatchNorm1d(oup) if batch_norm else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p) if dropout_p else nn.Identity()
            )

        self.cnn_layers = nn.Sequential()
        cnn_output_len = seq_len
        cnn_hidden_list = [input_size] + cnn_hidden_list
        for i, (inp, oup) in enumerate(zip(cnn_hidden_list[:-1], cnn_hidden_list[1:])):
            self.cnn_layers.add_module(f'conv{i}', conv(inp, oup, kernel=3, stride=1, pad=1, 
                                                       batch_norm=batch_norm, dropout_p=dropout_p))
            cnn_output_len = (cnn_output_len + 2 - 3) // 1 + 1  # (n + 2p - k) / s + 1
            self.cnn_layers.add_module(f'pool{i}', nn.MaxPool1d(kernel_size=2, stride=2))
            cnn_output_len = (cnn_output_len - 2) // 2 + 1  # (n - k) / s + 1

        self.fc_layers = nn.Sequential()
        fc_hidden_list = [cnn_hidden_list[-1] * cnn_output_len] + fc_hidden_list + [output_size]
        for i, (inp, oup) in enumerate(zip(fc_hidden_list[:-1], fc_hidden_list[1:])):
            self.fc_layers.add_module(f'fc{i}', nn.Linear(inp, oup))
            if i < len(fc_hidden_list) - 2:
                self.fc_layers.add_module(f'fc{i}_relu', nn.ReLU())
                if dropout_p:
                    self.fc_layers.add_module(f'fc{i}_dropout', nn.Dropout(dropout_p))

        self.init_weights()

    
    def forward(self, x):
        # input x: (batch_size, input_size, seq_len)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x  # (batch_size, output_size)

    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)


class LSTM(BaseModel):
    ''' Initialize the model '''
    def __init__(self, input_size=15, seq_len=200, output_size=1, 
                 lstm_hidden_dim=64, lstm_n_layer=2, bidirectional=True,
                 fc_hidden_list=[], dropout_p=0, batch_norm=False):
        '''
        input_size: 입력 데이터의 feature 수
        seq_len: 입력 데이터의 시퀀스 길이
        output_size: 출력 데이터의 차원 (=1)
        lstm_hidden_dim: LSTM layer의 hidden 차원
        lstm_n_layer: LSTM layer의 layer 수
        fc_hidden_list: FC layer의 hidden 차원 리스트 ([]일 경우, 1차원으로 요약하는 layer 하나만 적용)
        dropout_p: dropout 확률 (0~1, 0이면 dropout을 사용하지 않음)
        batch_norm: batch normalization 사용 여부
        '''
        super(LSTM, self).__init__()

        self.lstm_layers = nn.LSTM(input_size=input_size, 
                                   hidden_size=lstm_hidden_dim, 
                                   num_layers=lstm_n_layer, 
                                   batch_first=True, 
                                   bidirectional=bidirectional, 
                                   dropout=dropout_p)
        
        self.fc_layers = nn.Sequential()
        fc_hidden_list = [lstm_hidden_dim * (2 if bidirectional else 1)] + fc_hidden_list + [output_size]
        for i, (inp, oup) in enumerate(zip(fc_hidden_list[:-1], fc_hidden_list[1:])):
            self.fc_layers.add_module(f'fc{i}', nn.Linear(inp, oup))
            if i < len(fc_hidden_list) - 2:
                self.fc_layers.add_module(f'fc{i}_relu', nn.ReLU())
                if dropout_p:
                    self.fc_layers.add_module(f'fc{i}_dropout', nn.Dropout(dropout_p))

        self.init_weights()


    def forward(self, x):
        # input x: (batch_size, input_size, seq_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, input_size)
        x, _ = self.lstm_layers(x)  # (batch_size, seq_len, lstm_hidden_dim)
        x = x[:, -1, :]  # (batch_size, lstm_hidden_dim)
        x = self.fc_layers(x)

        return x  # (batch_size, output_size)
    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)


class GRU(BaseModel):
    def __init__(self, input_size, seq_len, hidden_size, num_layers=1, output_size=1, dropout_p=0.0):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # x: (batch_size, seq_len, input_size)
        x, _ = self.gru(x)  # x: (batch_size, seq_len, hidden_size)
        x = x[:, -1, :]  # x: (batch_size, hidden_size) - 마지막 time step의 output만 사용
        x = self.fc(x)
        return x


class CRNN(nn.Module):
    def __init__(self, input_size, hidden_size, pool_size =2, dropout_p = 0.2, num_layers = 1):
        super(CRNN, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size,num_layers=num_layers, batch_first=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # 이거 해줘야 돼 원준아
        x, _ = self.gru(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        energy = torch.tanh(self.attention(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        attention = torch.bmm(v, energy).squeeze(1)
        return F.softmax(attention, dim=1)

class CR2SeqWithAttention(BaseModel):
    def __init__(self, input_size, seq_len, hidden_size, output_size=1, gru_decoder_layers=3):
        super(CR2SeqWithAttention, self).__init__()
        self.encoder = CRNN(input_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.gru_decoder = nn.GRU(hidden_size, hidden_size, num_layers=gru_decoder_layers, batch_first=True)
        self.decoder_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 인코더 슥삭
        encoder_outputs = self.encoder(x)  # [batch size, seq_len, hidden_size]

        # 어텐션 슥삭
        hidden = encoder_outputs[:, -1, :]  # 마지막 hidden state를 사용
        attention_weights = self.attention(hidden, encoder_outputs)
        attention_applied = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        # 디코더 야미
        # 디코더에 어텐션 인풋으로 슥삭
        decoder_output, _ = self.gru_decoder(attention_applied.unsqueeze(1))


        x = self.decoder_fc(decoder_output.squeeze(1))
        return x




if __name__ == '__main__':
    import torch
    
    batch_size = 7
    input_size = 6
    seq_len = 20

    # model = CNN(input_size=input_size,
    #             seq_len=seq_len,
    #             cnn_hidden_list=[4,8],
    #             fc_hidden_list=[128, 64],
    #             dropout_p=0.2, 
    #             batch_norm=True)

    model = LSTM(input_size=input_size,
                 seq_len=seq_len,
                 lstm_hidden_dim=16,
                 lstm_n_layer=2,
                 bidirectional=True,
                 fc_hidden_list=[128, 64],
                 dropout_p=0.2, 
                 batch_norm=True)
    
    print(model)

    # (batch_size, input_size, seq_len) -> (batch_size, 1) 가 잘 되는지 확인
    x = torch.randn(batch_size, input_size, seq_len)
    y = model(x)
    print(f'\nx: {x.shape} => y: {y.shape}')