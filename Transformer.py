import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout):
        
        super(Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 입력 임베딩과 출력 임베딩
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        # embedding을 torch 내부 함수인 Embedding을 활용하여 정의 (dimension(input, hidden)은 파라미터로써 정의)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        # 일반적인 encoding에 위치 정보를 넣기 위해 positional encoding을 진행 
        # -> 이미 embedding이 된 hidden에 position을 추가해주기 때문에 input_dim은 고려x, hidden_dim을 고려하여 파라미터로 정의
        
        # 인코더와 디코더
        self.encoder_layers = nn.ModuleList([EncoderLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        
        # 최종 선형 레이어
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        
        # 가중치 초기화
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, src, trg):
        src_mask, trg_mask = self.create_masks(src, trg)
        
        # 입력 임베딩과 위치 인코딩
        # src : 소스(번역 작업 등에서 기존의 문장을 의미), trg : 타겟(번역 작업등에서 번역되어야하는 목표 문장)
        src = self.embedding(src)
        src = self.pos_encoding(src)
        trg = self.embedding(trg)
        trg = self.pos_encoding(trg)
        
        # 인코더
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        
        # 디코더
        for layer in self.decoder_layers:
            trg = layer(trg, src, trg_mask, src_mask)
        
        output = self.fc_out(trg)
        return output
    
    def create_masks(self, src, trg):
        # encoder, decoder에서 attention을 진행하기 위한 masking 작업 실시
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # 패딩 마스크
        trg_mask = (trg != 0).unsqueeze(1).unsqueeze(2)  # 패딩 마스크

        # unsqueeze(1)를 통해 마스크 텐서를 batch_size × 1 × seq_length의 크기로 변경
        # unsqueeze(2)를 통해 다시 마스크 텐서의 차원을 확장 시킴 -> 마스크를 batch_size × 1 × 1 × seq_length의 크기로 변경
        # -> 이를 통해 마스크가 모든 위치에 적용될 수 있도록 차원이 변경됨 (원래 입력 텐서와 같은 크기로 바뀜)
        
        trg_len = trg.size(1) # Decoder의 다음 Seq인 trg의 길이 = Decoder가 처리해야할 단어의 수
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()  # 다음 단어 가리기 마스크
        # tril() 함수를 사용해서 trg_len x trg_len 크기의 하삼각행렬을 생성
        trg_mask = trg_mask & trg_sub_mask
        # 기존의 trg_mask와 하삼각 행렬인 trg_sub_mask를 합쳐서 현재 위치 이후의 단어들을 보지 못하도록 마스킹 진행
        # -> Decoder 부분의 mulit-head attention 부분에서 현재 위치 이후의 데이터를 먼저 학습하는 leakage를 막기 위해 진행해야하는 마스킹
        
        return src_mask, trg_mask

class PositionalEncoding(nn.Module):
    # pos_encoding을 정의하기 위해 PositionalEncoding 클래스 생성
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) # 여기서 dropout을 사용하는 건 일반적이지 않을 수 있음
        # but,왜 사용? -> 모델 정규화를 진행하여 모델의 과적합을 막기 위해서
        
        pe = torch.zeros(max_len, d_model)
        # max_len, d_model에 맞는 0 행렬을 생성 -> 초기화를 하기 위해

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 각 위치의 순서에 맞는 열 벡터(unsqueeze(1)를 사용해서 열로 변환)를 생성 (ex, 0,1,2,3,4,...)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # d_model 크기의 차원을 생성 각 위치에는 div_term 값이 있음 -> div_term은 PositionalEncoding을 생성하기 위한 계수
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 홀수 차원에는 sin 값, 짝수 차원에는 cos 값을 할당

        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe 차원을 마지막으로 변환하여 (1, max_len, d_model)의 형태로 만듦

        self.register_buffer('pe', pe)
        # pe를 Buffer 타입으로 저장 -> Buffer 타입으로 저장 시 모델을 저장하거나 불러올 때, 새롭게 pe를 연산하지 않고 저장된 pe를 사용할 수 있음
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    # Transformer의 근간 1 Encoder를 정의하는 부분
    # Encoder, Decoder 부분이 은근 단순한게 각각들에 들어갈 attention, forward, layer_norm 부분만 self.xx 으로 정의 후 src, trg만 업데이트하면 됨
    def __init__(self, hidden_dim, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        # forward 정의 방법 : src를 계속해서 새롭게 업데이트 하면서 위치 이동 시 이동 된 src를 사용하도록 함 (return src를 통해)
        src = self.norm(src + self.dropout(self.self_attn(src, src, src, src_mask)))
        src = self.norm(src + self.dropout(self.feed_forward(src)))
        return src

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(hidden_dim, num_heads, dropout)
        self.src_attn = MultiheadAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, src, trg_mask, src_mask):
        # Decoder의 forward 부분은 Encoder의 forward와 다른점이 있음
        # attention이 2가지로 나뉘기 때문 - self attention(masked), Encoder-Decoder attention으로 나뉨
        trg = self.norm(trg + self.dropout(self.self_attn(trg, trg, trg, trg_mask)))
        # self attention(masked) 부분
        trg = self.norm(trg + self.dropout(self.src_attn(trg, src, src, src_mask)))
        # Encoder-Decoder attention 부분 (Query만 trg에서, Key, Value는 src에서 가져옴)
        trg = self.norm(trg + self.dropout(self.feed_forward(trg)))
        return trg

class MultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super(MultiheadAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        # attention의 정상적인 연산을 위해 hidden_dim을 num_heads로 나누면 나누어 떨어져야함, 만약 그렇지 않으면 실행이 중단 됨
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # 총 연산할 head의 dimension을 구하기 위해 hidden_dim을 num_heads로 나눔 -> 정수값이 head의 dim으로 출력
        
        # Q,K,V, O 모두 행렬곱을 사용하기 때문에 nn.Linear로 생성
        self.fc_q = nn.Linear(hidden_dim, hidden_dim)
        self.fc_k = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        # attention 과정에서 scaling을 하기 위해 진행
        
    def forward(self, query, key, value, mask=None):
        # attention 과정에서의 forward 부분은 차원의 순서가 중요함 -> permute로 계속 변경하면서 진행해야함
        batch_size = query.shape[0]
        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # hidden_dim을 num_heads와 head_dim으로 나눔 -> 이렇게 함으로써 각 헤드에 대한 정보를 병렬로 처리할 수 있음
        # permute(0, 2, 1, 3)를 사용하여 num_heads와 seq_len 차원을 교환 -> 각 헤드에 대한 정보가 서로 다른 차원에 나란히 배치

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # Query와 Key 사이의 유사도 연산 후 scaling 진행

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = F.softmax(energy, dim=-1)
        # attention의 마지막 부분인 softmax를 취함
        x = torch.matmul(self.dropout(attention), V)
        
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_dim)
        # attention 연산 후 다시 차원을 원상복구 시킴
        
        x = self.fc_o(x)
        return x

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # fully connected layer의 결과값에 relu를 수행한 값과 기존의 x를 같이 넣어서 x를 생성
        # -> residual connection 구현
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
