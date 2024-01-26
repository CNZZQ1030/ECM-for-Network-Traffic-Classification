import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

torch.manual_seed(2020)
torch.cuda.manual_seed_all(2020)
np.random.seed(2020)
random.seed(2020)
torch.backends.cudnn.deterministic = True


class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=128):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)
        attn = self.softmax(attn)
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        out = self.mv(attn)

        return out


class OneDimCNN(nn.Module):
    """docstring for OneDimCNN"""

    def __init__(self, max_byte_len, d_dim=32,
                 kernel_size=3, filters=256, dropout=0.1):
        super(OneDimCNN, self).__init__()
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(in_channels=d_dim, out_channels=filters, kernel_size=kernel_size)
        self.activate = nn.ReLU()
        self.pooling = nn.MaxPool1d(kernel_size=max_byte_len - kernel_size + 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.conv1d(x.transpose(-2, -1))
        out = self.activate(out)
        out = self.pooling(out)
        out = out.view(-1, out.size(1))
        return self.dropout(out)


class ECM(nn.Module):
    """docstring for SAM"""

    def __init__(self, num_class, max_byte_len, kernel_size=3, d_dim=32, dropout=0.1, filters=256):
        super(ECM, self).__init__()
        self.posembedding = nn.Embedding(num_embeddings=max_byte_len,
                                         embedding_dim=d_dim)
        self.byteembedding = nn.Embedding(num_embeddings=300,
                                          embedding_dim=d_dim)
        self.attention = ExternalAttention(d_model=32)
        self.cnn = OneDimCNN(max_byte_len, d_dim, kernel_size, filters, dropout)
        self.fc = nn.Linear(in_features=256,
                            out_features=num_class)

    def forward(self, x, y):
        out = self.byteembedding(x) + self.posembedding(y)
        out = self.attention(out)
        out = self.cnn(out)
        out = self.fc(out)
        if not self.training:
            return F.softmax(out, dim=-1).max(1)[1]
        return out


if __name__ == '__main__':
    x = np.random.randint(0, 255, (10, 12))
    y = np.random.randint(0, 12, (10, 12))

    model = ECM(num_class=8, max_byte_len=12)

    out = model(torch.from_numpy(x).long(), torch.from_numpy(y).long())
    print(out[0])
    print(out.shape)
