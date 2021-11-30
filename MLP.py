import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import RNN


class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.activate = F.relu
        self.linear2 = nn.Linear(hidden_dim, num_class)

    def forward(self, data):
        embeddings = self.embedding(data)
        embedding = embeddings.mean(dim)
        hidden = self.linear1(embedding)
        activation = self.activate(hidden)
        outputs = self.linear2(activation)
        probs = F.log_softmax(outputs, dim=1)
        return probs


if __name__ == '__main__':
    # mlp = MLP(4, 5, 2)
    # d = torch.rand(3, 4)
    # print(mlp(d))

    # rnn = RNN(input_size=4, hidden_size=5, batch_first=True)
    # inputs = torch.rand(2, 3, 4)  # 批次大小，序列长度，隐藏层大小
    # outputs, hn = rnn(inputs)
    # print(outputs)

    encoder_layer = nn.TransformerEncoderLayer(d_model=4, nhead=2)
    src = torch.rand(2, 3, 4)
    print(src)
    # out=encoder_layer(src)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    memory = transformer_encoder(src)
    decoder_layer = nn.TransformerDecoderLayer(d_model=4, nhead=2)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    out_part = torch.rand(2, 3, 4)
    out = transformer_decoder(out_part, memory)
    print(out)
