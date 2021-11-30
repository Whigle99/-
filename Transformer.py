import math
import pandas as pd
import re
from tqdm.auto import tqdm
# from ltp import LTP
from code.data_process import Vocab, get_train_data, data_without_segment
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import pickle


def collate_fn(examples):
    # print('ex',examples)
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # print('in',inputs)
    # print('len',lengths)
    # print('tar',targets)

    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, lengths, targets


class BowDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += self.pe[:x.size(0), :]
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class, dim_feedforward=512, num_head=2, num_layers=2,
                 dropout=0.1, max_len=256, activation: str = 'relu'):
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoding(embedding_dim, dropout, max_len)  # 位置编码
        # encoder
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_head, dim_feedforward, dropout, activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 输出层
        self.output = nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        inputs = torch.transpose(inputs, 0, 1)
        hidden_states = self.embeddings(inputs)
        hidden_states = self.position_embedding(hidden_states)
        attention_mask = length_to_mask(lengths) == False
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states = hidden_states[0, :, :]
        output = self.output(hidden_states)
        log_probs = F.log_softmax(output, dim=1)
        return log_probs


def length_to_mask(lengths):
    max_len = torch.max(lengths)
    max_len = int(max_len)
    mask = torch.arange(max_len).expand(lengths.shape[0], max_len).cuda() < lengths.unsqueeze(1)
    return mask


def dynamic_adjust_lr(optimizer, epoch, start_lr):
    decent = epoch // 5
    lr = start_lr - decent * 0.0001
    # lr = 0.0007
    for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embedding_dim = 128
    hidden_dim = 128
    num_class = 6
    batch_size = 64
    num_epoch = 5
    start_lr = 0.001
    # 数据载入
    train_data, test_data, vocab = data_without_segment('../data/train_one_sen.csv', '../data/test_one_sen.csv')
    len_with_seg = len(vocab)
    # print(len_with_seg)
    train_dataset = BowDataset(train_data)
    test_dataset = BowDataset(test_data)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
    # with open('train_dl.pk','wb') as train_dl:
    #     pickle.dump(train_data_loader,train_dl)
    # with open('test_dl.pk','wb') as test_dl:
    #     pickle.dump(test_data_loader,test_dl)
    # with open('vocab_dl.pk', 'wb') as vocab_dl:
    #     pickle.dump(vocab,vocab_dl)
    # with open('vocab_dl.pk', 'rb') as pk:
    #     vocab = pickle.load(pk)
    # train_data, test_data, vocab = data_without_segment('../data/train_sen.csv','../data/test_sen.csv')
    # train_data, test_data, vocab = get_train_data('../data/train_sen.csv', '../data/test_sen.csv')

    # print(len_with_seg)

    # train_dataset = BowDataset(train_data)
    # test_dataset = BowDataset(test_data)
    # train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    # test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)
    # with open('../pk/train_with_seg.pk', 'wb') as train_dl:
    #     pickle.dump(train_data_loader, train_dl)
    # with open('../pk/test_with_seg.pk', 'wb') as test_dl:
    #     pickle.dump(test_data_loader, test_dl)
    # with open('../pk/vocab_with_seg.pk', 'wb') as vocab_dl:
    #     pickle.dump(vocab, vocab_dl)
    # with open('../pk/vocab_with_seg.pk', 'rb') as pk:
    #     vocab = pickle.load(pk)
    # with open('../pk/train_with_seg.pk', 'rb') as pk:
    #     train_data_loader = pickle.load(pk)
    # with open('../pk/test_with_seg.pk', 'rb') as pk:
    #     test_data_loader = pickle.load(pk)
    # len_with_seg = len(vocab)

    model = Transformer(len_with_seg, embedding_dim, hidden_dim, num_class)
    model.to(device)
    nll_loss = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    model.train()

    # 迭代训练
    for epoch in range(num_epoch):
        total_loss = 0
        dynamic_adjust_lr(optimizer, epoch, start_lr)
        for batch in tqdm(train_data_loader, desc=f'Training Epoch {epoch}'):
            inputs, lengths, targets = [x.to(device) for x in batch]
            log_probs = model(inputs, lengths)
            loss = nll_loss(log_probs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Loss:{total_loss:.2f}')

    # 测试结果
    acc = 0
    s_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for batch in tqdm(test_data_loader, desc=f'Testing Epoch '):
        inputs, lengths, targets = [x.to(device) for x in batch]
        with torch.no_grad():
            output = model(inputs, lengths)
            # print(int(output.argmax(dim=1)),int(targets))
            s_dict[int(output.argmax(dim=1))] += 1
            acc += (output.argmax(dim=1) == targets).sum().item()
    print(f'Acc:{acc / len(test_data_loader):.2f}')
    total = 0
    for k in s_dict.keys():
        total += s_dict[k]
    for k in s_dict.keys():
        s_dict[k] /= total
    print(s_dict)
#
# 5    0.446983
# 0    0.209032
# 1    0.153704
# 4    0.113429
# 2    0.064836
# 3    0.012016
#
# 5    0.435196
# 0    0.221165
# 1    0.158145
# 4    0.098692
# 2    0.070155
# 3    0.016647

