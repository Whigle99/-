from collections import defaultdict
import pandas as pd
import re

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from tqdm.auto import tqdm
from torch.nn import functional as F
from code.data_process import data_without_segment, Vocab


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_class):
        super(LSTM, self).__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=4)
        self.output = torch.nn.Linear(hidden_dim, num_class)

    def forward(self, inputs, lengths):
        embeddings = self.embeddings(inputs)
        x_pack = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        hidden, (hn, cn) = self.lstm(x_pack)
        outputs = self.output(hn[-1])
        log_probs = F.log_softmax(outputs, dim=-1)
        return log_probs


def collate_fn(examples):
    # print('ex',examples)
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embedding_dim = 128
    hidden_dim = 256
    num_class = 6
    batch_size = 128
    num_epoch = 5
    # 数据载入
    train_data, test_data, vocab = data_without_segment('../data/train_one_sen.csv', '../data/test_one_sen.csv')
    train_dataset = BowDataset(train_data)
    test_dataset = BowDataset(test_data)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

    model = LSTM(len(vocab), embedding_dim, hidden_dim, num_class)
    model.to(device)

    nll_loss = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    # 迭代训练
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(train_data_loader, desc=f'Training Epoch {epoch}'):
            inputs, lengths, targets = [x.to(device) for x in batch]
            log_probs = model(inputs, lengths)
            loss = nll_loss(log_probs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Loss:{total_loss:.2f}')

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
