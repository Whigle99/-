from collections import defaultdict
import pandas as pd
import re

import torch
from ltp import LTP
from torch.utils.data import DataLoader

from MLP import MLP
from tqdm.auto import tqdm


class Vocab:
    """
    构建词表
    """

    def __init__(self, tokens=None):
        self.idx_to_token = list()
        self.token_to_idx = dict()

        if tokens is not None:
            if '<unk>' not in tokens:
                tokens = tokens + ['<unk>']
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
            self.unk = self.token_to_idx['<unk>']

    @classmethod
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in text:
            sentence = sentence[0]
            for token in sentence:
                token_freqs[token] += 1
        unique_tokens = ['<unk>'] + (reserved_tokens if reserved_tokens else [])
        unique_tokens += [token for token, freq in token_freqs.items() if freq >= min_freq and token != '<unk>']
        return cls(unique_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, token):
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        return [self.idx_to_token[index] for index in indices]


def get_train_data(path):
    data = pd.read_csv(path, encoding='UTF-8', header=None, names=['Text', 'Label'],
                       index_col=False)
    data = data[1:][['Text', 'Label']]
    data = data.dropna(axis=0, how='any')
    data = data[:10]
    # print(data.head())

    ltp = LTP()
    segments = []
    i = 0
    for index, row in data.iterrows():
        # i+=1
        if index - 1 < data.shape[0]:
            try:
                seg, h = ltp.seg([data.iloc[index - 1]['Text']])
                segments.append(seg)

            except:
                print(data.iloc[index - 1])
                print(index)
        # if i== 5:
        #     break

    # print(segments)
    vocab = Vocab.build(segments)
    # print(vocab.token_to_idx)
    train_data = [(vocab.convert_tokens_to_ids(data.iloc[index - 1]['Text']), int(data.iloc[index - 1]['Label'])) for
                  index, row
                  in data.iterrows()]
    # print(train_data[0])
    return train_data, vocab


def collate_fn(examples):
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    offsets = [0] + [i.shape[0] for i in inputs]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    inputs = torch.cat(inputs)
    return inputs, offsets, targets


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
    batch_size = 32
    num_epoch = 5

    train_data, vocab = get_train_data('80train_sentences.csv')
    train_dataset = BowDataset(train_data)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    model = MLP(len(vocab), embedding_dim, hidden_dim, num_class)
    model.to(device)

    nll_loss = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f'Training Epoch {epoch}'):
            inputs, offsets, targets = [x.to(device) for x in batch]
            log_probs = model(inputs, offsets)
            loss = nll_loss(log_probs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Loss:{total_loss:.2f}')
