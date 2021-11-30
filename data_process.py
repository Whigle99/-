import pandas as pd
from ltp import LTP
from collections import defaultdict

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


def undersampling():
    pass


def load_stopwords():
    with open('../data/stopwords.txt', encoding='UTF-8') as fp:
        content = fp.read()
    sw = [line.strip() for line in content]
    return sw


def get_train_data(path_train, path_test):
    data_train = pd.read_csv(path_train, encoding='UTF-8', header=None, names=['Sentence', 'Label'],
                             index_col=False)
    data_test = pd.read_csv(path_test, encoding='UTF-8', header=None, names=['Sentence', 'Label'],
                            index_col=False)
    data_train = data_train[1:]
    data_train = data_train.dropna(axis=0, how='any')
    data_test = data_test[1:]
    data_test = data_test.dropna(axis=0, how='any')

    ltp = LTP()
    segments = []
    stopwords = load_stopwords()
    for index in range(data_train.shape[0]):
        if (index + 1) % 100 == 0:
            print('i:', index + 1)
        try:
            seg, h = ltp.seg([data_train.iloc[index]['Sentence']])
            seg = seg[0]
            for s in seg:
                if s in stopwords:
                    seg.remove(s)
            if len(seg) == 0:
                seg.append('<unk>')
            data_train.iloc[index]['Sentence'] = seg
            segments.append(seg)
        except:
            print('eeeee', index)
            pass

    for index in range(data_test.shape[0]):
        if (index + 1) % 100 == 0:
            print('j', index + 1)
        try:
            seg, h = ltp.seg([data_test.iloc[index]['Sentence']])
            seg = seg[0]
            for s in seg:
                if s in stopwords:
                    seg.remove(s)
            if len(seg) == 0:
                seg.append('<unk>')
            data_test.iloc[index]['Sentence'] = seg
            segments.append(seg)
        except:
            print('eeeeee', index)
            pass

    vocab = Vocab.build(segments)

    data_train = [
        (vocab.convert_tokens_to_ids(data_train.iloc[index]['Sentence']), int(data_train.iloc[index]['Label'])) for
        index in range(data_train.shape[0])]
    data_test = [
        (vocab.convert_tokens_to_ids(data_test.iloc[index]['Sentence']), int(data_test.iloc[index]['Label'])) for
        index in range(data_test.shape[0])]

    return data_train, data_test, vocab


def data_without_segment(path_train, path_test):
    data_train = pd.read_csv(path_train, encoding='UTF-8', header=None, names=['Sentence', 'Label'],
                             index_col=False)
    data_test = pd.read_csv(path_test, encoding='UTF-8', header=None, names=['Sentence', 'Label'],
                            index_col=False)

    data_train = data_train[1:]
    # print(data_train.shape)
    data_train = data_train.dropna(axis=0, how='any')
    data_test = data_test[1:]
    data_test = data_test.dropna(axis=0, how='any')

    data_all = pd.concat([data_train, data_test], ignore_index=True)

    sens = list(data_all['Sentence'])
    sens = list(map(lambda x: [x], sens))

    vocab = Vocab.build(sens)

    data_train = [
        (vocab.convert_tokens_to_ids(data_train.iloc[index - 1]['Sentence']), int(data_train.iloc[index - 1]['Label']))
        for
        index in range(data_train.shape[0])]
    data_test = [
        (vocab.convert_tokens_to_ids(data_test.iloc[index - 1]['Sentence']), int(data_test.iloc[index - 1]['Label']))
        for
        index in range(data_test.shape[0])]

    return data_train, data_test, vocab
