import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import jieba
from torchtext.legacy import data as data
import torchtext
import random
from sklearn.metrics import f1_score, precision_score, recall_score
import os


def get_score(y_ture, y_pred):
    y_ture = np.array(y_ture)
    y_pred = np.array(y_pred)
    f1 = f1_score(y_ture, y_pred, average='macro')
    p = precision_score(y_ture, y_pred, average='macro')
    r = recall_score(y_ture, y_pred, average='macro')
    return f1, p, r


def cut(text):
    return [word for word in jieba.cut(text)]


text = data.Field(sequential=True, tokenize=cut, include_lengths=True)
label = data.LabelField(sequential=False, use_vocab=False)
fields = [('Text', text), ('Label', label)]
test_fields = [('Text', text)]
train_data = data.TabularDataset(path='../data/train2.csv', format='csv', fields=fields, skip_header=True)
test_data = data.TabularDataset(path='../data/test_ych.csv', format='csv', fields=test_fields, skip_header=True)

train_data, dev_data = train_data.split(split_ratio=0.95, random_state=random.seed(1234))
# for i in range(len(test_data)):
#     print(vars(test_data[i]))

# pretrained_name = 'sgns.baidubaike.bigram-char'
pretrained_name = 'sgns.weibo.word'
pretrained_path = '../embedding'
vectors = torchtext.vocab.Vectors(name=pretrained_name, cache=pretrained_path)
text.build_vocab(train_data, dev_data, test_data, vectors=vectors)

train_iter = data.Iterator(
    train_data,  # 需要生成迭代器的数据集
    batch_size=128,  # 每个迭代器分别以多少样本为一个batch
    sort_key=lambda x: len(x.Text),
    sort_within_batch=True
    # 按什么顺序来排列batch，这里是以句子的长度，就是上面说的把句子长度相近的放在同一个batch里面
)
dev_iter = data.Iterator(dev_data, batch_size=128, train=False, sort=False)
test_iter = data.Iterator(test_data, batch_size=1, train=False, sort=False)


class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim, vectors, class_num):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding = self.embedding.from_pretrained(vectors)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, num_layers=3)
        self.fc = nn.Linear(hidden_size * 2, class_num)

    def forward(self, x, length):
        x = self.embedding(x)  # 句子长度，batch_size,embedding_dim
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, length, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(packed)  # seqlen,bacth_size,2*hidden_size
        output = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output = self.fc(output)  # bacth_size,class_num
        return output


def dynamic_adjust_lr(optimizer, epoch, start_lr):
    decent = epoch // 2
    lr = start_lr - decent * 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr


def train_step(epoch, train_iter, model):
    correct_sum = 0
    loss_sum = 0
    model.train()
    dynamic_adjust_lr(optimizer,epoch,learning_rate)
    for batch in train_iter:
        feature, length = batch.Text
        target = batch.Label
        feature, length, target = feature.to(device), length.to(device), target.to(device)
        length = length.cpu()
        optimizer.zero_grad()
        output = model(feature, length)
        loss = criterion(output, target)  # 计算损失函数 采用交叉熵损失函数
        loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
        optimizer.step()  # 放在loss.backward()后进行参数的更新
        loss_sum += loss.item()
        correct = (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
        correct_sum += correct
    # loss_avg = loss_sum / len(train_iter.dataset)
    # train_acc = 100.0 * correct_sum / len(train_iter.dataset)  # 计算每个mini batch中的准确率
    # print('epoch:{} - loss: {:.6f}  acc:{:.4f}'.format(epoch, loss_avg, train_acc))


def eval_step(epoch, dev_iter, model):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch in dev_iter:
            feature, length = batch.Text
            label = batch.Label
            feature, length, label = feature.to(device), length.to(device), label.to(device)
            length = length.cpu()
            output = model(feature, length)
            pred = torch.max(output, 1)[1].cpu().numpy().tolist()
            y_pred.extend(pred)
            true = label.cpu().numpy().tolist()
            y_true.extend(true)
    f1, p, r = get_score(y_true, y_pred)
    print('epoch:{} - acc: {:.4f}  recall:{:.4f}  f1:{:.4f}'.format(epoch, p, r, f1))
    return f1


def save_model(out_dir, model, epoch):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.save({
        "model": model.state_dict(),
    }, os.path.join(out_dir, "bests.tar"))


def load_model(model, checkpoint_path):
    assert os.path.exists(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'], strict=False)
    return model


def test(dest_iter, model):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch in test_iter:
            feature, length = batch.Text
            feature, lengt = feature.to(device), length.to(device)
            length = length.cpu()
            output = model(feature, length)
            pred = (torch.max(output, 1)[1].cpu().numpy() + 1).tolist()
            y_pred.extend(pred)
    return y_pred


class_num = 6
vocab_size = len(text.vocab)
embedding_dim = text.vocab.vectors.size()[-1]
hidden_size = 256
vectors = text.vocab.vectors
learning_rate = 0.0020
epochs = 10
batch_size = 64
save_dir = './/save/LSTM'
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
save_step = 2

model = LSTM(vocab_size=vocab_size, hidden_size=hidden_size, embedding_dim=embedding_dim,
             vectors=vectors, class_num=class_num)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
criterion = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(1, epochs + 1):
    train_step(epoch, train_iter, model)
    f1 = eval_step(epoch, dev_iter, model)
    if f1 > best_acc:
        best_acc = f1
#         save_model(save_dir, model, epoch)

# model_path = './save/LSTM/bests.tar'
# model = load_model(model,model_path)
# model.to(device)

# predict = test(test_iter, model)
# ids = [(i + 1) for i in range(len(predict))]
# df = pd.DataFrame({'ID': ids, 'Last Label': predict})
# df.to_csv('./result.csv', columns=['ID', 'Last Label'], index=False)
