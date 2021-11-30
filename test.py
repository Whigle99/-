import pandas as pd
import re


def labels_to_labelList(lables: str):
    label_list = list()
    try:
        for label in lables:
            label_list.append(int(label))
    except:
        print(lables)
    return label_list


def line_to_sentences(line: str, labels,if_test=False):
    lab_list = labels_to_labelList(labels)
    sentences = re.split('__eou__', line)
    pro_sen = []
    i = 0
    if not if_test:
        for s in sentences:  # 去掉&nbsp
            # print(s)
            s = ''.join(s.split())
            s.replace('\n', '').replace('\t', '')
            s += str(lab_list[i])
            i += 1
            pro_sen.append(s)
    else:
        for s in sentences[:-1]:  # 去掉&nbsp

            s = ''.join(s.split())
            s.replace('\n', '').replace('\t', '')
            s += str(lab_list[i])
            i += 1
            pro_sen.append(s)
        last = ''.join(sentences[-1].split())
        last.replace('\n', '').replace('\t', '')
        pro_sen.append(last)

    return pro_sen


def concat_to_one_sentence(text: str):
    sentences = re.split('__eou__', text)
    pro_sen = ''
    for s in sentences:  # 去掉&nbsp
        s = ''.join(s.split())
        s.replace('\n', '').replace('\t', '')
        pro_sen += s
    return pro_sen


def qu_chong(S):  # 去除相邻重复字符
    str1 = [""]
    for i in S:
        if i == str1[-1]:
            str1.pop()
        else:
            str1.append(i)
    return ''.join(str1)


def get_train_sen(line: str, labels):
    sens = line_to_sentences(line, labels)
    t_sens = ['']
    for sen in sens:
        t_sens.append(t_sens[-1] + sen)
    t_sens.pop(0)
    return t_sens


def get_test_sen(line: str, labels):
    sens = line_to_sentences(line, labels,if_test=True)
    t_sens = ['']
    for sen in sens:
        t_sens.append(t_sens[-1] + sen)
    t_sens.pop(0)
    return t_sens


# raw_data = pd.read_csv('../data/train_data.csv', encoding='UTF-8', header=None, names=['ID', 'Text', 'Labels'],
#                        index_col=False)
raw_data = pd.read_csv('../data/syc.csv', encoding='UTF-8', header=None, names=['ID', 'Text', 'Labels'],
                       index_col=False)

raw_data1 = raw_data[1:][['Text', 'Labels']]
raw_data1 = raw_data1.dropna(axis=0, how='any')

# df_train = raw_data1.sample(frac=0.9)
df_train = raw_data1
# df_test = raw_data1[~raw_data1.index.isin(df_train.index)]
df_test = raw_data1
df_train = df_train.reset_index()[['Text', 'Labels']]
df_test = df_test.reset_index()[['Text', 'Labels']]
pro_train_data = pd.DataFrame(columns=['Sentence', 'Label'])
pro_test_data = pd.DataFrame(columns=['Text'])

# for index in range(df_train.shape[0]):
#     try:
#         cur_row = df_train.iloc[index]
#     except:
#         print(1, index)
#
#     sens = get_train_sen(cur_row['Text'],cur_row['Labels'])
#
#     for sen in sens:
#         label = sen[-1]
#         qc_sen = qu_chong(sen[:-1])
#         pattern = re.compile('[^\u4e00-\u9fa5^a-z^A-Z^0-9^!]')
#         new_sen = re.sub(pattern, '', qc_sen)  # 只保留中英文,数字,!
#         pro_train_data = pro_train_data.append([{'Sentence': new_sen, 'Label': label}], ignore_index=True)


for index in range(df_test.shape[0]):
    try:
        cur_row = df_test.iloc[index]
        print(cur_row)

    except:
        print(2, index)
    try:
        sen = get_test_sen(cur_row['Text'], cur_row['Labels'])
        sen = sen[-1]
        # label = sen[-1]
    except Exception as e:
        continue

    pro_test_data = pro_test_data.append([{'Text': sen}], ignore_index=True)
#
# pro_train_data.to_csv('../data/90train.csv', encoding='UTF-8', index=False)
pro_test_data.to_csv('../data/syctest.csv', encoding='UTF-8', index=False)

# train_sen = pd.read_csv('../data/train_one_sen.csv')
# test_sen = pd.read_csv('../data/result.csv')
# series1 = pd.Series(train_sen['Label'].values)
# series2 = pd.Series(test_sen['Last Label'].values)
# vc1 = pd.value_counts(series1)
# vc1 = vc1/vc1.sum()
# vc2 = pd.value_counts(series2)
# vc2 = vc2/vc2.sum()
# print(vc1)
# print(vc2)
