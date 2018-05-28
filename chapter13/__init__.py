# encoding:utf-8
import os
import re
import tarfile
import urllib.request

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


# 下载文件并解压
def download_data():
    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    filepath = 'aclImb_v1.tar.gz'
    if not os.path.isfile(filepath):
        result = urllib.request.urlretrieve(url, filepath)
        print('downloaded:', result)
    else:
        print('file exist!')
    if not os.path.exists('aclImdb'):
        tfile = tarfile.open('aclImb_v1.tar.gz', 'r:gz')
        result = tfile.extractall('')
        print(result)
    else:
        print('already extracted!')


# 使用正则表达是移除html标签
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


# 读取数据
def read_files(filetype):
    path = 'aclImdb/'
    file_list = []

    positive_path = path + filetype + '/pos/'
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    negtive_path = path + filetype + '/neg/'
    for f in os.listdir(negtive_path):
        file_list += [negtive_path + f]

    all_labels = ([1] * 12500 + [0] * 12500)

    all_texts = []
    num = 0
    for fi in file_list:
        num += 1
        with open(fi, encoding='utf-8') as file_input:
            all_texts += [rm_tags(''.join(file_input.readline()))]
        if num % 5000 == 0:
            print(num, 'files processed!')
    return all_labels, all_texts


# 建立token
def text_to_vec(train_texts, test_texts):
    token = Tokenizer(num_words=2000)
    token.fit_on_texts(train_texts)
    train_seq = token.texts_to_sequences(train_texts)
    test_seq = token.texts_to_sequences(test_texts)
    x_train = sequence.pad_sequences(train_seq, maxlen=100)
    x_test = sequence.pad_sequences(test_seq, maxlen=100)
    return x_train, x_test


if __name__ == '__main__':
    download_data()
    train_labels, train_texts = read_files('train')
    test_labels, test_texts = read_files('test')
    tokens = create_token(texts)
