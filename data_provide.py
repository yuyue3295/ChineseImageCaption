import os
import sys
import tensorflow as tf
from tensorflow import gfile
from tensorflow import logging
import pprint
import pickle
import numpy as np
import math
import jieba
import tqdm
import random

input_description_file = "./data/clean_zh_results_20130124.token"
input_img_feature_dir = "./feature_extraction_inception_v3"
input_vocab_file = "./data/zh_vocab.txt"
num_vocab_word_threshold = 2
vocab_pkl = './vocab_pkl'

class Vocab(object):
    def __init__(self, filename, word_num_threshold):
        self._id_to_word = {}
        self._word_to_id = {}
        self._unk = -1
        self._eos = -1  # 两个特殊的字符，一个是unk，一个是eos end of the sentence
        self._word_num_threshold = word_num_threshold
        self._read_dict(filename)

    def _read_dict(self, filename):
        '''
        这个函数读取vocab.txt文件，解析单词和单词出现的词频。
        构建映射，word2id，id2word，ids是词语的编号。
        '''
        with gfile.GFile(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            word, occurence = line.strip('\r\n').split('\t')
            occurence = int(occurence)  # occurence 这个就是单词出现的词频
            if word != '<UNK>' and occurence < self._word_num_threshold:
                continue
            idx = len(self._id_to_word)
            if word == '<UNK>':
                self._unk = idx
            elif word == '。':
                self._eos = idx
            if idx in self._id_to_word or word in self._word_to_id:
                raise Exception('duplicate words in vocab file')
            # 如果这个词已经在词表中出现了，那么就会跑出异常。
            self._word_to_id[word] = idx
            self._id_to_word[idx] = word

    @property
    def unk(self):
        return self._unk

    @property
    def eos(self):
        return self._eos

    def word_to_id(self, word):
        return self._word_to_id.get(word, self.unk)

    def id_to_word(self, cur_id):
        return self._id_to_word.get(cur_id, '<UNK>')

    def size(self):
        return len(self._word_to_id)

    def encode(self, sentence):
        '''
        将句子转换成为id列表，当然先是将句子转化成为单词列表。
        '''
        word_ids = [self.word_to_id(cur_word) for cur_word in list(jieba.cut(sentence))]
        return word_ids

    def decode(self, sentence_id):
        '''
        将id列表转化成为句子。
        '''
        words = [self.id_to_word(word_id) for word_id in sentence_id]
        return ''.join(words)


def parse_token_file(token_file):
    """Parses token file.得到的输出是一个dict，key图像名称，value是一个列表，列表里面是对图像描述,
        这里解析的文件是results_20130124.token
    """
    img_name_to_tokens = {}
    with gfile.GFile(token_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        img_id, description = line.strip('\r\n').split('\t')
        img_name, _ = img_id.split('#')
        img_name_to_tokens.setdefault(img_name, [])
        img_name_to_tokens[img_name].append(description)
    return img_name_to_tokens


def convert_token_to_id(img_name_to_tokens, vocab):
    """Converts tokens of each description of imgs to id. 将图片对应的描述转化成为id，
    得到的也是一个map，key是图片名称，value是关于图像的描述的ids"""
    img_name_to_token_ids = {}
    for img_name in img_name_to_tokens:
        img_name_to_token_ids.setdefault(img_name, [])
        descriptions = img_name_to_tokens[img_name]
        for description in descriptions:
            token_ids = vocab.encode(description)
            img_name_to_token_ids[img_name].append(token_ids)
    return img_name_to_token_ids


class ImageCaptionData(object):
    '''
    Provides data for image caption model.为图像提供
    以图像文件名称作为索引，嘻嘻。
    数据提供主要是提供图像名称对应的保存的特征，以及图像的相关描述，这里已经转化成
    了id的形式，后面还会根据num_timesteps对图像的描述进行裁剪或者填充，以weight的形式表明
    哪些是正常的描述数据，哪些是填充的数据。

    '''

    def __init__(self,
                 img_name_to_token_ids,#这是一个字典，key是图片名称，value是一个列表，列表中是图片中的描述，转化成为了id之后的
                 img_feature_dir,
                 vocab,
                 deterministic=False  # deterministic = FALSE 表示可以shuffle
                 ):
        self._vocab = vocab

        '''这里是获取图像特征的pickle文件路径，并添加到一个列表中'''
        self._all_img_feature_filepaths = []
        for filename in gfile.ListDirectory(img_feature_dir):  # gfile.ListDirectory代替了os.listdir
            self._all_img_feature_filepaths.append(os.path.join(img_feature_dir, filename))
        pprint.pprint(self._all_img_feature_filepaths)

        self._img_name_to_token_ids = img_name_to_token_ids
        self._indicator = 0
        self._deterministic = deterministic

        self._img_feature_filenames = []  # 第一个存储图像所有的名字。
        self._img_feature_data = []  # 这个是存储图像提取的向量。
        self._load_img_feature_pickle()  # 加载图像特征的pickle文件
        if not self._deterministic:
            self._random_shuffle()  # 打乱一下

    def _load_img_feature_pickle(self):
        '''Loads img feature data from pickle
        解析pickle，将会得到图像文件名称，以及抽取的图像特征
        这两个会分别保存到self._img_feature_filenames
        以及self._img_feature_data.中去

        '''
        for filepath in self._all_img_feature_filepaths:
            logging.info("loading %s" % filepath)
            with gfile.GFile(filepath, 'rb') as f:
                filenames, features = pickle.load(f)
                self._img_feature_filenames += filenames
                self._img_feature_data.append(features)
        # 此时self._img_feature_data存储的是[(100,1,1,2048),...,(100,1,1,2048)]
        # 之前将100个（1,1,1,2048）的矩阵通过np.vstack在竖直方向堆叠，成了（100,1,1,2048）
        self._img_feature_data = np.vstack(self._img_feature_data)  # 按第1个维度进行合并 np.hstack 才是按照第二个维度进行合并
        origin_shape = self._img_feature_data.shape
        print(origin_shape, ' origin_shape')
        self._img_feature_data = np.reshape(
            self._img_feature_data, (origin_shape[0], origin_shape[3]))
        # 经过变换，就把(？,1,1,2048) 变成了（？，2048）的形状了。

        self._img_feature_filenames = np.asarray(self._img_feature_filenames)  # list还真不行，必须np
        print(self._img_feature_data.shape)
        print(self._img_feature_filenames.shape)
        if not self._deterministic:
            self._random_shuffle()

    def size(self):
        return len(self._img_feature_filenames)

    def img_feature_size(self):
        '''
        返回图像特征的维度是多大，如果是self._img_feature_data.shape[0]就是
        图像的总数，这里是30000多，如果是self._img_feature_data.shape[1]，就是
        2048
        '''
        return self._img_feature_data.shape[1]

    def _random_shuffle(self):
        '''Shuffle data randomly.'''
        p = np.random.permutation(self.size())
        self._img_feature_filenames = self._img_feature_filenames[p]
        self._img_feature_data = self._img_feature_data[p]

    def _img_desc(self, filenames):
        '''Gets descriptions for filenames in batch.为一个batch中的每个img找到对应的描述。
        找描述的方式是，从name_to_token字典中根据图像名称取出文本描述列表，从列表中随机选择一个作为该
        图像的描述
        根据这个batch的最大长度来进行padding

        '''
        batch_sentence_ids = []
        batch_sentence_lengths = []
        batch_weights = []
        for filename in filenames:
            token_ids_set = self._img_name_to_token_ids[filename]
            # chosen_token_ids = random.choice(token_ids_set) # 这里应该从描述中随机选择一个作为他的描述
            chosen_token_ids = token_ids_set[1] #第二个描述做为它的描述。第二个对白通常来说更短。
            chosen_token_length = len(chosen_token_ids)

            batch_sentence_lengths.append(chosen_token_length)
            batch_sentence_ids.append(chosen_token_ids)



        #现在来padding
        padded_batch_sentence_ids = []
        max_len = max(batch_sentence_lengths)
        for chosen_token_ids,length in list(zip(batch_sentence_ids,batch_sentence_lengths)):
            weight = [1 for i in range(length)]

            remaining_length = max_len - length
            chosen_token_ids += [self._vocab.eos for i in range(remaining_length)]
            weight += [0 for i in range(remaining_length)]  # 否则做填充，用0来填充
            batch_weights.append(weight)
            padded_batch_sentence_ids.append(chosen_token_ids)




        # 都转化成为numpy数组。
        batch_sentence_ids = np.asarray(padded_batch_sentence_ids)
        batch_sentence_lengths = np.asarray(batch_sentence_lengths)
        batch_weights = np.asarray(batch_weights)
        return batch_sentence_ids,batch_sentence_lengths, batch_weights

    def next(self, batch_size):
        '''这个是返回下一批数据'''
        end_indicator = self._indicator + batch_size
        if end_indicator > self.size():
            if not self._deterministic:
                self._random_shuffle()
            self._indicator = 0
            end_indicator = self._indicator + batch_size
        assert end_indicator <= self.size()

        batch_img_features = self._img_feature_data[self._indicator: end_indicator]
        batch_img_names = self._img_feature_filenames[self._indicator: end_indicator]
        batch_sentence_ids, batch_sentence_lengths,batch_weights = self._img_desc(batch_img_names)  # 这里是根据batch_img_names找描述
        '''
        batch_weights的作用，比如sentence_ids:[100,101,102,10,3,0,0,0]->[1,1,1,1,1,0,0,0]表示这个senten_ids里面有5个是真实的
        token_id,由单词映射而来，有3个是填充而来，这后面三个不参与计算，也不计算它们的损失函数，相当于mask的作用。
        '''

        self._indicator = end_indicator
        return batch_img_features, batch_sentence_ids,batch_sentence_lengths, batch_weights, batch_img_names

if __name__ == '__main__':
    # vocab = Vocab(input_vocab_file, num_vocab_word_threshold)
    # with open(vocab_pkl,'wb') as g:
    #     pickle.dump(vocab,g)
    vocab = pickle.load(open('./vocab_pkl', 'rb'))
    img_name_to_tokens = parse_token_file(input_description_file)
    img_name_to_token_ids = convert_token_to_id(img_name_to_tokens, vocab)
    img_feature_dir = './feature_extraction_inception_v3'
    data_provider = ImageCaptionData(img_name_to_token_ids,img_feature_dir,vocab)

    # for i in range(1):
    #     print('开始了嘻嘻')
    #     print(data_provider.next(batch_size=3))
    print(vocab.size())






