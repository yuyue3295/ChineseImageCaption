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
from data_provide import Vocab,parse_token_file,convert_token_to_id,ImageCaptionData
from sequence2sequence import get_train_model

input_description_file = "./data/clean_zh_results_20130124.token"
input_img_feature_dir = "./feature_extraction_inception_v3"
input_vocab_file = "./data/zh_vocab.txt"
output_dir = "./local_run"
vocab_size=15939
img_feature_dim=2048
datasize= 31784
num_epochs = 50
num_step_per_epoch = int(datasize/100)
if datasize%100 !=0:
    num_step_per_epoch = num_step_per_epoch+1

if not gfile.Exists(output_dir):
    gfile.MakeDirs(output_dir)

def get_default_params():
    return tf.contrib.training.HParams(
        num_vocab_word_threshold=2,#出现频率比较低的词呢，就会被过滤掉，这里频率小于3的会被过滤掉。
        num_embedding_nodes=350,
        num_lstm_nodes=[350, 350], #每一层LSTM的大小，cell size
        num_lstm_layers=2,# 一共有两层LSTM
        num_fc_nodes=32,
        batch_size=100,
        cell_type='lstm',
        clip_lstm_grads=1.0,#梯度剪裁
        learning_rate=0.001,
        keep_prob=0.8,#每一层循环神经网络都需要添加一个dropout么？还是全连接层才添加dropout，fc层也需要。
        log_frequent=100,#每隔100会打印一次log
        save_frequent=1000,#每隔多少次保存一次
    )

def main():
    hps = get_default_params()
    placeholders, metrics, global_step = get_train_model(hps, vocab_size, img_feature_dim)
    img_feature, sentence,input_sentence_len, mask, keep_prob = placeholders
    loss, accuracy, train_op = metrics

    summary_op = tf.summary.merge_all()

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=10)

    '''
    提供数据
    '''
    vocab = pickle.load(open('./vocab_pkl', 'rb'))
    img_name_to_tokens = parse_token_file(input_description_file)
    img_name_to_token_ids = convert_token_to_id(img_name_to_tokens, vocab)
    img_feature_dir = './feature_extraction_inception_v3'
    data_provider = ImageCaptionData(img_name_to_token_ids, img_feature_dir, vocab)

    with tf.Session() as sess:
        sess.run(init_op)
        # saver.restore(sess,save_path='./local_run/image_caption_50-15899')
        writer = tf.summary.FileWriter(output_dir, sess.graph)

        for i in range(1,num_epochs+1):
            bar = tqdm.tqdm(range(num_step_per_epoch))
            for j in bar:
                batch_img_features, batch_sentence_ids, batch_sentence_lengths, batch_weights, batch_img_names =data_provider.next(hps.batch_size)
                input_vals = (batch_img_features, batch_sentence_ids,batch_sentence_lengths, batch_weights, hps.keep_prob)
                feed_dict = dict(zip(placeholders, input_vals))
                fetches = [global_step, loss, accuracy, train_op]

                should_log = (j + 1) % hps.log_frequent == 0
                if should_log:
                    fetches += [summary_op]
                outputs = sess.run(fetches, feed_dict)
                global_step_val, loss_val, accuracy_val = outputs[0:3]
                bar.set_description('%s:Epoch,Step: %5d, loss: %3.3f, accuracy: %3.3f'
                                    % (str(i),global_step_val, loss_val, accuracy_val))
                if should_log:
                    summary_str = outputs[4]
                    writer.add_summary(summary_str, global_step_val)
                    bar.set_description('%s:Epoch, Step: %5d, loss: %3.3f, accuracy: %3.3f'
                                        % (str(i),global_step_val, loss_val, accuracy_val))

            saver.save(sess, os.path.join(output_dir, "image_caption_%s" % str(i)), global_step=global_step_val)

        # for i in tf.global_variables():
        #     print(i.name)

if __name__ == '__main__':
    main()

