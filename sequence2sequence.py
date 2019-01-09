import tensorflow as tf
import math
from tensorflow import logging
from tensorflow.contrib import seq2seq
from tensorflow import layers

vocab_size=15939
img_feature_dim=2048


def create_rnn_cell(hidden_dim, cell_type):
    '''根据cell_type 返回特定的cell'''
    if cell_type == 'lstm':
        return tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
    elif cell_type == 'gru':
        return tf.contrib.rnn.GRUCell(hidden_dim)
    else:
        raise Exception("%s has not been supported" % cell_type)

def dropout(cell, keep_prob):
    '''Wrap cell with dropout，这里是专门为LSTM搞的'''
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)


def get_train_model(hps, vocab_size, img_feature_dim):
    # img_feature_dim 图像特征的维度
    batch_size = hps.batch_size

    img_feature = tf.placeholder(tf.float32, (batch_size, img_feature_dim))
    sentence = tf.placeholder(tf.int32, (batch_size, None))
    input_sentence_len = tf.placeholder(tf.int32,shape =(batch_size,))
    mask = tf.placeholder(tf.float32, (batch_size, None))
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step', trainable=False)

    # prediction process
    # sentence: [a,b,c,d,e] 这个是ground trues
    # input：[img,a,b,c,d,e] 所以需要将img_feature reshape成embedding_word 类似的形状，
    # 好拼接在一起，第二个维度上面进行拼接
    # img_feature:[0.4,0.3,10,2,5]

    # 下面是真实的预测场景
    # predict #1: img_feature -> embedding_img ->lstm -> a
    # predict #2:a -> embedding_word -> lstm -> (b)
    # predict #3:b -> embedding_word -> lstm -> (c)
    # predict #4:c -> embedding_word -> lstm -> (d)
    # predict #5:d -> embedding_word -> lstm -> (e)
    # predict #6:e -> embedding_word -> lstm -> eos

    # Sets up the embedding layer.
    embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)
    with tf.variable_scope('embedding', initializer=embedding_initializer):
        embeddings = tf.get_variable(
            'embeddings',
            [vocab_size, hps.num_embedding_nodes],
            tf.float32)

        embed_token_ids = tf.nn.embedding_lookup(embeddings, sentence[:, 0:-1])  # 应该还剩一个词语,剩下的多半是填充的。

    img_feature_embed_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('image_feature_embed', initializer=img_feature_embed_init):
        embed_img = tf.layers.dense(img_feature, hps.num_embedding_nodes)
        embed_img = tf.layers.batch_normalization(embed_img)
        embed_img = tf.nn.relu(embed_img)


        embed_img = tf.expand_dims(embed_img, 1)  # 这个是在第1这个维度上面扩展的，扩展后embed_img 变成了
        # (batchsize,1,32)
        # embed_token_ids现在的形状是(batchsize,num_timesteps-1,32),所以下面进行了拼接。
        embed_inputs = tf.concat([embed_img, embed_token_ids], axis=1)
        # 现在embed_inputs的维度是(batchsize,num_timesteps,32)

    decoder_output_projection = layers.Dense(
        vocab_size,
        dtype=tf.float32,
        use_bias=False,
        name='decoder_output_projection'
    )

    # Sets up LSTM network.
    scale = 1.0 / math.sqrt(hps.num_embedding_nodes + hps.num_lstm_nodes[-1]) / 3.0
    lstm_init = tf.random_uniform_initializer(-scale, scale)
    with tf.variable_scope('lstm_nn', initializer=lstm_init) as train_scope:
        cells = []
        for i in range(hps.num_lstm_layers):
            cell = create_rnn_cell(hps.num_lstm_nodes[i], hps.cell_type)
            cell = dropout(cell, keep_prob)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells)

        initial_state = cell.zero_state(hps.batch_size, tf.float32)
        # rnn_outputs: [batch_size, num_timesteps, hps.num_lstm_node[-





        # seq2seq的一个类，用来帮助feeding参数。
        training_helper = seq2seq.TrainingHelper(
            inputs=embed_inputs,
            sequence_length=input_sentence_len,
            time_major=False,
            name='training_helper'
        )

        training_decoder = seq2seq.BasicDecoder(
            cell=cell,
            helper=training_helper,
            initial_state=initial_state
        )

        # decoder在当前的batch下的最大time_steps
        max_decoder_length = tf.reduce_max(
            input_sentence_len
        )

        (
            outputs,
            final_state,
            final_sequence_lengths
        ) = seq2seq.dynamic_decode(
            decoder=training_decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=max_decoder_length,
            swap_memory=True,
            scope=train_scope
        )
        # rnn_outputs, _ = tf.nn.dynamic_rnn(cell,
        #                                    embed_inputs,
        #                                    initial_state=initial_state)  # 这里实际上是有一个sequence length的参数，
        # 但是实际操作中是忽略的这个参数，因为LSTM的输入拼接了图像特征的缘故吧，并且使用了mask来标明数据位和填充位

        rnn_outputs = outputs.rnn_output
        print('rnn_outputs ', rnn_outputs)


    # Sets up the fully-connected layer.因为我们需要在[batch_size, num_timesteps, hps.num_lstm_node[-1]] 上的第3个维度上去做全连接
    # 因此，我们需要把batch_size,num_timesteps 合并成1个维度，因此我们使用reshape函数。
    # fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    # with tf.variable_scope('lstm_nn/fc', initializer=fc_init):
    #     rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hps.num_lstm_nodes[-1]])
    #     fc1 = tf.layers.dense(rnn_outputs_2d, hps.num_fc_nodes, name='fc1')
    #     fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
    #     fc1_dropout = tf.nn.relu(fc1_dropout)
    #     logits = tf.layers.dense(fc1_dropout, vocab_size, name='logits')

        decoder_logits_train = decoder_output_projection(
            outputs.rnn_output
        )

    masks = tf.sequence_mask(
        lengths= input_sentence_len,
        maxlen=max_decoder_length,
        dtype=tf.float32,
        name='masks'
    )

    with tf.variable_scope('loss'):

        '''
        这个tf.nn.sparse_softmax_cross_entropy_with_logits中，我们需要做三件事情：
        1.对logits做softmax
        2.对labels 做one-hot 编码，这label做了one-hot编码以后，形状也变成了（sentence，vocab_size）,和logits的size相同，
        正好可以作交叉熵。
        3.计算它们的交叉熵
        '''

        loss = seq2seq.sequence_loss(
            logits=decoder_logits_train,
            targets=sentence,
            weights=masks,
            average_across_timesteps=True,
            average_across_batch=True
        )

        logits_flatted = tf.reshape(decoder_logits_train,(-1,vocab_size))
        prediction = tf.argmax(logits_flatted, 1, output_type=tf.int32)
        sentence_flatten = tf.reshape(sentence, [-1])  # 因为我们把LSTM的输出给展平了，所以sentence也需要展平
        mask_flatten = tf.reshape(mask, [-1])
        mask_flatten = tf.cast(mask_flatten,tf.float32)

        correct_prediction = tf.equal(prediction, sentence_flatten)
        print(correct_prediction.get_shape)
        print(mask_flatten.get_shape)
        correct_prediction_with_mask = tf.multiply(
            tf.cast(correct_prediction, tf.float32),
            mask_flatten)

        mask_sum = tf.reduce_sum(mask_flatten)
        accuracy = tf.reduce_sum(correct_prediction_with_mask) / mask_sum



        tf.summary.scalar('loss', loss)

    with tf.variable_scope('train_op'):
        tvars = tf.trainable_variables()
        for var in tvars:
            logging.info("variable name: %s" % (var.name))
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, tvars), hps.clip_lstm_grads)

        for grad, var in zip(grads, tvars):
            tf.summary.histogram('%s_grad' % (var.name), grad)
        optimizer = tf.train.AdamOptimizer(hps.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    return ((img_feature, sentence,input_sentence_len, mask, keep_prob),
            (loss,accuracy, train_op),
            global_step)

if __name__ == '__main__':
    tensor1 = tf.constant([[1,2,3],[4,5,6]])
    print(tensor1.get_shape)
    tensor2 = tf.expand_dims(tensor1,1)
    print(tensor2.get_shape)

    with tf.Session() as sess:
        print(sess.run(tensor1))
        print(sess.run(tensor2))
