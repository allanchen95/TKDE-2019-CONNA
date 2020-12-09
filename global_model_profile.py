from os.path import join
import sys
sys.path.append("..")
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import multiprocessing as mp
import numpy as np 
import tensorflow as tf
from utils import eval_utils, data_utils, settings
from collections import defaultdict
import gen_train_data_profile
import logging
from time import time
from model_base import BaseNN
np.set_printoptions(suppress=True)
import gc
import random
from traitlets.config import Configurable
from traitlets import (
    Int,
    Float,
    Bool,
    Unicode,
)
def print_logging(message):
	logging.info(message)
	print(message)



class GlobalTripletModel(BaseNN):
    epsilon = Float(0.00001, help="Epsilon for Adam").tag(config=True)
    lamb = Float(0.5, help="guassian_sigma = lamb * bin_size").tag(config=True)

    def __init__(self, sess, **kwargs):
        super(GlobalTripletModel, self).__init__(**kwargs)

        self._prepare_parameter(sess)
        self._prepare_data()
        self._build_graph()
        self.saver = tf.train.Saver(max_to_keep = 3)

    def _prepare_parameter(self, sess):
        self.learning_rate = settings.LEARNING_RATE
        self.fine_tune_lr = settings.FINE_TUNE_LR
        self.batch_size = settings.BATCH_SIZE
        self.sess = sess
        self.mus = self.kernal_mus(self.n_bins, use_exact=True)
        self.sigmas = self.kernel_sigmas(self.n_bins, self.lamb, use_exact=True)
        self.weight_size = 50
        self.filter_num = 128
        print("kernel sigma = %s, mu = %s" %(str(self.sigmas), str(self.mus)))


    def _prepare_data(self):
        # self.author_emb = data_utils.load_data('/root/zhangjing/NA/emb/', "author_emb.array")
        self.author_emb = data_utils.load_data(settings.EMB_DATA_DIR, "author_emb.array")
        print_logging("Loaded author embeddings")
        # self.word_emb = data_utils.load_data('/root/zhangjing/NA/emb/', "title_emb.array")
        self.word_emb = data_utils.load_data(settings.EMB_DATA_DIR, "word_emb.array")
        print_logging("Author emb shape = %s, word emb shape = %s"% (str(self.author_emb.shape), str(self.word_emb.shape)))
        self.author_num = len(self.author_emb)
        print_logging("#author = %d" %self.author_num)
        self.word_num = len(self.word_emb)
        print_logging("#Word = %d" %self.word_num)

    	

    def _create_placeholders(self):
        self.train_words_q = tf.placeholder(tf.int32, shape=[None, None], name='train_words_q') # batch_size * max_q_len
        self.train_words_q_weights = tf.placeholder(tf.float32, shape=[None, None], name='words_idf')   # batch_size * max_q_len
        self.pos_train_words_d = tf.placeholder(tf.int32, shape=[None, None],name='pos_train_words_d') # batch_size * max_d_len
        self.pos_train_person_words_d = tf.placeholder(tf.int32, shape=[None, None, None],name='pos_train_person_words_d') # batch_size * max_paper_num * per_max_d_len
        self.neg_train_words_d = tf.placeholder(tf.int32, shape=[None, None],name='neg_train_words_d') # batch_size * max_d_len
        self.neg_train_person_words_d = tf.placeholder(tf.int32, shape=[None, None, None],name='neg_train_person_words_d') # batch_size * max_paper_num * per_max_d_len


        self.train_authors_q = tf.placeholder(tf.int32, shape=[None, None], name='train_authors_q') # batch_size * max_q_len
        self.train_authors_q_weights = tf.placeholder(tf.float32, shape=[None, None], name='authors_idf')   # batch_size * max_q_len
        self.pos_train_authors_d = tf.placeholder(tf.int32, shape=[None, None],name='pos_train_authors_d') # batch_size * max_d_len
        self.pos_train_person_authors_d = tf.placeholder(tf.int32, shape=[None, None, None],name='pos_train_person_authors_d') # batch_size * max_paper_num * per_max_d_len
        self.neg_train_authors_d = tf.placeholder(tf.int32, shape=[None, None],name='neg_train_authors_d') # batch_size * max_d_len
        self.neg_train_person_authors_d = tf.placeholder(tf.int32, shape=[None, None, None],name='neg_train_person_authors_d') # batch_size * max_paper_num * per_max_d_len      

        self.pos_paper_num = tf.placeholder(tf.int32, shape=[None],name='pos_paper_num') # [batch_size]
        self.neg_paper_num = tf.placeholder(tf.int32, shape=[None],name='neg_paper_num') # [batch_size]
        
        
        self.test_authors_q = tf.placeholder(tf.int32, shape=[None, None], name='test_authors_q') # batch_size * max_q_len
        self.test_authors_q_weights = tf.placeholder(tf.float32, shape=[None, None], name='test_authors_idf')
        self.test_words_q = tf.placeholder(tf.int32, shape=[None, None], name='test_words_q') # batch_size * max_q_len
        self.test_words_q_weights = tf.placeholder(tf.float32, shape=[None, None], name='test_words_idf')   # batch_size * max_q_len 
        self.test_authors_d = tf.placeholder(tf.int32, shape=[None, None],name='test_authors_d') # batch_size * max_d_len
        self.test_words_d = tf.placeholder(tf.int32, shape=[None, None],name='test_words_d') # batch_size * max_d_len
        self.test_person_authors_d = tf.placeholder(tf.int32, shape=[None, None, None],name='test_person_authors_d') # batch_size * max_paper_num * per_max_d_len
        self.test_person_words_d = tf.placeholder(tf.int32, shape=[None, None, None],name='test_person_words_d') # batch_size * max_paper_num * per_max_d_len
        self.test_paper_num = tf.placeholder(tf.int32, shape=[None],name='test_paper_num') # [batch_size]

        self.input_pos_feature = tf.placeholder(tf.float32, shape=[None, None], name='input_pos_feature') #ins_num * feature_size
        self.input_neg_feature = tf.placeholder(tf.float32, shape=[None, None], name='input_neg_feature') #ins_num * feature_size


        self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

    def _create_variables(self):
        self.c1 = tf.Variable(tf.constant(self.word_emb, dtype='float32', shape=[self.word_num, settings.EMB_DIM]),name='c1',trainable=False)
        self.c2 = tf.constant(0.0, tf.float32, [1, settings.EMB_DIM], name='c2')
        self.word_embeddings = tf.concat([self.c1, self.c2], 0, name='word_embedding')

        self.a1 = tf.Variable(tf.constant(self.author_emb, dtype='float32', shape=[self.author_num, settings.EMB_DIM]),name='a1',trainable=False)
        self.a2 = tf.constant(0.0, tf.float32, [1, settings.EMB_DIM], name='a2')
        self.author_embeddings = tf.concat([self.a1, self.a2], 0, name='author_embedding')

        self.t_W = self.weight_variable([self.n_bins * 2, self.n_bins], name = "t_w")
        self.t_b = tf.Variable(tf.zeros([self.n_bins]), name='t_b')

        self.m_W = self.weight_variable([self.n_bins, 1], name = "m_w")
        self.m_b = tf.Variable(tf.zeros([1]), name='m_b')
        
        self.author_W = self.weight_variable([self.n_bins, 1], name = "author_w")
        self.author_b = tf.Variable(tf.zeros([1]), name = "author_b")

        self.word_W = self.weight_variable([self.n_bins, 1], name = "word_w")
        self.word_b = tf.Variable(tf.zeros([1]), name = "word_b")

        self.fc_w1 = self.weight_variable([self.n_bins *2* 2, self.n_bins *2* 2], name = "fc_w1")
        self.fc_b1 = tf.Variable(tf.zeros([self.n_bins *2* 2]), name='fc_b1')

        self.fc_w2 = self.weight_variable([self.n_bins *2* 2, self.n_bins *2* 2], "fc_w2")
        self.fc_b2 = tf.Variable(tf.zeros([self.n_bins *2* 2]), name='fc_b2')

        self.fc_w3 = self.weight_variable([self.n_bins *2* 2, self.n_bins *2* 2], "fc_w3")
        self.fc_b3 = tf.Variable(tf.zeros([self.n_bins *2* 2]), name='fc_b3')

        self.fc_w4 = self.weight_variable([self.n_bins *2* 2, 1], "fc_w4")
        self.fc_b4 = tf.Variable(tf.zeros([1]), name='fc_b4')

        w_values = np.asarray(
            np.random.uniform(
                low=-np.sqrt(6. / (self.n_bins + self.n_bins)),
                high=np.sqrt(6. / (self.n_bins + self.n_bins)),
                size=(self.n_bins * 2, self.n_bins * 2)
            ),
            dtype=np.float32
        )
        self.att_w = tf.Variable(w_values, name='Attention.W')

        v_values = np.asarray(
            np.random.normal(scale=0.1, size=(self.n_bins *2,)),
            dtype=np.float32
        )
        self.att_v = tf.Variable(v_values, name='Attention.v')

        b_values = np.zeros((self.n_bins * 2,), dtype=np.float32)
        self.att_b = tf.Variable(b_values, name='Attention.b')


        # weights for words and authors
        self.w = tf.Variable(tf.constant(0.5), name = 'w')

        #CNN 
        #filter_shape = [filter_size, embedding_size, 1, num_filters]
        
    def AttentionLayer(self, inputs, size, lengths):
        r = tf.tanh(tf.nn.bias_add(tf.matmul(tf.reshape(inputs, [-1, size]), self.att_w), self.att_b))

        attention = tf.reshape(r, tf.shape(inputs))
        attention = tf.reduce_sum(attention * self.att_v, 2)

        att_len = tf.shape(attention)[1]   

        y = tf.exp(attention) * tf.sequence_mask(lengths, att_len, dtype=tf.float32)
        #y = tf.exp(attention)
        sum_x = tf.reduce_sum(y, 1)
        attention = y / tf.expand_dims(sum_x, 1)                 # [batch_size, time_step]

        output = tf.expand_dims(attention, 2) * inputs           # [batch_size, time_step, state_size]
        output = tf.reduce_sum(output, 1)              # [batch_size, state_size]
        return output

    def get_merge_vec(self, word_sim, author_sim):
        word_att = tf.tanh(tf.nn.bias_add(tf.matmul(word_sim, self.m_W), self.m_b))
        author_att = tf.tanh(tf.nn.bias_add(tf.matmul(author_sim, self.m_W), self.m_b))

        attention = tf.concat([word_att, author_att], 1)
        softmax = tf.nn.softmax(attention)
        a1, a2 = tf.split(softmax, num_or_size_splits=2, axis=1)
                #concat_hidden = tf.concat([a1 * name_cnn, a2 * aff_cnn, a3 * edu_cnn, a4 * pub_cnn], 1)

        merge_vec = tf.concat([a1 * word_sim, a2 * author_sim], 1)

        return merge_vec



    def get_interaction_vec(self, q_embed, d_embed, person_d_embed, q_weights, paper_num):

        normalized_q_embed = tf.nn.l2_normalize(q_embed, axis=2)
        normalized_d_embed = tf.nn.l2_normalize(d_embed, axis=2)
        tmp = tf.transpose(normalized_d_embed, perm=[0, 2, 1])



        # [batch_size, paper_num, dlen, emb]
        normalize_person_d_emb = tf.nn.l2_normalize(person_d_embed, axis = 3)
        person_tmp = tf.transpose(normalize_person_d_emb, perm = [0, 1, 3, 2])


        # similarity matrix [n_batch, qlen, dlen]
        sim = tf.einsum('ijk,ikl->ijl',normalized_q_embed, tmp, name='similarity_matrix')
        sim_shape = tf.shape(sim)
        
        # person sim mat [n_batch, paper_num, qlen, dlen]
        person_sim = tf.einsum('ijk,ipkl->ipjl',normalized_q_embed, person_tmp, name='person_similarity_matrix')
        person_sim_shape = tf.shape(person_sim)


        # compute gaussian kernel
        rs_sim = tf.reshape(sim, [sim_shape[0], sim_shape[1], sim_shape[2], 1])

        # compute person kernel
        person_rs_sim = tf.expand_dims(person_sim, -1)

        # compute Gaussian scores of each kernel
        mu = tf.reshape(self.mus, shape=[1, 1, self.n_bins])
        sigma = tf.reshape(self.sigmas, shape=[1, 1, self.n_bins])

        tmp = tf.exp(-tf.square(tf.subtract(rs_sim, mu)) / (tf.multiply(tf.square(sigma), 2)))

        # compute person score of each kernel

        
        person_tmp = tf.exp(-tf.square(tf.subtract(person_rs_sim, mu)) / (tf.multiply(tf.square(sigma), 2)))

        person_kde = tf.reduce_sum(person_tmp, [3])
        person_kde = tf.log(tf.maximum(person_kde, 1e-10)) * 0.01
        # [batch_size, paper_num, qlen, n_bins]
        
        # mask those non-existing words.
        # tmp = tmp * mask

        # sum up gaussian scores
        kde = tf.reduce_sum(tmp, [2])
        kde = tf.log(tf.maximum(kde, 1e-10)) * 0.01  # 0.01 used to scale down the data.
        # [batch_size, qlen, n_bins]

        # aggregated query terms
        # q_weights = [1, 1, 0, 0...]. Works as a query word mask.
        # Support query-term weigting if set to continous values (e.g. IDF).
        q_weights_shape = tf.shape(q_weights)
        q_weights_re = tf.reshape(q_weights, [q_weights_shape[0], q_weights_shape[1],1])
        aggregated_kde = tf.reduce_sum(kde * q_weights_re, [1])  # [batch, n_bins]

        person_kde_shape = tf.shape(person_kde)


        person_weights = tf.reshape(tf.tile(q_weights, [1, person_kde_shape[1]]), [person_kde_shape[0], person_kde_shape[1], person_kde_shape[2]])

        # [batch_size, paper_num, qlen, 1]
        person_weights = tf.expand_dims(person_weights, -1)

        # [batch_size, paper_num, n_bins]
        person_aggregated_kde = tf.reduce_sum(person_kde * person_weights, [2])

        attention_person_kde = self.AttentionLayer(person_aggregated_kde, self.n_bins, paper_num)

        total_kde = tf.concat([attention_person_kde, aggregated_kde], 1)

        # full-connected layer
        total_kde_f = tf.nn.relu(tf.matmul(total_kde, self.t_W) + self.t_b)

        return total_kde_f

    def get_interaction_vec_ins(self, q_embed, person_d_embed, q_weights):

        normalized_q_embed = tf.nn.l2_normalize(q_embed, axis=2)
        # [batch_size, paper_num, dlen, emb]
        normalize_person_d_emb = tf.nn.l2_normalize(person_d_embed, axis = 3)
        person_tmp = tf.transpose(normalize_person_d_emb, perm = [0, 1, 3, 2])
        
        # person sim mat [n_batch, paper_num, qlen, dlen]
        person_sim = tf.einsum('ijk,ipkl->ipjl',normalized_q_embed, person_tmp, name='person_similarity_matrix')
        person_sim_shape = tf.shape(person_sim)


        person_rs_sim = tf.expand_dims(person_sim, -1)
        # compute person score of each kernel
        mu = tf.reshape(self.mus, shape=[1, 1, self.n_bins])
        sigma = tf.reshape(self.sigmas, shape=[1, 1, self.n_bins])

        person_tmp = tf.exp(-tf.square(tf.subtract(person_rs_sim, mu)) / (tf.multiply(tf.square(sigma), 2)))

        person_kde = tf.reduce_sum(person_tmp, [3])
        person_kde = tf.log(tf.maximum(person_kde, 1e-10)) * 0.01
        # [batch_size, paper_num, qlen, n_bins]

        person_kde_shape = tf.shape(person_kde)


        person_weights = tf.reshape(tf.tile(q_weights, [1, person_kde_shape[1]]), [person_kde_shape[0], person_kde_shape[1], person_kde_shape[2]])

        # [batch_size, paper_num, qlen, 1]
        person_weights = tf.expand_dims(person_weights, -1)

        # [batch_size, paper_num, n_bins]
        person_aggregated_kde = tf.reduce_sum(person_kde * person_weights, [2])

        # attention_person_kde = self.AttentionLayer(person_aggregated_kde, self.n_bins, paper_num)

        return person_aggregated_kde

    def get_interaction_vec_bags(self, q_embed, d_embed, q_weights):

        normalized_q_embed = tf.nn.l2_normalize(q_embed, axis=2)
        normalized_d_embed = tf.nn.l2_normalize(d_embed, axis=2)
        tmp = tf.transpose(normalized_d_embed, perm=[0, 2, 1])


        # similarity matrix [n_batch, qlen, dlen]
        sim = tf.einsum('ijk,ikl->ijl',normalized_q_embed, tmp, name='similarity_matrix')
        sim_shape = tf.shape(sim)

        # compute gaussian kernel
        rs_sim = tf.reshape(sim, [sim_shape[0], sim_shape[1], sim_shape[2], 1])


        # compute Gaussian scores of each kernel
        mu = tf.reshape(self.mus, shape=[1, 1, self.n_bins])
        sigma = tf.reshape(self.sigmas, shape=[1, 1, self.n_bins])

        tmp = tf.exp(-tf.square(tf.subtract(rs_sim, mu)) / (tf.multiply(tf.square(sigma), 2)))

        # compute person score of each kernel

        
        # [batch_size, paper_num, qlen, n_bins]
        
        # mask those non-existing words.
        # tmp = tmp * mask

        # sum up gaussian scores
        kde = tf.reduce_sum(tmp, [2])
        kde = tf.log(tf.maximum(kde, 1e-10)) * 0.01  # 0.01 used to scale down the data.
        # [batch_size, qlen, n_bins]

        # aggregated query terms
        # q_weights = [1, 1, 0, 0...]. Works as a query word mask.
        # Support query-term weigting if set to continous values (e.g. IDF).
        q_weights_shape = tf.shape(q_weights)
        q_weights_re = tf.reshape(q_weights, [q_weights_shape[0], q_weights_shape[1],1])
        aggregated_kde = tf.reduce_sum(kde * q_weights_re, [1])  # [batch, n_bins]


        return aggregated_kde


    def new_model_bags(self, q_word, q_author, d_word, d_author, word_embeddings, author_embeddings, q_word_weight, q_author_weight):

        q_word_embed = tf.nn.embedding_lookup(word_embeddings, q_word)
        q_author_embed = tf.nn.embedding_lookup(author_embeddings, q_author)

        d_word_embed = tf.nn.embedding_lookup(word_embeddings, d_word)
        d_author_embed = tf.nn.embedding_lookup(author_embeddings, d_author)

        word_sim_vec = self.get_interaction_vec_bags(q_word_embed, d_word_embed, q_word_weight)
        author_sim_vec = self.get_interaction_vec_bags(q_author_embed, d_author_embed, q_author_weight)


        word_att = tf.tanh(tf.nn.bias_add(tf.matmul(word_sim_vec, self.word_W), self.word_b))
        author_att = tf.tanh(tf.nn.bias_add(tf.matmul(author_sim_vec, self.author_W), self.author_b))

        attention = tf.concat([word_att, author_att], 1)
        softmax = tf.nn.softmax(attention)
        a1, a2 = tf.split(softmax, num_or_size_splits=2, axis=1)
                #concat_hidden = tf.concat([a1 * name_cnn, a2 * aff_cnn, a3 * edu_cnn, a4 * pub_cnn], 1)

        merge_vec = tf.concat([a1 * word_sim_vec, a2 * author_sim_vec], 1)

        return merge_vec

    def new_model_ins(self, q_word, q_author, person_d_word, person_d_author, word_embeddings, author_embeddings, q_word_weight, q_author_weight, paper_num):
        q_word_embed = tf.nn.embedding_lookup(word_embeddings, q_word)
        q_author_embed = tf.nn.embedding_lookup(author_embeddings, q_author)

        person_d_word_embed = tf.nn.embedding_lookup(word_embeddings, person_d_word)
        person_d_author_embed = tf.nn.embedding_lookup(author_embeddings, person_d_author)

        person_word_sim_vec = self.get_interaction_vec_ins(q_word_embed, person_d_word_embed, q_word_weight)
        person_author_sim_vec = self.get_interaction_vec_ins(q_author_embed, person_d_author_embed, q_author_weight)

        person_word_att = tf.tanh(tf.nn.bias_add(tf.einsum('ijk,kl->ijl',person_word_sim_vec, self.word_W), self.word_b))
        person_author_att = tf.tanh(tf.nn.bias_add(tf.einsum('ijk,kl->ijl',person_author_sim_vec, self.author_W), self.author_b))

        attention = tf.concat([person_word_att, person_author_att], 2)

        softmax = tf.nn.softmax(attention, axis = -1)
        a1, a2 = tf.split(softmax, num_or_size_splits=2, axis= -1)

        merge_vec = tf.concat([a1 * person_word_sim_vec, a2 * person_author_sim_vec], 2)

        attention_person_vec = self.AttentionLayer(merge_vec, self.n_bins*2, paper_num)

        return attention_person_vec


    def _calcuate_matching_score(self, sim_vec):
        # layer1 = tf.nn.relu(tf.matmul(sim_vec, self.w_W1) + self.w_b1)
        # # layer2 = tf.nn.dropout(layer1, keep_prob= self.keep_prob)
        # sim = tf.matmul(layer1, self.w_W2) + self.w_b2
        layer1 = tf.nn.relu(tf.matmul(sim_vec, self.fc_w1) + self.fc_b1)
        layer1 = tf.nn.dropout(layer1, keep_prob= self.keep_prob)
        
        layer2 = tf.nn.relu(tf.matmul(layer1, self.fc_w2) + self.fc_b2)
        layer2 = tf.nn.dropout(layer2, keep_prob= self.keep_prob)
        
        layer3 = tf.nn.relu(tf.matmul(layer2, self.fc_w3) + self.fc_b3)
        layer3 = tf.nn.dropout(layer3, keep_prob= self.keep_prob)
        
        sim = tf.matmul(layer3, self.fc_w4) + self.fc_b4
        return sim



    def _create_loss(self):

        pos_sim_vec_bags = self.new_model_bags(self.train_words_q, self.train_authors_q, self.pos_train_words_d, self.pos_train_authors_d,
            self.word_embeddings, self.author_embeddings, self.train_words_q_weights, self.train_authors_q_weights)

        pos_sim_vec_ins = self.new_model_ins(self.train_words_q, self.train_authors_q, self.pos_train_person_words_d, self.pos_train_person_authors_d,
            self.word_embeddings, self.author_embeddings, self.train_words_q_weights, self.train_authors_q_weights, self.pos_paper_num)

        pos_sim_vec = tf.concat([pos_sim_vec_ins, pos_sim_vec_bags], 1)

        neg_sim_vec_bags = self.new_model_bags(self.train_words_q, self.train_authors_q, self.neg_train_words_d, self.neg_train_authors_d,
            self.word_embeddings, self.author_embeddings, self.train_words_q_weights, self.train_authors_q_weights)

        neg_sim_vec_ins = self.new_model_ins(self.train_words_q, self.train_authors_q, self.neg_train_person_words_d, self.neg_train_person_authors_d,
            self.word_embeddings, self.author_embeddings, self.train_words_q_weights, self.train_authors_q_weights, self.neg_paper_num)       


        neg_sim_vec = tf.concat([neg_sim_vec_ins, neg_sim_vec_bags], 1)

        

        # Learning-To-Rank layer. o is the final matching score.
        self.pos_score = self._calcuate_matching_score(pos_sim_vec)
        self.neg_score = self._calcuate_matching_score(neg_sim_vec)

        # self.loss = tf.reduce_mean(tf.maximum(0.0, 1 - pos_score + neg_score))
        self.tr_variables = tf.trainable_variables()
        self.l2_loss = 0
        for v in self.tr_variables:
            self.l2_loss += tf.nn.l2_loss(v)

        self.loss = tf.reduce_mean(tf.maximum(0.0, 1 - self.pos_score + self.neg_score)) + 0.01 * self.l2_loss
        # self.loss = tf.reduce_mean(tf.maximum(0.0, 1 - pos_score + neg_score))

        # predict

        test_sim_vec_bags = self.new_model_bags(self.test_words_q, self.test_authors_q, self.test_words_d, self.test_authors_d,
            self.word_embeddings, self.author_embeddings, self.test_words_q_weights, self.test_authors_q_weights)

        test_sim_vec_ins = self.new_model_ins(self.test_words_q, self.test_authors_q, self.test_person_words_d, self.test_person_authors_d,
            self.word_embeddings, self.author_embeddings, self.test_words_q_weights, self.test_authors_q_weights, self.test_paper_num)

        # test_str_vec = self.get_str_features(self.test_str_features)
        # test_sim_vec = tf.concat([test_sim_vec_ins, test_sim_vec_bags, test_str_vec], 1)
        test_sim_vec = tf.concat([test_sim_vec_ins, test_sim_vec_bags], 1)


        self.test_score = self._calcuate_matching_score(test_sim_vec)
        self.feature_vec = test_sim_vec




    def _create_optimizer(self):
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon).minimize(self.loss)
        self.fine_tune_optimizer = tf.train.AdamOptimizer(learning_rate=self.fine_tune_lr, epsilon=self.epsilon).minimize(self.loss)


    def _build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        logging.info("already build the computing graph...")

    def get_top_feature(self, test_authors_q, test_authors_q_weights, test_words_q, test_words_q_weights, 
        test_authors_d, test_words_d, test_person_authors_d, test_person_words_d, test_paper_num):



        score, fea_vec = self.sess.run([self.test_score, self.feature_vec],
            feed_dict = {self.test_words_q: np.array(test_words_q), self.test_words_q_weights: np.array(test_words_q_weights), 
            self.test_authors_q: np.array(test_authors_q), self.test_authors_q_weights: np.array(test_authors_q_weights),
            self.test_words_d: np.array(test_words_d),self.test_authors_d: np.array(test_authors_d),
            self.test_person_authors_d: np.array(test_person_authors_d),
            self.test_person_words_d: np.array(test_person_words_d),
            self.test_paper_num: np.array(test_paper_num),
            self.keep_prob : 1.0
            })

        return score, fea_vec

    def _predict_a_batch(self, test_authors_q, test_authors_q_weights, test_words_q, test_words_q_weights, 
        test_authors_d, test_words_d, test_person_authors_d, test_person_words_d, test_paper_num):

        return self.sess.run(self.test_score,
            feed_dict = {self.test_words_q: np.array(test_words_q), self.test_words_q_weights: np.array(test_words_q_weights), 
            self.test_authors_q: np.array(test_authors_q), self.test_authors_q_weights: np.array(test_authors_q_weights),
            self.test_words_d: np.array(test_words_d),self.test_authors_d: np.array(test_authors_d),
            self.test_person_authors_d: np.array(test_person_authors_d),
            self.test_person_words_d: np.array(test_person_words_d),
            self.test_paper_num: np.array(test_paper_num),
            self.keep_prob : 1.0
            })

 
    def predict(self, test_authors_q_list, test_authors_q_weights_list, test_words_q_list, test_words_q_weights_list, 
        test_authors_d_list, test_words_d_list, test_person_authors_d_list, test_person_words_d_list, test_paper_num_list):

        predictions_cat = np.empty([0, 1])
        # print("word batch size = ", len(test_words_q_list))
        # print("author batch size = ", len(test_authors_q_list))
        for batch_index in np.arange(len(test_words_q_list)):
            predictions = self._predict_a_batch(
                test_authors_q_list[batch_index], test_authors_q_weights_list[batch_index], test_words_q_list[batch_index], test_words_q_weights_list[batch_index], 
                test_authors_d_list[batch_index], test_words_d_list[batch_index], test_person_authors_d_list[batch_index], test_person_words_d_list[batch_index], test_paper_num_list[batch_index])
            predictions_cat = np.concatenate((predictions_cat, np.reshape(predictions,[-1,1])))
        return predictions_cat

    def predict_cl(self, test_authors_q_list, test_authors_q_weights_list, test_words_q_list, test_words_q_weights_list, 
        test_authors_d_list, test_words_d_list, test_person_authors_d_list, test_person_words_d_list, test_paper_num_list, test_id_list):

        predictions_cat = np.empty([0, 1])
        ids_cat = []
        # print("word batch size = ", len(test_words_q_list))
        # print("author batch size = ", len(test_authors_q_list))
        for batch_index in np.arange(len(test_words_q_list)):
            predictions = self._predict_a_batch(
                test_authors_q_list[batch_index], test_authors_q_weights_list[batch_index], test_words_q_list[batch_index], test_words_q_weights_list[batch_index], 
                test_authors_d_list[batch_index], test_words_d_list[batch_index], test_person_authors_d_list[batch_index], test_person_words_d_list[batch_index], test_paper_num_list[batch_index])
            predictions_cat = np.concatenate((predictions_cat, np.reshape(predictions,[-1,1])))
            ids_cat.extend(test_id_list[batch_index])
        return predictions_cat, ids_cat

    def train(self, train_words_q, train_words_q_weights, 
            pos_train_words_d, pos_train_person_words_d, pos_papers,
            neg_train_words_d, neg_train_person_words_d, neg_papers,
            train_authors_q, train_authors_q_weights, 
            pos_train_authors_d, pos_train_person_authors_d,
            neg_train_authors_d, neg_train_person_authors_d):


        return self.sess.run([self.loss, self.l2_loss, self.optimizer, self.pos_score, self.neg_score],
            feed_dict = {self.train_words_q: np.array(train_words_q), self.train_words_q_weights: np.array(train_words_q_weights), 
            self.pos_train_words_d: np.array(pos_train_words_d), self.pos_train_person_words_d: np.array(pos_train_person_words_d),
            self.neg_train_words_d: np.array(neg_train_words_d), self.neg_train_person_words_d: np.array(neg_train_person_words_d),
            self.train_authors_q: np.array(train_authors_q), self.train_authors_q_weights: np.array(train_authors_q_weights),
            self.pos_train_authors_d: np.array(pos_train_authors_d),self.pos_train_person_authors_d: np.array(pos_train_person_authors_d),
            self.neg_train_authors_d: np.array(neg_train_authors_d),self.neg_train_person_authors_d: np.array(neg_train_person_authors_d),  
            self.pos_paper_num: np.array(pos_papers), self.neg_paper_num: np.array(neg_papers),
            self.keep_prob : 1
            })


    def fine_tune(self, train_words_q, train_words_q_weights, 
            pos_train_words_d, pos_train_person_words_d, pos_papers,
            neg_train_words_d, neg_train_person_words_d, neg_papers,
            train_authors_q, train_authors_q_weights, 
            pos_train_authors_d, pos_train_person_authors_d,
            neg_train_authors_d, neg_train_person_authors_d):


        return self.sess.run([self.loss, self.fine_tune_optimizer],
            feed_dict = {self.train_words_q: np.array(train_words_q), self.train_words_q_weights: np.array(train_words_q_weights), 
            self.pos_train_words_d: np.array(pos_train_words_d), self.pos_train_person_words_d: np.array(pos_train_person_words_d),
            self.neg_train_words_d: np.array(neg_train_words_d), self.neg_train_person_words_d: np.array(neg_train_person_words_d),
            self.train_authors_q: np.array(train_authors_q), self.train_authors_q_weights: np.array(train_authors_q_weights),
            self.pos_train_authors_d: np.array(pos_train_authors_d),self.pos_train_person_authors_d: np.array(pos_train_person_authors_d),
            self.neg_train_authors_d: np.array(neg_train_authors_d),self.neg_train_person_authors_d: np.array(neg_train_person_authors_d),  
            self.pos_paper_num: np.array(pos_papers), self.neg_paper_num: np.array(neg_papers),
            self.keep_prob : 1
            })


def print_variable(sess):
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print(k,v)

if __name__=='__main__':

    #Save checkpoints
    save_path = './saved_ranking_model/'
    print("begin load training and test data ......")
    

    train_data = gen_train_data_profile.generate_train_data("train_author_pub_index_profile.json", "train_author_pub_index_test.json", settings.TRAIN_SCALE, settings.NEG_SAMPLE)
    
    test_data = gen_train_data_profile.generate_test_data("test_author_pub_index_profile.json", "test_author_pub_index_test.json" , settings.TEST_SCALE, settings.TEST_SAMPLE)

    TR_neg_person_author_ids, TR_neg_person_word_ids, TR_neg_per_person_author_ids, TR_neg_per_person_word_ids, TR_neg_person_papers,\
    TR_new_pos_person_author_ids, TR_new_pos_person_word_ids, TR_new_pos_per_person_author_ids, TR_new_pos_per_person_word_ids,TR_new_pos_person_papers,\
    TR_new_paper_author_ids, TR_new_paper_author_idfs, TR_new_paper_word_ids, TR_new_paper_word_idfs,\
    = train_data[0],train_data[1],train_data[2],train_data[3],train_data[4],\
    train_data[5],train_data[6],train_data[7],train_data[8],train_data[9],\
    train_data[10],train_data[11], train_data[12],train_data[13]

 

    TE_person_author_ids, TE_person_word_ids, TE_per_person_author_ids, TE_per_person_word_ids, TE_person_papers,\
    TE_new_paper_author_ids, TE_new_paper_author_idfs, TE_new_paper_word_ids, TE_new_paper_word_idfs, \
    = test_data[0],test_data[1],test_data[2],test_data[3],test_data[4],\
    test_data[5],test_data[6],test_data[7],test_data[8]
    
    # exit()
    
    batch_num = len(TR_neg_person_author_ids)
    print("batch_num: ", batch_num)
    batch_indexes = np.arange(batch_num)
    print("Author #train_batch=%d, #test_batch=%d" %(len(TR_new_pos_person_author_ids), len(TE_person_author_ids)))
    print("Word #train_batch=%d, #test_batch=%d" %(len(TR_neg_per_person_word_ids), len(TE_person_word_ids)))


    print_logging("begin training global triplet model ......")
    tf_config = tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model = GlobalTripletModel(sess)
        sess.run(tf.global_variables_initializer())
        hit_max = 0
        #train by epoch
        for epoch_count in range(settings.EPOCHES):


            # np.random.shuffle(batch_indexes)
            train_begin = time()
            avg_train_loss = 0
            avg_l2_loss = 0

            random.shuffle(batch_indexes)
            for batch_index in batch_indexes:
                # print(batch_index)
                # train_loss, _= model.train(
                    # TR_person_word_id_list[batch_index], TR_paper_word_id_list[batch_index], TR_person_word_idf_list[batch_index],TR_person_author_id_list[batch_index], TR_paper_author_id_list[batch_index], TR_person_author_idf_list[batch_index],TR_labels[batch_index],\
                    # TR_per_person_author_ids[batch_index], TR_per_person_author_idfs[batch_index], TR_per_person_word_ids[batch_index], TR_per_person_word_idfs[batch_index], TR_per_person_paper_num[batch_index])
                train_loss, l2_loss, _ , pos_score, neg_score = model.train(TR_new_paper_word_ids[batch_index], TR_new_paper_word_idfs[batch_index], \
                    TR_new_pos_person_word_ids[batch_index], TR_new_pos_per_person_word_ids[batch_index], TR_new_pos_person_papers[batch_index], \
                    TR_neg_person_word_ids[batch_index], TR_neg_per_person_word_ids[batch_index], TR_neg_person_papers[batch_index],\
                    TR_new_paper_author_ids[batch_index], TR_new_paper_author_idfs[batch_index],\
                    TR_new_pos_person_author_ids[batch_index], TR_new_pos_per_person_author_ids[batch_index],\
                    TR_neg_person_author_ids[batch_index], TR_neg_per_person_author_ids[batch_index])
                # print("batch: {} = {}".format(batch_index, train_loss))
                # print("pos:{} neg:{}".format(pos_score, neg_score))
                
                avg_train_loss += train_loss
                avg_l2_loss += l2_loss
            avg_train_loss /= batch_num
            avg_l2_loss /= batch_num
            train_time = time() - train_begin
            print("Epoch %d: train_loss = %.4f [%.1fs]" % (epoch_count, avg_train_loss, train_time))
            # print(avg_l2_loss)

            if epoch_count % settings.VERBOSE == 0:
                # print("test")
                # print_variable(sess)
                test_begin = time()
                # predictions, test_loss = model.predict(TE_person_word_id_list, TE_paper_word_id_list, TE_person_word_idf_list, 
                #     TE_person_author_id_list, TE_paper_author_id_list, TE_person_author_idf_list, TE_labels)

                predictions = model.predict(
                        TE_new_paper_author_ids, TE_new_paper_author_idfs, TE_new_paper_word_ids, TE_new_paper_word_idfs,\
                        TE_person_author_ids, TE_person_word_ids, TE_per_person_author_ids, TE_per_person_word_ids, TE_person_papers)
                # print("ed")
                # print(predictions)


                top_k, ratio_top_k, mrr = eval_utils.eval_hit(predictions)
                print("hits@{} = {} mrr: {} test_time: {}".format(top_k, ratio_top_k, mrr, round(time()-test_begin, 2)))

                if(ratio_top_k[0] > hit_max):
                    model.saver.save(sess, save_path + 'embedding_model.ckpt', global_step=epoch_count)
                    hit_max = ratio_top_k[0]
