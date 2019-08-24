from os.path import join
import sys
sys.path.append("..")
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import multiprocessing as mp
import numpy as np 
import tensorflow as tf
from utils import eval_utils
from utils import data_utils
from utils import settings
from collections import defaultdict
from global_ import gen_train_data
import logging
from time import time
from time import strftime
from time import localtime
from global_.model_base import BaseNN
np.set_printoptions(suppress=True)
from traitlets.config import Configurable
from traitlets import (
    Int,
    Float,
    Bool,
    Unicode,
)
import gc

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
        self.saver = tf.train.Saver(max_to_keep=5)

    def _prepare_parameter(self, sess):
        self.learning_rate = settings.LEARNING_RATE
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

        self.pos_str_features = tf.placeholder(tf.float32, shape=[None, None], name='pos_str_feature') #ins_num * feature_size
        self.neg_str_features = tf.placeholder(tf.float32, shape=[None, None], name='neg_str_feature') #ins_num * feature_size
        self.test_str_features = tf.placeholder(tf.float32, shape=[None, None], name='test_str_feature') #ins_num * feature_size

        self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

    def _create_variables(self):
        self.c1 = tf.Variable(tf.constant(self.word_emb, dtype='float32', shape=[self.word_num, settings.EMB_DIM]),name='c1',trainable=True)
        self.c2 = tf.constant(0.0, tf.float32, [1, settings.EMB_DIM], name='c2')
        self.word_embeddings = tf.concat([self.c1, self.c2], 0, name='word_embedding')

        self.a1 = tf.Variable(tf.constant(self.author_emb, dtype='float32', shape=[self.author_num, settings.EMB_DIM]),name='a1',trainable=True)
        self.a2 = tf.constant(0.0, tf.float32, [1, settings.EMB_DIM], name='a2')
        self.author_embeddings = tf.concat([self.a1, self.a2], 0, name='author_embedding')

        # parameters for total
        # self.t_W2 = self.weight_variable([self.weight_size, 2], "w_w2")
        # self.t_b2 = tf.Variable(tf.constant(0.1, shape=[2]), name="w_b2")

        self.t_W = self.weight_variable([self.n_bins * 2, self.n_bins], "t_w")
        self.t_b = tf.Variable(tf.zeros([self.n_bins]), name='t_b')

        self.m_W = self.weight_variable([self.n_bins, 1], name = "m_w")
        self.m_b = tf.Variable(tf.zeros([1]), name='m_b')
        
        self.author_W = self.weight_variable([self.n_bins, 1], name = "author_w")
        self.author_b = tf.Variable(tf.zeros([1]), name = "author_b")

        self.word_W = self.weight_variable([self.n_bins, 1], name = "word_w")
        self.word_b = tf.Variable(tf.zeros([1]), name = "word_b")
        # parameters for words
        self.w_W2 = self.weight_variable([self.weight_size, 1], "w_w2")
        self.w_b2 = tf.Variable(tf.constant(0.1, shape=[1]), name="w_b2")

        self.w_W1 = self.weight_variable([self.n_bins *2* 2 *2, self.weight_size], "w_w1")
        self.w_b1 = tf.Variable(tf.zeros([self.weight_size]), name='w_b1')


        self.s_W = self.weight_variable([17, self.n_bins *2* 2], "s_w")
        self.s_b = tf.Variable(tf.zeros([self.n_bins *2* 2]), name='s_b')
        # self.w_W2 = self.weight_variable([self.weight_size, 1], "w_w2")
        # self.w_b2 = tf.Variable(tf.constant(0.1, shape=[1]), name="w_b2")

        # self.w_W1 = self.weight_variable([self.n_bins * 2, self.weight_size], "w_w1")
        # self.w_b1 = tf.Variable(tf.zeros([self.weight_size]), name='w_b1')

        # parameters for authors
        # self.a_W2 = self.weight_variable([self.weight_size, 2], "a_w2")
        # self.a_b2 = tf.Variable(tf.constant(0.1, shape=[2]), name="a_b2")

        # self.a_W1 = self.weight_variable([self.n_bins* 2, self.weight_size], "a_w1")
        # self.a_b1 = tf.Variable(tf.zeros([self.weight_size]), name='a_b1')

        # weight for paper_attention
        # self.p_w = self.weight_variable([self.n_bins, 1], "p_w")
        # self.p_b = tf.Variable(tf.constant(0.1, shape=[1]), name='p_b')

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
        
        # uni
        uni_filter_shape = [1, 100, 128]
        self.uni_conv_W = tf.Variable(tf.truncated_normal(uni_filter_shape, stddev=0.1), name="uni_conv_W")
        self.uni_conv_b = tf.Variable(tf.constant(0.1, shape=[128]), name="uni_conv_b")
        
        # bi
        bi_filter_shape = [2, 100, 128]
        self.bi_conv_W = tf.Variable(tf.truncated_normal(bi_filter_shape, stddev=0.1), name="bi_conv_W")
        self.bi_conv_b = tf.Variable(tf.constant(0.1, shape=[128]), name="bi_conv_b")

        # tri
        tri_filter_shape = [3, 100, 128]
        self.tri_conv_W = tf.Variable(tf.truncated_normal(tri_filter_shape, stddev=0.1), name="tri_conv_W")
        self.tri_conv_b = tf.Variable(tf.constant(0.1, shape=[128]), name="tri_conv_b")


        # ins_filter_shape = [1, 1, 100, 100]
        # self.ins_W = tf.Variable(tf.truncated_normal(ins_filter_shape, stddev=0.1), name="ins_W")
        # self.ins_b = tf.Variable(tf.constant(0.1, shape=[100]), name="ins_b")

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

  
    def uni_cnn(self, inputs, dropout_keep_prob):

        # filter_sizes=[2,3]
        # self.embedded_chars=inputs
        pooled_outputs = []

        conv = tf.nn.conv1d(
            inputs,
            self.uni_conv_W,
            stride=1,
            padding="SAME",
            name="uni_conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, self.uni_conv_b))
        # Add dropout

        # self.h_drop = tf.nn.dropout(h, dropout_keep_prob)

        return h

    def bi_cnn(self, inputs, dropout_keep_prob):

        # filter_sizes=[2,3]
        # self.embedded_chars=inputs
        pooled_outputs = []

        conv = tf.nn.conv1d(
            inputs,
            self.bi_conv_W,
            stride=1,
            padding="SAME",
            name="bi_conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, self.bi_conv_b))
        # Add dropout

        # self.h_drop = tf.nn.dropout(h, dropout_keep_prob)

        return h

    def tri_cnn(self, inputs, dropout_keep_prob):

        # filter_sizes=[2,3]
        # self.embedded_chars=inputs
        pooled_outputs = []

        conv = tf.nn.conv1d(
            inputs,
            self.tri_conv_W,
            stride=1,
            padding="SAME",
            name="tri_conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, self.tri_conv_b))
        # Add dropout

        # self.h_drop = tf.nn.dropout(h, dropout_keep_prob)

        return h

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

        # ori attention
        # person_att = tf.einsum('ijk,kl->ijl', person_aggregated_kde, self.p_w, name='person_att')
        # person_att = person_att + self.p_b
        # person_att = tf.squeeze(person_att, -1)
        # mask_mat = tf.sequence_mask(paper_num, dtype = tf.float32)

        # # mask_person_att = person_att * mask_mat
        # # y = tf.reduce_sum(tf.exp(self.person_att) * mask_mat, -1)


        # softmax = tf.exp(person_att) * mask_mat / tf.expand_dims(tf.reduce_sum(tf.exp(person_att) * mask_mat, -1), -1)

        # # softmax = tf.nn.softmax(mask_person_att, -1)

        # #[batch_size, nbins]
        # attention_person_kde = tf.reduce_sum(person_aggregated_kde * tf.expand_dims(softmax, -1), [1]) 

        # ending

        # mask_mat = tf.one_hot()
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


    def model(self, inputs_q, inputs_d, q_weights, embeddings, person_d, paper_num):
        """
        The pointwise model graph
        :param inputs_q: input queries. [nbatch, qlen]  -- represents a paper
        :param inputs_d: input documents. [nbatch, dlen] -- represents a person
        :param mask: a binary mask. [nbatch, qlen, dlen]
        :param q_weights: query term weigths. Set to binary in the paper.
        :return: return the predicted score for each <query, document> in the batch
        """
        # look up embeddings for each term. [nbatch, qlen, emb_dim]
        q_embed = tf.nn.embedding_lookup(embeddings, inputs_q, name='qemb')
        d_embed = tf.nn.embedding_lookup(embeddings, inputs_d, name='demb')

        # q_embed_uni = self.uni_cnn(q_embed, self.keep_prob)
        # q_embed_bi = self.bi_cnn(q_embed, self.keep_prob)
        # q_embed_tri = self.tri_cnn(q_embed, self.keep_prob)

        # d_embed_uni = self.uni_cnn(d_embed, self.keep_prob)
        # d_embed_bi = self.bi_cnn(d_embed, self.keep_prob)
        # d_embed_tri = self.tri_cnn(d_embed, self.keep_prob)


        person_d_embed = tf.nn.embedding_lookup(embeddings, person_d, name = 'person_demb')
        
        total_kde_a = self.get_interaction_vec(q_embed, d_embed, person_d_embed, q_weights, paper_num)

        # person_d_embed_shape = tf.shape(person_d_embed)
        
        # # # [batch_size, paper_num, dlen, emb]
        # person_d_embed_re = tf.reshape(person_d_embed, shape=[person_d_embed_shape[0]*person_d_embed_shape[1], person_d_embed_shape[2], person_d_embed_shape[3]])

        # person_d_embed_uni = self.uni_cnn(person_d_embed_re, self.keep_prob)

        # person_d_embed_uni = tf.reshape(person_d_embed_uni, shape=[person_d_embed_shape[0], person_d_embed_shape[1], person_d_embed_shape[2], self.filter_num])

        # person_d_embed_bi = self.bi_cnn(person_d_embed_re, self.keep_prob)

        # person_d_embed_bi = tf.reshape(person_d_embed_bi, shape=[person_d_embed_shape[0], person_d_embed_shape[1], person_d_embed_shape[2], self.filter_num])

        # person_d_embed_tri = self.tri_cnn(person_d_embed_re, self.keep_prob)

        # person_d_embed_tri = tf.reshape(person_d_embed_tri, shape=[person_d_embed_shape[0], person_d_embed_shape[1], person_d_embed_shape[2], self.filter_num])



        # total_kde_uni_uni = self.get_interaction_vec(q_embed_uni, d_embed_uni, person_d_embed_uni, q_weights, paper_num)
        # total_kde_uni_bi = self.get_interaction_vec(q_embed_uni, d_embed_bi, person_d_embed_bi, q_weights, paper_num)
        # total_kde_uni_tri = self.get_interaction_vec(q_embed_uni, d_embed_tri, person_d_embed_tri, q_weights, paper_num)

        # total_kde_bi_uni = self.get_interaction_vec(q_embed_bi, d_embed_uni, person_d_embed_uni, q_weights, paper_num)
        # total_kde_bi_bi = self.get_interaction_vec(q_embed_bi, d_embed_bi, person_d_embed_bi, q_weights, paper_num)
        # total_kde_bi_tri = self.get_interaction_vec(q_embed_bi, d_embed_tri, person_d_embed_tri, q_weights, paper_num)

        # total_kde_tri_uni = self.get_interaction_vec(q_embed_tri, d_embed_uni, person_d_embed_uni, q_weights, paper_num)
        # total_kde_tri_bi = self.get_interaction_vec(q_embed_tri, d_embed_bi, person_d_embed_bi, q_weights, paper_num)
        # total_kde_tri_tri = self.get_interaction_vec(q_embed_tri, d_embed_tri, person_d_embed_tri, q_weights, paper_num)

        # total_kde_a = tf.concat([total_kde_uni_uni, total_kde_uni_bi, total_kde_uni_tri, total_kde_bi_uni, total_kde_bi_bi, total_kde_bi_tri, total_kde_tri_uni, total_kde_tri_bi, total_kde_tri_tri], 1) 

        return total_kde_a
        # return attention_person_kde
        # return attention_person_kde

    # def _calcuate_total_sim(self, sim_vec):
    #     layer1 = tf.nn.relu(tf.matmul(sim_vec, self.t_W1) + self.t_b1)
    #     # layer2 = tf.nn.dropout(layer1, keep_prob=0.5)
    #     sim = tf.matmul(layer1, self.t_W2) + self.t_b2
    #     # fc_1 = tf.contrib.layers.fully_connected(sim_vec, 11, activation_fn=tf.nn.relu)
    #     # # fc_2 = tf.nn.dropout(fc_1, 0.5)
    #     # sim = tf.contrib.layers.fully_connected(fc_1, 2, activation_fn=None)

    #     return sim
    def _calcuate_matching_score(self, sim_vec):
        layer1 = tf.nn.relu(tf.matmul(sim_vec, self.w_W1) + self.w_b1)
        # layer2 = tf.nn.dropout(layer1, keep_prob= self.keep_prob)
        sim = tf.matmul(layer1, self.w_W2) + self.w_b2
        return sim

    # def _calcuate_author_sim(self, sim_vec):
    #     layer1 = tf.nn.relu(tf.matmul(sim_vec, self.a_W1) + self.a_b1)
    #     # layer2 = tf.nn.dropout(layer1, keep_prob= self.keep_prob)
    #     sim = tf.matmul(layer1, self.a_W2) + self.a_b2
    #     return sim
    def get_str_features(self, str_features):

        return tf.nn.relu(tf.matmul(str_features, self.s_W) + self.s_b)


    def _create_loss(self):

        pos_sim_vec_bags = self.new_model_bags(self.train_words_q, self.train_authors_q, self.pos_train_words_d, self.pos_train_authors_d,
            self.word_embeddings, self.author_embeddings, self.train_words_q_weights, self.train_authors_q_weights)

        pos_sim_vec_ins = self.new_model_ins(self.train_words_q, self.train_authors_q, self.pos_train_person_words_d, self.pos_train_person_authors_d,
            self.word_embeddings, self.author_embeddings, self.train_words_q_weights, self.train_authors_q_weights, self.pos_paper_num)

        pos_str_vec = self.get_str_features(self.pos_str_features)

        pos_sim_vec = tf.concat([pos_sim_vec_ins, pos_sim_vec_bags, pos_str_vec], 1)


        neg_sim_vec_bags = self.new_model_bags(self.train_words_q, self.train_authors_q, self.neg_train_words_d, self.neg_train_authors_d,
            self.word_embeddings, self.author_embeddings, self.train_words_q_weights, self.train_authors_q_weights)

        neg_sim_vec_ins = self.new_model_ins(self.train_words_q, self.train_authors_q, self.neg_train_person_words_d, self.neg_train_person_authors_d,
            self.word_embeddings, self.author_embeddings, self.train_words_q_weights, self.train_authors_q_weights, self.neg_paper_num)       

        neg_str_vec = self.get_str_features(self.neg_str_features)
        neg_sim_vec = tf.concat([neg_sim_vec_ins, neg_sim_vec_bags, neg_str_vec], 1)



        
        # # POS SCORE
        # pos_word_sim_vec = self.model(self.train_words_q, self.pos_train_words_d, self.train_words_q_weights, self.word_embeddings, self.pos_train_person_words_d, self.pos_paper_num)
        # pos_author_sim_vec = self.model(self.train_authors_q, self.pos_train_authors_d, self.train_authors_q_weights, self.author_embeddings, self.pos_train_person_authors_d, self.pos_paper_num)
        # # pos_sim_vec = tf.concat([pos_word_sim_vec, pos_author_sim_vec], 1) 
        # # pos_sim_vec = pos_author_sim_vec
        # pos_sim_vec = pos_word_sim_vec
        # # pos_sim_vec = self.get_merge_vec(pos_word_sim_vec, pos_author_sim_vec)

        # # NEG SCORE
        # neg_word_sim_vec = self.model(self.train_words_q, self.neg_train_words_d, self.train_words_q_weights, self.word_embeddings, self.neg_train_person_words_d, self.neg_paper_num)
        # neg_author_sim_vec = self.model(self.train_authors_q, self.neg_train_authors_d, self.train_authors_q_weights, self.author_embeddings, self.neg_train_person_authors_d, self.neg_paper_num)
        # # neg_sim_vec = tf.concat([neg_word_sim_vec, neg_author_sim_vec], 1)
        # # neg_sim_vec = neg_author_sim_vec
        # neg_sim_vec = neg_word_sim_vec
        # # neg_sim_vec = self.get_merge_vec(neg_word_sim_vec, neg_author_sim_vec)

        # Learning-To-Rank layer. o is the final matching score.
        pos_score = self._calcuate_matching_score(pos_sim_vec)
        neg_score = self._calcuate_matching_score(neg_sim_vec)

        self.loss = tf.reduce_mean(tf.maximum(0.0, 1 - pos_score + neg_score))


        # predict

        test_sim_vec_bags = self.new_model_bags(self.test_words_q, self.test_authors_q, self.test_words_d, self.test_authors_d,
            self.word_embeddings, self.author_embeddings, self.test_words_q_weights, self.test_authors_q_weights)

        test_sim_vec_ins = self.new_model_ins(self.test_words_q, self.test_authors_q, self.test_person_words_d, self.test_person_authors_d,
            self.word_embeddings, self.author_embeddings, self.test_words_q_weights, self.test_authors_q_weights, self.test_paper_num)

        test_str_vec = self.get_str_features(self.test_str_features)
        test_sim_vec = tf.concat([test_sim_vec_ins, test_sim_vec_bags, test_str_vec], 1)


        # test_word_sim_vec = self.model(self.test_words_q, self.test_words_d, self.test_words_q_weights, self.word_embeddings, self.test_person_words_d, self.test_paper_num)
        # test_author_sim_vec = self.model(self.test_authors_q, self.test_authors_d, self.test_authors_q_weights, self.author_embeddings, self.test_person_authors_d, self.test_paper_num)
        # # test_sim_vec = tf.concat([test_word_sim_vec, test_author_sim_vec], 1) 
        # # test_sim_vec = test_author_sim_vec
        # test_sim_vec = test_word_sim_vec
        # test_sim_vec = self.get_merge_vec(test_word_sim_vec, test_author_sim_vec)

        self.test_score = self._calcuate_matching_score(test_sim_vec)
        self.feature_vec = test_sim_vec



        #_01_loss

        input_pos_score = self._calcuate_matching_score(self.input_pos_feature)
        # neg_score = self._calcuate_matching_score(neg_sim_vec)

        self._01_loss = tf.reduce_mean(tf.maximum(0.0, 1 - input_pos_score + neg_score))

        #_10_loss
        input_neg_score = self._calcuate_matching_score(self.input_neg_feature)
        # neg_score = self._calcuate_matching_score(neg_sim_vec)

        self._10_loss = tf.reduce_mean(tf.maximum(0.0, 1 - pos_score + input_neg_score))






        # Concentrate
        #total_sim_vec = tf.concat([author_sim_vec, word_sim_vec], 1) 


        #sim = self._calcuate_total_sim(total_sim_vec)



        #sim = word_sim * self.w + author_sim * (1- self.w)
        #self.predictions = tf.argmax(sim, 1, name="predictions")
        #self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sim, labels=self.train_y))


    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon).minimize(self.loss)
        self._01_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate/10, epsilon=self.epsilon).minimize(self._01_loss)
        self._10_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate/10, epsilon=self.epsilon).minimize(self._10_loss)


    def _build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()
        self._create_optimizer()
        logging.info("already build the computing graph...")

    def get_top_feature(self, test_authors_q, test_authors_q_weights, test_words_q, test_words_q_weights, 
        test_authors_d, test_words_d, test_person_authors_d, test_person_words_d, test_paper_num, str_featuresss):



        score, fea_vec = self.sess.run([self.test_score, self.feature_vec],
            feed_dict = {self.test_words_q: np.array(test_words_q), self.test_words_q_weights: np.array(test_words_q_weights), 
            self.test_authors_q: np.array(test_authors_q), self.test_authors_q_weights: np.array(test_authors_q_weights),
            self.test_words_d: np.array(test_words_d),self.test_authors_d: np.array(test_authors_d),
            self.test_person_authors_d: np.array(test_person_authors_d),
            self.test_person_words_d: np.array(test_person_words_d),
            self.test_paper_num: np.array(test_paper_num),
            self.test_str_features: np.array(str_featuresss),
            self.keep_prob : 1.0
            })

        return score, fea_vec

    def _predict_a_batch(self, test_authors_q, test_authors_q_weights, test_words_q, test_words_q_weights, 
        test_authors_d, test_words_d, test_person_authors_d, test_person_words_d, test_paper_num, str_featuresss):

        # test_paper_max = np.max(np.array(test_paper_num))

        # if (np.array(test_person_authors_d).shape[1] != test_paper_max):
        #     print("test not match:{}/{}".format(np.array(test_person_authors_d).shape[1], test_paper_max))
        #     exit()

        return self.sess.run(self.test_score,
            feed_dict = {self.test_words_q: np.array(test_words_q), self.test_words_q_weights: np.array(test_words_q_weights), 
            self.test_authors_q: np.array(test_authors_q), self.test_authors_q_weights: np.array(test_authors_q_weights),
            self.test_words_d: np.array(test_words_d),self.test_authors_d: np.array(test_authors_d),
            self.test_person_authors_d: np.array(test_person_authors_d),
            self.test_person_words_d: np.array(test_person_words_d),
            self.test_paper_num: np.array(test_paper_num),
            self.test_str_features : np.array(str_featuresss),
            self.keep_prob : 1.0
            })

 
    def predict(self, test_authors_q_list, test_authors_q_weights_list, test_words_q_list, test_words_q_weights_list, 
        test_authors_d_list, test_words_d_list, test_person_authors_d_list, test_person_words_d_list, test_paper_num_list, str_features_list):

        predictions_cat = np.empty([0, 1])
        # print("word batch size = ", len(test_words_q_list))
        # print("author batch size = ", len(test_authors_q_list))
        for batch_index in np.arange(len(test_words_q_list)):
            predictions = self._predict_a_batch(
                test_authors_q_list[batch_index], test_authors_q_weights_list[batch_index], test_words_q_list[batch_index], test_words_q_weights_list[batch_index], 
                test_authors_d_list[batch_index], test_words_d_list[batch_index], test_person_authors_d_list[batch_index], test_person_words_d_list[batch_index], test_paper_num_list[batch_index], 
                str_features_list[batch_index])
            predictions_cat = np.concatenate((predictions_cat, np.reshape(predictions,[-1,1])))
        return predictions_cat


    def train(self, train_words_q, train_words_q_weights, 
            pos_train_words_d, pos_train_person_words_d, pos_papers,
            neg_train_words_d, neg_train_person_words_d, neg_papers,
            train_authors_q, train_authors_q_weights, 
            pos_train_authors_d, pos_train_person_authors_d,
            neg_train_authors_d, neg_train_person_authors_d, pos_str_features, neg_str_features):

        # print(np.array(train_authors_q).shape, np.array(pos_train_authors_d).shape)
        # print(np.array(neg_train_person_authors_d).shape, np.array(neg_train_person_words_d).shape, np.array(neg_papers))
        # exit()
        # print(np.array(train_person_words_d).shape)
        # pos_paper_max = np.max(np.array(pos_papers))
        # neg_paper_max = np.max(np.array(neg_papers))

        # if (np.array(pos_train_person_authors_d).shape[1] != pos_paper_max):
        #     print("not match:{}/{}".format(np.array(pos_train_person_authors_d).shape[1], pos_paper_max))
        #     exit()

        # if (np.array(neg_train_person_words_d).shape[1] != neg_paper_max):
        #     print("not match:{}/{}".format(np.array(neg_train_person_words_d).shape[1], neg_paper_max))
        #     exit()


        return self.sess.run([self.loss, self.optimizer],
            feed_dict = {self.train_words_q: np.array(train_words_q), self.train_words_q_weights: np.array(train_words_q_weights), 
            self.pos_train_words_d: np.array(pos_train_words_d), self.pos_train_person_words_d: np.array(pos_train_person_words_d),
            self.neg_train_words_d: np.array(neg_train_words_d), self.neg_train_person_words_d: np.array(neg_train_person_words_d),
            self.train_authors_q: np.array(train_authors_q), self.train_authors_q_weights: np.array(train_authors_q_weights),
            self.pos_train_authors_d: np.array(pos_train_authors_d),self.pos_train_person_authors_d: np.array(pos_train_person_authors_d),
            self.neg_train_authors_d: np.array(neg_train_authors_d),self.neg_train_person_authors_d: np.array(neg_train_person_authors_d),  
            self.pos_paper_num: np.array(pos_papers), self.neg_paper_num: np.array(neg_papers),
            self.pos_str_features: np.array(pos_str_features), self.neg_str_features: np.array(neg_str_features),
            self.keep_prob : 1
            })



    def fine_tune_01(self, _01re_pos_feature, _01re_new_paper_word_ids, _01re_new_paper_word_idfs, _01re_new_paper_author_ids, _01re_new_paper_author_idfs,
        _01re_new_neg_person_word_ids, _01re_new_neg_per_person_word_ids, _01re_new_neg_person_papers, _01re_new_neg_person_author_ids, _01re_new_neg_per_person_author_ids):
        # print("neg_word_id:", np.array(_01re_new_neg_person_word_ids).shape)
        # print("per_neg_word_id:", np.array(_01re_new_neg_per_person_word_ids).shape)
        # print("person papers:", np.array(_01re_new_neg_person_papers).shape)
        # print("neg_author_id: ", np.array(_01re_new_neg_person_author_ids).shape)
        # print("per_neg_author_id:",np.array(_01re_new_neg_per_person_author_ids).shape)
        # print(_01re_new_neg_person_papers)

        loss, _ = self.sess.run([self._01_loss, self._01_optimizer], 
            feed_dict = {self.input_pos_feature: np.array(_01re_pos_feature),
                        self.train_words_q: np.array(_01re_new_paper_word_ids), self.train_words_q_weights: np.array(_01re_new_paper_word_idfs),
                        self.train_authors_q: np.array(_01re_new_paper_author_ids), self.train_authors_q_weights: np.array(_01re_new_paper_author_idfs),
                        self.neg_train_words_d: np.array(_01re_new_neg_person_word_ids), self.neg_train_person_words_d: np.array(_01re_new_neg_per_person_word_ids),
                        self.neg_train_authors_d: np.array(_01re_new_neg_person_author_ids),self.neg_train_person_authors_d: np.array(_01re_new_neg_per_person_author_ids),
                        self.neg_paper_num: np.array(_01re_new_neg_person_papers),
                        self.keep_prob : 0.5})
        return loss


    def fine_tune_10(self, _10re_neg_feature, _10re_new_paper_word_ids, _10re_new_paper_word_idfs, _10re_new_paper_author_ids, _10re_new_paper_author_idfs,
        _10re_new_pos_person_word_ids, _10re_new_pos_per_person_word_ids, _10re_new_pos_person_papers, _10re_new_pos_person_author_ids, _10re_new_pos_per_person_author_ids):
        # print("neg_word_id:", np.array(_10re_new_pos_person_word_ids).shape)
        # print("per_neg_word_id:", np.array(_10re_new_pos_per_person_word_ids).shape)
        # print("person papers:", np.array(_10re_new_pos_person_papers).shape)
        # print("neg_author_id: ", np.array(_10re_new_pos_person_author_ids).shape)
        # print("per_neg_author_id:",np.array(_10re_new_pos_per_person_author_ids).shape)
        # print(_10re_new_pos_person_papers)

        loss, _ = self.sess.run([self._10_loss, self._10_optimizer], 
            feed_dict = {self.input_neg_feature: np.array(_10re_neg_feature),
                        self.train_words_q: np.array(_10re_new_paper_word_ids), self.train_words_q_weights: np.array(_10re_new_paper_word_idfs),
                        self.train_authors_q: np.array(_10re_new_paper_author_ids), self.train_authors_q_weights: np.array(_10re_new_paper_author_idfs),
                        self.pos_train_words_d: np.array(_10re_new_pos_person_word_ids), self.pos_train_person_words_d: np.array(_10re_new_pos_per_person_word_ids),
                        self.pos_train_authors_d: np.array(_10re_new_pos_person_author_ids),self.pos_train_person_authors_d: np.array(_10re_new_pos_per_person_author_ids),
                        self.pos_paper_num: np.array(_10re_new_pos_person_papers), 
                        self.keep_prob: 0.5})
        return loss


def print_variable(sess):
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print(k,v)

if __name__=='__main__':

    #logging
    save_path = "./new_checkpoints_god_tw/"
    log_dir = "Log/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, "log_%s"%(strftime('%Y-%m-%d%H:%M:%S', localtime()))), level=logging.INFO)

    print_logging("begin load training and test data ......")
    
    # pubs_dict = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'pubs_raw.json')

    # train_data = gen_train_data.generate_data_batches("name_to_pubs_train_500.json", '',settings.TRAIN_SCALE, 'TRAIN')
    
    # test_data = gen_train_data.generate_data_batches("name_to_pubs_test_100.json", settings.TEST_SCALE, 'TEST')
    
    train_data = gen_train_data.generate_data_batches("name_to_pubs_train.json", 'name_to_pubs_test.json',settings.TRAIN_SCALE, 'TRAIN')
    
    test_data = gen_train_data.generate_data_batches("name_to_pubs_train.json", "name_to_pubs_test.json", settings.TEST_SCALE, 'TEST')


    # exit()
    # data_utils.dump_data(train_data, settings.DATA_DIR, "train_data")
    # data_utils.dump_data(test_data, settings.DATA_DIR, "test_data")
     
    # train_data = data_utils.load_data(settings.DATA_DIR, "train_data")
    # test_data = data_utils.load_data(settings.DATA_DIR, "test_data")

    TR_neg_person_author_ids, TR_neg_person_word_ids, TR_neg_per_person_author_ids, TR_neg_per_person_word_ids, TR_neg_person_papers,\
    TR_new_pos_person_author_ids, TR_new_pos_person_word_ids, TR_new_pos_per_person_author_ids, TR_new_pos_per_person_word_ids,TR_new_pos_person_papers,\
    TR_new_paper_author_ids, TR_new_paper_author_idfs, TR_new_paper_word_ids, TR_new_paper_word_idfs,TR_pos_str_features, TR_neg_str_features,\
    = train_data[0],train_data[1],train_data[2],train_data[3],train_data[4],\
    train_data[5],train_data[6],train_data[7],train_data[8],train_data[9],\
    train_data[10],train_data[11], train_data[12],train_data[13], train_data[14], train_data[15]

    # print(TR_per_person_paper_num)
    # exit(0)

    TE_person_author_ids, TE_person_word_ids, TE_per_person_author_ids, TE_per_person_word_ids, TE_person_papers,\
    TE_new_paper_author_ids, TE_new_paper_author_idfs, TE_new_paper_word_ids, TE_new_paper_word_idfs, TE_pos_str_features,\
    = test_data[0],test_data[1],test_data[2],test_data[3],test_data[4],\
    test_data[5],test_data[6],test_data[7],test_data[8], test_data[9]
    
    # exit()
    
    batch_num = len(TR_neg_person_author_ids)
    print("batch_num: ", batch_num)
    batch_indexes = np.arange(batch_num)
    print("Author #train_batch=%d, #test_batch=%d" %(len(TR_new_pos_person_author_ids), len(TE_person_author_ids)))
    print("Word #train_batch=%d, #test_batch=%d" %(len(TR_neg_per_person_word_ids), len(TE_person_word_ids)))
    # TE_person_pid_cat, TE_pid_cat, TE_labels_cat = [],[],[]
    # for batch_index in np.arange(len(TE_person_author_id_list)):
    #     TE_person_pid_cat.extend(TE_person_pid_list[batch_index])
    #     TE_pid_cat.extend(TE_pid_list[batch_index])
    #     TE_labels_cat.extend(TE_labels[batch_index])
    # exit()

    print_logging("begin training global triplet model ......")
    tf_config = tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model = GlobalTripletModel(sess)
        sess.run(tf.global_variables_initializer())
        # print_variable(sess)
        hit_max = 0
        #train by epoch
        for epoch_count in range(settings.EPOCHES):

            # train_data = gen_train_data.generate_data_batches("name_to_pubs_train_500.json", settings.TRAIN_SCALE, 'TRAIN')

            # TR_neg_person_author_ids, TR_neg_person_word_ids, TR_neg_per_person_author_ids, TR_neg_per_person_word_ids, TR_neg_person_papers,\
            # TR_new_pos_person_author_ids, TR_new_pos_person_word_ids, TR_new_pos_per_person_author_ids, TR_new_pos_per_person_word_ids,TR_new_pos_person_papers,\
            # TR_new_paper_author_ids, TR_new_paper_author_idfs, TR_new_paper_word_ids, TR_new_paper_word_idfs,\
            # = train_data[0],train_data[1],train_data[2],train_data[3],train_data[4],\
            # train_data[5],train_data[6],train_data[7],train_data[8],train_data[9],\
            # train_data[10],train_data[11], train_data[12],train_data[13]

            # np.random.shuffle(batch_indexes)
            train_begin = time()
            avg_train_loss = 0
            
            # print(batch_indexes)

            # pool = mp.Pool(processes = 1)
            # resu = pool.apply_async(run1,())
            # #resu= pool.map(run1,())
            # pool.close()
            # pool.join()
            # # print(res)
            # # for i in resu:
            # #   print(i.get())
            # print(resu.get())

            for batch_index in batch_indexes:
                if(batch_index % 1000 == 0) and (batch_index > 0):
                    print("batch_index: ",batch_index)
                # print(batch_index)
                # train_loss, _= model.train(
                    # TR_person_word_id_list[batch_index], TR_paper_word_id_list[batch_index], TR_person_word_idf_list[batch_index],TR_person_author_id_list[batch_index], TR_paper_author_id_list[batch_index], TR_person_author_idf_list[batch_index],TR_labels[batch_index],\
                    # TR_per_person_author_ids[batch_index], TR_per_person_author_idfs[batch_index], TR_per_person_word_ids[batch_index], TR_per_person_word_idfs[batch_index], TR_per_person_paper_num[batch_index])
                train_loss, _ = model.train(TR_new_paper_word_ids[batch_index], TR_new_paper_word_idfs[batch_index], \
                    TR_new_pos_person_word_ids[batch_index], TR_new_pos_per_person_word_ids[batch_index], TR_new_pos_person_papers[batch_index], \
                    TR_neg_person_word_ids[batch_index], TR_neg_per_person_word_ids[batch_index], TR_neg_person_papers[batch_index],\
                    TR_new_paper_author_ids[batch_index], TR_new_paper_author_idfs[batch_index],\
                    TR_new_pos_person_author_ids[batch_index], TR_new_pos_per_person_author_ids[batch_index],\
                    TR_neg_person_author_ids[batch_index], TR_neg_per_person_author_ids[batch_index],
                    TR_pos_str_features[batch_index], TR_neg_str_features[batch_index])
                # print(train_loss)
                
                avg_train_loss += train_loss
            avg_train_loss /= batch_num
            train_time = time() - train_begin
            print("Epoch %d: train_loss = %.4f [%.1fs]" % (epoch_count, avg_train_loss, train_time))

            if epoch_count % settings.VERBOSE == 0:
                # print("test")
                # print_variable(sess)
                test_begin = time()
                # predictions, test_loss = model.predict(TE_person_word_id_list, TE_paper_word_id_list, TE_person_word_idf_list, 
                #     TE_person_author_id_list, TE_paper_author_id_list, TE_person_author_idf_list, TE_labels)

                predictions = model.predict(
                        TE_new_paper_author_ids, TE_new_paper_author_idfs, TE_new_paper_word_ids, TE_new_paper_word_idfs,\
                        TE_person_author_ids, TE_person_word_ids, TE_per_person_author_ids, TE_per_person_word_ids, TE_person_papers, TE_pos_str_features
                        )
                # print("ed")
                # print(predictions)


                hits = eval_utils.eval_hit(predictions)
                print("test_time: ",round(time()-test_begin, 2))
                # if(hits[0] > hit_max):
                if(epoch_count < 5):
                    model.saver.save(sess, save_path + 'global_model.ckpt', global_step=epoch_count)
                    if(hits[0] > hit_max):
                        hit_max = hits[0]
                elif(hits[0] > hit_max):
                    hit_max = hits[0]
                    model.saver.save(sess, save_path + 'global_model.ckpt', global_step=epoch_count)

            # del train_data
            # del TR_neg_person_author_ids
            # del TR_neg_person_word_ids
            # del TR_neg_per_person_author_ids
            # del TR_neg_per_person_word_ids
            # del TR_neg_person_papers
            # del TR_new_pos_person_author_ids
            # del TR_new_pos_person_word_ids
            # del TR_new_pos_per_person_author_ids
            # del TR_new_pos_per_person_word_ids
            # del TR_new_pos_person_papers
            # del TR_new_paper_author_ids
            # del TR_new_paper_author_idfs
            # del TR_new_paper_word_ids
            # del TR_new_paper_word_idfs

            # gc.collect()


            # if epoch_count % settings.VERBOSE == 0:
            #     # print("test")
            #     # print_variable(sess)
            #     test_begin = time()
            #     # predictions, test_loss = model.predict(TE_person_word_id_list, TE_paper_word_id_list, TE_person_word_idf_list, 
            #     #     TE_person_author_id_list, TE_paper_author_id_list, TE_person_author_idf_list, TE_labels)

            #     predictions = model.predict(
            #             TE_new_paper_author_ids, TE_new_paper_author_idfs, TE_new_paper_word_ids, TE_new_paper_word_idfs,\
            #             TE_person_author_ids, TE_person_word_ids, TE_per_person_author_ids, TE_per_person_word_ids, TE_person_papers
            #             )
            #     # print("ed")
            #     # print(predictions)
            #     eval_utils.eval_hit(predictions)



