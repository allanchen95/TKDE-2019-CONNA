from ranking_main import *
import classifier_data_profile as p_d
import classifier_settings as c_s
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import random
import numpy as np
import pickle
# print(c_s.TEST_BATCH_SIZE)
import json
from time import time

class classifier_model:
	def __init__(self, sess, learning_rate, feature_len, layer1_size):

		self.sess = sess
		self.learning_rate = learning_rate
		self.feature_len = feature_len
		self.layer1_size = layer1_size
		self.epsilon = 0.00001
		
		self._build_graph()
		self.saver = tf.train.Saver(max_to_keep=3)

	def weight_variable(self, shape, name):
		tmp = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
		initial = tf.random_uniform(shape, minval=-tmp, maxval=tmp)
		return tf.Variable(initial, name=name)

	def _create_placeholders(self):
		self.top_features = tf.placeholder(tf.float32, shape=[None, None], name='top_features')   # batch_size * feature_len
		self.top_str = tf.placeholder(tf.float32, shape=[None, None], name='top_str')   # batch_size * feature_len
		self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
		self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

	def _create_variables(self):
		self.w_w4 = self.weight_variable([self.layer1_size, 2], "w_w4")
		self.w_b4 = tf.Variable(tf.zeros([2]), name='w_b4')

		self.w_w3 = self.weight_variable([self.layer1_size, self.layer1_size], "w_w3")
		self.w_b3 = tf.Variable(tf.zeros([self.layer1_size]), name='w_b3')

		self.w_w2 = self.weight_variable([self.layer1_size, self.layer1_size], "w_w2")
		self.w_b2 = tf.Variable(tf.zeros([self.layer1_size]), name='w_b2')

		self.w_w1 = self.weight_variable([self.feature_len, self.layer1_size], "w_w1")
		self.w_b1 = tf.Variable(tf.zeros([self.layer1_size]), name='w_b1')

		self.s_w = self.weight_variable([10, 44], "w_w1")
		self.s_b = tf.Variable(tf.zeros([44]), name='w_b1')	

	def get_str_features(self, str_features):

		return tf.nn.leaky_relu(tf.matmul(str_features, self.s_w) + self.s_b)

	def _create_loss(self):

		# str_vec = self.get_str_features(self.top_str)

		# total_vec = tf.concat([self.top_features, str_vec], 1)
		total_vec = self.top_features

		layer1 = tf.nn.leaky_relu(tf.matmul(total_vec, self.w_w1) + self.w_b1)        
		layer1 = tf.nn.dropout(layer1, keep_prob = self.keep_prob)

		layer2 = tf.nn.leaky_relu(tf.matmul(layer1, self.w_w2) + self.w_b2)
		layer2 = tf.nn.dropout(layer2, keep_prob= self.keep_prob)

		layer3 = tf.nn.leaky_relu(tf.matmul(layer2, self.w_w3) + self.w_b3)
		layer3 = tf.nn.dropout(layer3, keep_prob= self.keep_prob)

		score = tf.matmul(layer3, self.w_w4) + self.w_b4

		self.predictions = tf.argmax(score, 1, name="predictions")
		self.prob = tf.nn.softmax(score, -1)
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score, labels = self.input_y))

		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.epsilon).minimize(self.loss)

		# fc_1 = tf.contrib.layers.fully_connected(sim_vec, 11, activation_fn=tf.nn.relu)
		# # fc_2 = tf.nn.dropout(fc_1, 0.5)
		# sim = tf.contrib.layers.fully_connected(fc_1, 2, activation_fn=None)
	def _build_graph(self):
		self._create_placeholders()
		self._create_variables()
		self._create_loss()


	def predict(self, input_feature_list, input_str_list, labels_list):
		predict_set = []
		true_set = []
		prob_set = []
		total_loss = 0
		for batch_index in range(len(input_feature_list)):
			test_loss, test_pre, prob= self.sess.run([self.loss, self.predictions, self.prob],
			feed_dict = {
						self.top_features: np.array(input_feature_list[batch_index]),
						self.top_str: np.array(input_str_list[batch_index]),
						self.input_y: np.array(labels_list[batch_index]),
						self.keep_prob: 1
			})
			prob_set.extend(prob)
			predict_set.extend(test_pre)
			true_set.extend(labels_list[batch_index])
			total_loss += test_loss

		avg_loss = round(total_loss/len(input_feature_list), 6)

		return avg_loss, predict_set, true_set, prob_set

	def train(self, input_feature, input_str, labels):
		return self.sess.run([self.loss, self.optimizer],
			feed_dict = {
						self.top_features: np.array(input_feature),
						self.top_str : np.array(input_str),
						self.input_y: np.array(labels),
						self.keep_prob: 0.5
			})

def eval_hit(predictions, sim_vec, str_features, scale, sample):

    top_k = [1, 3, 5]
    top_features = []
    top_str = []
    top_labels = []
    mrr = 0
    top_k_metric = np.array([0 for k in top_k])
    if (len(predictions)!= scale * (sample + 1)):
        print("predictions: {} don't match {}".format(len(predictions), scale * (sample + 1)))
        exit()

    predictions = np.array(predictions).reshape((scale, (sample+1)))

    for i in range(len(predictions)):
        rank = np.argsort(-predictions[i,:])
        # print(rank)
        # print(sim_vec[i])
        # print("r",sim_vec[i])
        true_index = np.where(rank == sample)[0][0]
        # if(true_index == 0):
        # 	sim_vec[i] = sim_vec[i][rank]
        # 	top_features.append(sim_vec[i][0])
        # 	top_labels.append(1)
        # 	top_features.append(sim_vec[i][1])
        # 	top_labels.append(0)


        mrr += 1/(true_index + 1)

        if(true_index != 0):
        	# tmp = rank[0]
        	# print(rank)
        	for i in np.arange(true_index-1, -1 ,-1):
        		rank[i+1] = rank[i]
        	# for i in np.arange(true_index-1, -1 ,-1):
        	# 	rank[i+1] = rank[i]


        	rank[0] = sample
        	# rank[true_index] = tmp
        	# print("re: ",rank)
        	# exit()
        # print(rank)
        # print(sim_vec[i])


        # if(true_index == 0):
        sim_vec[i] = sim_vec[i][rank]
        str_features[i] = str_features[i][rank]
        # concat_1 = sim_vec[i][0] + str_features[i][0] 
        top_features.append(sim_vec[i][0])
        top_str.append(str_features[i][0])
        top_labels.append(1)
        top_features.append(sim_vec[i][1])
        top_str.append(str_features[i][1])
        top_labels.append(0)

        # print(top_features)
        # print(top_labels)
        # top_k:[1, 5, 10, 50]
        for k in range(len(top_k)):
            if true_index < top_k[k]:
                top_k_metric[k] += 1
    # print(top_features[:10])
    # print(top_labels[:10])  

    ratio_top_k = np.array([0 for i in top_k], dtype = np.float32)

    for i in range(len(ratio_top_k)):
        ratio_top_k[i] = round(top_k_metric[i] / scale, 3)

    mrr /= scale
    print("train_eval hits@{} = {} mrr: {}".format(top_k, ratio_top_k, mrr))
    return ratio_top_k, np.array(top_features), np.array(top_str), np.array(top_labels), sim_vec


def test_eval_hit(glo_id2neg, predictions, ids_cat, sim_vec, str_features, scale, sample):

    top_k = [1, 3, 5]
    top_features = []
    top_str = []
    top_labels = []
    mrr = 0
    top_k_metric = np.array([0 for k in top_k])
    if (len(predictions)!= scale * (sample+1)):
        print("predictions: {} don't match {}".format(len(predictions), scale * (sample + 1)))
        exit()

    predictions = np.array(predictions).reshape((scale, (sample+1)))
    assert len(predictions) == len(ids_cat)
    with open('test_id2neg.json', 'r') as files:
    	id2neg = json.load(files)
    # id2neg = glo_id2neg
    for i in range(len(predictions)):
        rank = np.argsort(-predictions[i,:])
        # print(sim_vec[i])
        true_index = np.where(rank == sample)[0][0]
        mrr += 1/(true_index + 1)
		
        paper_id = ids_cat[i][0]
        if(id2neg.get(paper_id) == None):
        	continue
        neg_rank = int(id2neg[paper_id])
        top_features.append(sim_vec[i][-1])
        top_str.append(str_features[i][-1])
        top_labels.append(1)
        top_features.append(sim_vec[i][neg_rank])
        top_str.append(str_features[i][neg_rank])
        top_labels.append(0)

        for k in range(len(top_k)):
            if true_index < top_k[k]:
                top_k_metric[k] += 1

    ratio_top_k = np.array([0 for i in top_k], dtype = np.float32)

    for i in range(len(ratio_top_k)):
        ratio_top_k[i] = round(top_k_metric[i] / scale, 3)
    
    mrr /= scale
    print("test_eval hits@{} = {} mrr: {}".format(top_k, ratio_top_k, mrr))
    return _, np.array(top_features), np.array(top_str), np.array(top_labels), sim_vec



def rank_eval_hit(predictions, ids_cat, scale, sample):

    top_k = [1, 3, 5]
    top_k_metric = np.array([0 for k in top_k])
    mrr = 0
    if (len(predictions)!= scale * (sample+1)):
        print("predictions: {} don't match {}".format(len(predictions), scale * (sample+1)))
        exit()
    predictions = np.array(predictions).reshape((scale, (sample + 1)))
    assert len(predictions) == len(ids_cat)
	# self.loss = tf.reduce_mean(tf.maximum(0.0, 1 - self.pos_score + self.neg_score))
	# 
	# id2neg = {}
    test_loss = []
    for i in range(len(predictions)):
        ins_loss = []
        tmp_scores = predictions[i,:]
        right_s = tmp_scores[-1]
        err_ss = tmp_scores[:-1]
        assert len(err_ss) == c_s.RANK_TEST_SAMPLE
        for e_s in err_ss:
        	# print(right_s, e_s)
        	# e_s = float(e_s)
        	tmp_loss = max(0.0, 1.0-right_s+e_s)
        	ins_loss.append(tmp_loss)
        ins_loss = np.mean(np.array(ins_loss))
        test_loss.append(ins_loss)
        rank = np.argsort(-tmp_scores)
        tmp_id = ids_cat[i]
        true_index = np.where(rank == sample)[0][0]
        mrr += 1/(true_index + 1)
        # print(rank)		
        # if(true_index != 0):
        # 	for i in np.arange(true_index-1, -1 ,-1):
        # 		rank[i+1] = rank[i]
        # 	rank[0] = sample
		# right_score = 
		# predictions[i,:]

        # # print(rank, rank[1])
        # paper_id = tmp_id[0]
        # id2neg[paper_id] = str(rank[1])
        # top_k:[1, 5, 10, 50]
        for k in range(len(top_k)):
            if true_index < top_k[k]:
                top_k_metric[k] += 1
    # with open("test_id2neg.json", 'w') as files:
    # 	json.dump(id2neg, files, indent = 4)

    ratio_top_k = np.array([0 for i in top_k], dtype = np.float32)

    for i in range(len(ratio_top_k)):
        ratio_top_k[i] = round(top_k_metric[i] / scale, 3)

    mrr /= scale
    assert len(test_loss) == c_s.RANK_TEST_SCALE
    avg_tes = np.mean(np.array(test_loss))
    print("test_loss: {} hits@{} = {} mrr: {}".format(avg_tes, top_k, ratio_top_k, mrr))
	
    return ratio_top_k

def get_metric(true_set, predict_set):
	assert len(true_set) == len(predict_set) 
	true_num = 0
	pre_1_num = 0
	pre_1_total = 0
	pre_0_num = 0
	pre_0_total = 0
	gt_num = np.sum(true_set)
	no_num = len(true_set) - gt_num
	for i in range(len(true_set)):
		if(predict_set[i] == 1):
			pre_1_total += 1
		if(predict_set[i] == 0):
			pre_0_total += 1


		if(true_set[i] == predict_set[i]):
			true_num +=1
			if(true_set[i] == 1):
				pre_1_num += 1
			elif(true_set[i] == 0):
				pre_0_num += 1

		
	acc = round(true_num / len(true_set), 6)
	pre_1 = round(pre_1_num / max(pre_1_total, 1), 6)
	re_1 = round(pre_1_num / gt_num, 6)

	# print("gt: {} nb: {}".format(gt_num, no_num))		

	pre_0 = round(pre_0_num / max(pre_0_total, 1), 6)
	re_0 = round(pre_0_num / no_num, 6)
	# print("pre1: {} reca1: {}".format(pre_1, re_1))
	f1_1 = round(2 * pre_1 * re_1 / (pre_1 + re_1), 6)
	f1_0 = round(2 * pre_0 * re_0 / (pre_0 + re_0), 6)
	print("no_nil acc: {} pre: {} recall: {} f1: {}".format(round(acc, 4), round(pre_1, 4), round(re_1, 4), round(f1_1, 4)))
	print("nil acc: {}, pre: {} recall: {} f1 : {} ".format(round(acc, 4), round(pre_0, 4), round(re_0, 4), round(f1_0, 4)))
	# return acc, precision, recall, f1
	return acc


def get_id_neg(predictions, ids_cat, scale, sample):

    top_k = [1, 3, 5]
    top_k_metric = np.array([0 for k in top_k])
    mrr = 0
    if (len(predictions)!= scale * (sample+1)):
        print("predictions: {} don't match {}".format(len(predictions), scale * (sample+1)))
        exit()
    predictions = np.array(predictions).reshape((scale, (sample + 1)))
    assert len(predictions) == len(ids_cat)
	# self.loss = tf.reduce_mean(tf.maximum(0.0, 1 - self.pos_score + self.neg_score))
	# 
    id2neg = {}
    test_loss = []
    for i in range(len(predictions)):
        tmp_scores = predictions[i,:]
        rank = np.argsort(-tmp_scores)
        tmp_id = ids_cat[i]
        true_index = np.where(rank == sample)[0][0]
        mrr += 1/(true_index + 1)
        # print(rank)		
        if(true_index != 0):
        	for i in np.arange(true_index-1, -1 ,-1):
        		rank[i+1] = rank[i]
        	rank[0] = sample
		# right_score = 
		# predictions[i,:]

        # # print(rank, rank[1])
        paper_id = tmp_id[0]
        id2neg[paper_id] = str(rank[1])
	
    return id2neg

def batch_iterator(x, y, z, batch_size):
	data_len = len(x)
	assert len(x) == len(y) == len(z)
	for i in range(0, data_len, batch_size):
		yield (x[i:min(i + batch_size, data_len)], y[i:min(i + batch_size, data_len)], z[i:min(i + batch_size, data_len)]) 

# def batch_iterator(x, y, batch_size):
# 	data_len = len(x)
# 	for i in range(0, data_len, batch_size):
# 		yield (x[i:min(i + batch_size, data_len)], y[i:min(i + batch_size, data_len)]) 

if __name__ == '__main__':

	ranking_load_path = "./saved_ranking_model/"
	cl_save_path = './final_cl_profile/'


	use_rank_checkpoint = True
	use_classifier_checkpoint = False

	iter_num = 30

	top_k = [1, 3, 5]

	epoch_num = 100

	test_data = p_d.generate_test_data('train_author_pub_index_profile.json', 'train_author_pub_index_test.json', c_s.TEST_SCALE, c_s.TEST_SAMPLE, c_s.TEST_BATCH_SIZE, 'Train')
	# print(len(test_data))
	# test_data_t = p_d.generate_data_batches("name_to_pubs_train.json", "name_to_pubs_test.json", c_s.TEST_SCALE_T, c_s.TEST_SAMPLE_T, c_s.TEST_BATCH_SIZE_T, 'Test')
	b_t = time()
	test_data_rank = p_d.generate_test_data('test_author_pub_index_profile.json', 'test_author_pub_index_test.json', c_s.RANK_TEST_SCALE, c_s.RANK_TEST_SAMPLE, c_s.RANK_TEST_BATCH_SIZE, 'rank_test')
	feature_time = time()
	print("feature_processing_time: ", round(feature_time - b_t, 6))
	TE_person_author_ids, TE_person_word_ids, TE_per_person_author_ids, TE_per_person_word_ids, TE_person_papers,\
	TE_new_paper_author_ids, TE_new_paper_author_idfs, TE_new_paper_word_ids, TE_new_paper_word_idfs, TE_str_features, TE_ins_id,\
	= test_data[0], test_data[1], test_data[2], test_data[3], test_data[4],\
	test_data[5], test_data[6], test_data[7],test_data[8], test_data[9], test_data[10]

	TR_person_author_ids, TR_person_word_ids, TR_per_person_author_ids, TR_per_person_word_ids, TR_person_papers,\
	TR_new_paper_author_ids, TR_new_paper_author_idfs, TR_new_paper_word_ids, TR_new_paper_word_idfs, TR_str_features, TR_ins_id, \
	= test_data_rank[0], test_data_rank[1], test_data_rank[2], test_data_rank[3], test_data_rank[4],\
	test_data_rank[5], test_data_rank[6], test_data_rank[7],test_data_rank[8], test_data_rank[9], test_data_rank[10]
	# print(TE_instances)
	# print(TE_ins_dict)
	# print(len(TE_ins_dict))
	# exit()

	print("Test for rank model Author #test_batch=%d" %(len(TR_person_author_ids)))
	print("Test for rank Word #test_batch=%d" %(len(TR_person_word_ids)))

	print("Train classifier Author #test_batch=%d" %(len(TE_person_author_ids)))
	print("Train classifier #test_batch=%d" %(len(TE_person_word_ids)))

	print("Test classifier Author #test_batch=%d" %(len(TR_person_author_ids)))
	print("Test classifier Word #test_batch=%d" %(len(TR_person_word_ids)))
	# print(TE_labels)
	tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	tf_config.gpu_options.allow_growth = True


	rank_graph = tf.Graph()

	rank_sess = tf.Session(graph = rank_graph,config=tf_config)

	classifier_graph = tf.Graph()

	classifier_sess = tf.Session(graph = classifier_graph,config=tf_config)



	with rank_graph.as_default():
		with rank_sess.as_default():
			rank_model = GlobalTripletModel(rank_sess)
			if use_rank_checkpoint:
				ckpt = tf.train.get_checkpoint_state(load_path)
				if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
					print('Reloading rank model parameters...')
					print(ckpt.model_checkpoint_path)
					rank_model.saver.restore(rank_sess, ckpt.model_checkpoint_path)
			else:
				print("Create new rank model parameters...")
				rank_sess.run(tf.global_variables_initializer())

	with classifier_graph.as_default():
		with classifier_sess.as_default():
			cl_model = classifier_model(classifier_sess, 0.001, 44, 44)
			if use_classifier_checkpoint:
				ckpt = tf.train.get_checkpoint_state(cl_save_path)
				if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
					print('Reloading classifier model parameters...')
					print(ckpt.model_checkpoint_path)
					cl_model.saver.restore(classifier_sess, ckpt.model_checkpoint_path)
			else:
				print("Create new classifier model parameters...")
				classifier_sess.run(tf.global_variables_initializer())


	test_begin = time()


	max_hits = 0
	max_acc = 0
	glo_id2neg = {}
	for i in range(iter_num):
		print("==========Iteration: {}==========".format(i+1))

		print("Test ranking model")
		predictions, ids_cat = rank_model.predict_cl(
            TR_new_paper_author_ids, TR_new_paper_author_idfs, TR_new_paper_word_ids, TR_new_paper_word_idfs,\
            TR_person_author_ids, TR_person_word_ids, TR_per_person_author_ids, TR_per_person_word_ids, TR_person_papers, TR_ins_id)


		hits = rank_eval_hit(predictions, ids_cat, c_s.RANK_TEST_SCALE, c_s.RANK_TEST_SAMPLE)
		if(i==0):
			print("Generate ID2NEG")
			glo_id2neg = get_id_neg(predictions, ids_cat, c_s.RANK_TEST_SCALE, c_s.RANK_TEST_SAMPLE)

		print("Get train top_features")

		pre_cat = np.empty([0, 1])
		sim_vec = []
		str_features = []
		# ids_cat = []
		for batch_index in np.arange(len(TE_person_author_ids)):
			pre_score, feature = rank_model.get_top_feature(TE_new_paper_author_ids[batch_index], TE_new_paper_author_idfs[batch_index], TE_new_paper_word_ids[batch_index], TE_new_paper_word_idfs[batch_index], 
				TE_person_author_ids[batch_index], TE_person_word_ids[batch_index], TE_per_person_author_ids[batch_index], TE_per_person_word_ids[batch_index], TE_person_papers[batch_index])

			pre_cat = np.concatenate((pre_cat, np.reshape(pre_score,[-1,1])))
			# print(np.array(feature).shape)
			# sim_vec = np.concatenate((sim_vec, np.reshape(feature, [-1,22])))
			sim_vec.extend(feature)
			str_features.extend(TE_str_features[batch_index])
			# train_top_labels.extend(TE_labels[batch_index])
		sim_vec = np.array(sim_vec)
		# print(sim_vec)
		# print(np.array(sim_vec).shape)

		sim_vec = np.reshape(sim_vec, [c_s.TEST_SCALE, c_s.TEST_SAMPLE + 1, c_s.FEATURE_SIZE])
		str_features = np.reshape(str_features, [c_s.TEST_SCALE, c_s.TEST_SAMPLE + 1, 10])
		_, train_top_features, train_top_str, train_top_labels, train_sim_vec = eval_hit(pre_cat, sim_vec, str_features, c_s.TEST_SCALE, c_s.TEST_SAMPLE)
		# print("reshape:", sim_vec)
		# print(top_features)
		# for i in zip(train_top_str, train_top_labels):
		# 	print(i[0])
		# 	print(i[1])
		# exit()

		true_candidate_num = np.sum(np.array(train_top_labels))
		faker_candidate_num = len(train_top_labels) - true_candidate_num
		print("train true candidate: {} faker candidate: {}".format(true_candidate_num, faker_candidate_num))

		print("Get test top_features")
		pre_cat = np.empty([0, 1])
		sim_vec = []
		str_features = []
		ids_cat = []
		test_top_features = []
		test_top_labels = []
		matchting_time = time()
		for batch_index in np.arange(len(TR_person_author_ids)):
			pre_score, feature = rank_model.get_top_feature(TR_new_paper_author_ids[batch_index], TR_new_paper_author_idfs[batch_index], TR_new_paper_word_ids[batch_index], TR_new_paper_word_idfs[batch_index], 
				TR_person_author_ids[batch_index], TR_person_word_ids[batch_index], TR_per_person_author_ids[batch_index], TR_per_person_word_ids[batch_index], TR_person_papers[batch_index])

			pre_cat = np.concatenate((pre_cat, np.reshape(pre_score,[-1,1])))
			# print(np.array(feature).shape)
			# sim_vec = np.concatenate((sim_vec, np.reshape(feature, [-1,22])))
			ids_cat.extend(TR_ins_id[batch_index])
			sim_vec.extend(feature)
			str_features.extend(TR_str_features[batch_index])
			# test_top_labels.extend(TT_labels[batch_index])
		sim_vec = np.array(sim_vec)
		# print(sim_vec)
		# print(np.array(sim_vec).shape)
		sim_vec = np.reshape(sim_vec, [c_s.RANK_TEST_SCALE, c_s.RANK_TEST_SAMPLE + 1, c_s.FEATURE_SIZE])
		str_features = np.reshape(str_features, [c_s.RANK_TEST_SCALE, c_s.RANK_TEST_SAMPLE + 1, 10])
		_, test_top_features, test_top_str, test_top_labels, _ = test_eval_hit(glo_id2neg, pre_cat, ids_cat, sim_vec, str_features, c_s.RANK_TEST_SCALE, c_s.RANK_TEST_SAMPLE)
		end_matching_time = time()
		print("matching component time: ", round(end_matching_time - matchting_time, 6))


		true_candidate_num = np.sum(np.array(test_top_labels))
		faker_candidate_num = len(test_top_labels) - true_candidate_num
		print("test true candidate: {} faker candidate: {}".format(true_candidate_num, faker_candidate_num))

		# print("hits@{} = {} time cost: {}".format(top_k, hits, round(time() - test_begin, 3)))

		# Train classifier model
		print("Train classifier model")
		train_batch_x, train_batch_str_x, train_batch_y = [], [], []
		test_batch_x, test_batch_str_x, test_batch_y = [], [], []
		# print(train_top_labels)
		for x, y, z in batch_iterator(train_top_features, train_top_str, train_top_labels, 50):
			train_batch_x.append(x)
			train_batch_str_x.append(y)
			train_batch_y.append(z)

		for x, y, z in batch_iterator(test_top_features, test_top_str, test_top_labels, 128):
			test_batch_x.append(x)
			test_batch_str_x.append(y)
			test_batch_y.append(z)


		print("train_batch num: {}".format(len(train_batch_x)))
		print("test_batch num: {}".format(len(test_batch_x)))

		min_loss = 100
		each_ge = int(epoch_num/5)
		best_test_hit = 0
		for epoch_count in range(epoch_num):
			total_train_loss = 0
			for batch_index in range(len(train_batch_x)):

				train_loss, _= cl_model.train(train_batch_x[batch_index], train_batch_str_x[batch_index], train_batch_y[batch_index])
				total_train_loss += train_loss
			train_avg_loss = round(total_train_loss/len(train_batch_x), 6)

			
			if((epoch_count+1) % each_ge == 0):
				print("epoch :{} train_loss: {}".format(epoch_count, train_avg_loss))
				decision_time = time()
				test_avg_loss, predict_set, true_set, _ = cl_model.predict(test_batch_x, test_batch_str_x, test_batch_y)
				end_decision_time = time()
				print("decision component: ", round(end_decision_time - decision_time ,6))
				tmp_acc = get_metric(true_set, predict_set)
				print("test loss: {} ".format(test_avg_loss))
				

				if(tmp_acc > max_acc):
					cl_model.saver.save(cl_model.sess, cl_save_path + 'cl_model.ckpt', global_step=epoch_count)
					max_acc = tmp_acc

		# Create error case
		test_avg_loss, predict_set, true_set, train_prob = cl_model.predict(train_batch_x, train_batch_str_x, train_batch_y)
		# print("Train true_set_len: ", len(true_set))
		# print(train_prob[:10])
		assert len(true_set) == len(train_prob)
		acc = accuracy_score(true_set, predict_set)
		performance = metrics.precision_recall_fscore_support(true_set, predict_set, average='binary')
		precision = performance[0]
		recall = performance[1]
		f1 = performance[2]
		print("Train test loss: {} accuracy: {} precision: {} recall: {} f1: {}".format(test_avg_loss, round(acc, 6), round(precision, 6), round(recall, 6), round(f1, 6)))
		t1p0 = 0
		t1p0_list = []
		t1p0_prob = []
		
		t0p1 = 0
		t0p1_list = []
		t0p1_prob = []

		t1p1 = 0
		t1p1_list = []
		t1p1_prob = []

		t0p0 = 0
		t0p0_list = []
		t0p0_prob = []

		for i in range(len(true_set)):
			if true_set[i] != predict_set[i]:
				if(true_set[i] == 1):
					t1p0 += 1
					t1p0_list.append(i)
					t1p0_prob.append(train_prob[i][1])

				elif(true_set[i] == 0):
					t0p1 += 1
					t0p1_list.append(i)
					t0p1_prob.append(train_prob[i][1])

			elif(true_set[i] == predict_set[i]):
				if(true_set[i] == 1):
					t1p1 += 1
					t1p1_list.append(i)
					t1p1_prob.append(train_prob[i][1])

				elif(true_set[i] == 0):
					t0p0 += 1
					t0p0_list.append(i)
					t0p0_prob.append(train_prob[i][1])

		print("t1p1: {} t0p0: {} t1p0: {} t0p1: {}".format(t1p1, t0p0, t1p0, t0p1))

		
		print("Train sim_vec:", train_sim_vec.shape)
		

		# Reinforce both the two modules by their feedback.
		# We first implement it with REINFORCE algorithm, and find its performance is the same as 
		# Retraining the right cases, i.e., the accurately predicted cases, so for simplicity, we just 
		# retrain the right cases.
		ins_num = c_s.TEST_BATCH_SIZE / (c_s.TEST_SAMPLE + 1)
		total_matching_loss = []
		for index in t1p1_list:
			_11_neg_person_author_ids, _11_neg_person_word_ids, _11_neg_per_person_author_ids, _11_neg_per_person_word_ids, _11_neg_person_papers = [], [], [], [], []
			_11_new_pos_person_author_ids, _11_new_pos_person_word_ids, _11_new_pos_per_person_author_ids, _11_new_pos_per_person_word_ids, _11_new_pos_person_papers = [], [], [], [], []
			_11_new_paper_author_ids, _11_new_paper_author_idfs, _11_new_paper_word_ids, _11_new_paper_word_idfs, _11_pos_str_features, _11_neg_str_features = [], [], [], [], [], []
			if((index+1) % 2 != 1):
				print("error 1/1 index: ", index)
			vec_num = int(index/2)
			batch_num = int((vec_num+1) / ins_num)
			ins_num = ((vec_num+1) % ins_num)
			if(ins_num == 0):
				batch_num = batch_num - 1
				ins_num = ins_num - 1
			else:
				ins_num = ins_num - 1

			tmp_feature = [TE_new_paper_word_ids[batch_num][ins_num * (c_s.TEST_SAMPLE + 1) + c_s.TEST_SAMPLE]]
			tmp_feature = np.repeat(np.array(tmp_feature), c_s.TEST_SAMPLE, axis = 0).tolist()
			_11_new_paper_word_ids.extend(tmp_feature)

			tmp_feature = [TE_new_paper_word_idfs[batch_num][ins_num * (c_s.TEST_SAMPLE + 1) + c_s.TEST_SAMPLE]]
			tmp_feature = np.repeat(np.array(tmp_feature), c_s.TEST_SAMPLE, axis = 0).tolist()
			_11_new_paper_word_idfs.extend(tmp_feature)

			tmp_feature = [TE_new_paper_author_ids[batch_num][ins_num * (c_s.TEST_SAMPLE + 1) + c_s.TEST_SAMPLE]]
			tmp_feature = np.repeat(np.array(tmp_feature), c_s.TEST_SAMPLE, axis = 0).tolist()
			_11_new_paper_author_ids.extend(tmp_feature)

			tmp_feature = [TE_new_paper_author_idfs[batch_num][ins_num * (c_s.TEST_SAMPLE + 1) + c_s.TEST_SAMPLE]]
			tmp_feature = np.repeat(np.array(tmp_feature), c_s.TEST_SAMPLE, axis = 0).tolist()
			_11_new_paper_author_idfs.extend(tmp_feature)



			tmp_feature = [TE_person_word_ids[batch_num][ins_num * (c_s.TEST_SAMPLE + 1) + c_s.TEST_SAMPLE]]
			tmp_feature = np.repeat(np.array(tmp_feature), c_s.TEST_SAMPLE, axis = 0).tolist()
			_11_new_pos_person_word_ids.extend(tmp_feature)

			# print("TE_person_author_ids")
			# print(TE_person_author_ids[batch_num])
			# print(TE_person_author_ids[batch_num][ins_num * c_s.TEST_SAMPLE:])
			tmp_feature = [TE_person_author_ids[batch_num][ins_num * (c_s.TEST_SAMPLE + 1) + c_s.TEST_SAMPLE]]
			tmp_feature = np.repeat(np.array(tmp_feature), c_s.TEST_SAMPLE, axis = 0).tolist()
			_11_new_pos_person_author_ids.extend(tmp_feature)
			# exit()
			# print("re")
			# print(re_new_pos_person_author_ids)
			# exit()
			tmp_feature = [TE_per_person_word_ids[batch_num][ins_num * (c_s.TEST_SAMPLE + 1) + c_s.TEST_SAMPLE]]
			tmp_feature = np.repeat(np.array(tmp_feature), c_s.TEST_SAMPLE, axis = 0).tolist()
			_11_new_pos_per_person_word_ids.extend(tmp_feature)

			# print("TE_person_author_ids")
			# print(TE_per_person_author_ids[batch_num])
			tmp_feature = [TE_per_person_author_ids[batch_num][ins_num * (c_s.TEST_SAMPLE + 1) + c_s.TEST_SAMPLE]]
			tmp_feature = np.repeat(np.array(tmp_feature), c_s.TEST_SAMPLE, axis = 0).tolist()
			_11_new_pos_per_person_author_ids.extend(tmp_feature)

			tmp_feature = [TE_person_papers[batch_num][ins_num * (c_s.TEST_SAMPLE + 1) + c_s.TEST_SAMPLE]]
			tmp_feature = np.repeat(np.array(tmp_feature), c_s.TEST_SAMPLE, axis = 0).tolist()
			_11_new_pos_person_papers.extend(tmp_feature)


			tmp_feature = TE_person_word_ids[batch_num][(ins_num * (c_s.TEST_SAMPLE + 1)): ((ins_num) * (c_s.TEST_SAMPLE + 1) + c_s.TEST_SAMPLE)]
			# tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_neg_person_word_ids.extend(tmp_feature)

			tmp_feature = TE_person_author_ids[batch_num][(ins_num * (c_s.TEST_SAMPLE + 1)) : ((ins_num) * (c_s.TEST_SAMPLE + 1) + c_s.TEST_SAMPLE)]
			# tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_neg_person_author_ids.extend(tmp_feature)
			# exit()
			# print("re")
			# print(re_new_pos_person_author_ids)
			# exit()
			tmp_feature = TE_per_person_word_ids[batch_num][(ins_num * (c_s.TEST_SAMPLE + 1)) : ((ins_num) * (c_s.TEST_SAMPLE + 1) + c_s.TEST_SAMPLE)]
			# tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_neg_per_person_word_ids.extend(tmp_feature)

			# print("TE_person_author_ids")
			# print(TE_per_person_author_ids[batch_num])
			tmp_feature = TE_per_person_author_ids[batch_num][(ins_num * (c_s.TEST_SAMPLE + 1)): ((ins_num) * (c_s.TEST_SAMPLE + 1) + c_s.TEST_SAMPLE)]
			# tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_neg_per_person_author_ids.extend(tmp_feature)

			tmp_feature = TE_person_papers[batch_num][(ins_num * (c_s.TEST_SAMPLE + 1)) : ((ins_num) * (c_s.TEST_SAMPLE + 1) + c_s.TEST_SAMPLE)]
			# tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_neg_person_papers.extend(tmp_feature)

			# print("#retrain: ", len(_11_neg_person_papers))
			if(len(_11_neg_per_person_author_ids) % c_s.TEST_SAMPLE!= 0):
				print("NEG NUM DO NOT MATCH!!!")
				exit()
