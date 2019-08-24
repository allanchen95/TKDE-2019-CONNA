from global_model import *
import classifier_data as p_d
import classifier_settings as c_s
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import random
# print(c_s.TEST_BATCH_SIZE)


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
		self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
		self.keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

	def _create_variables(self):
		self.w_W2 = self.weight_variable([self.layer1_size, 2], "w_w2")
		self.w_b2 = tf.Variable(tf.constant(0.1, shape=[2]), name="w_b2")

		self.w_W1 = self.weight_variable([self.feature_len, self.layer1_size], "w_w1")
		self.w_b1 = tf.Variable(tf.zeros([self.layer1_size]), name='w_b1')

	def _create_loss(self):
		layer1 = tf.nn.relu(tf.matmul(self.top_features, self.w_W1) + self.w_b1)
		layer2 = tf.nn.dropout(layer1, keep_prob = self.keep_prob)
		score = tf.matmul(layer2, self.w_W2) + self.w_b2

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


	def predict(self, input_feature_list, labels_list):
		predict_set = []
		true_set = []
		prob_set = []
		total_loss = 0
		for batch_index in range(len(input_feature_list)):
			test_loss, test_pre, prob= self.sess.run([self.loss, self.predictions, self.prob],
			feed_dict = {
						self.top_features: np.array(input_feature_list[batch_index]),
						self.input_y: np.array(labels_list[batch_index]),
						self.keep_prob: 1
			})
			prob_set.extend(prob)
			predict_set.extend(test_pre)
			true_set.extend(labels_list[batch_index])
			total_loss += test_loss

		avg_loss = round(total_loss/len(input_feature_list), 6)

		return avg_loss, predict_set, true_set, prob_set

	def train(self, input_feature, labels):
		return self.sess.run([self.loss, self.optimizer],
			feed_dict = {
						self.top_features: np.array(input_feature),
						self.input_y: np.array(labels),
						self.keep_prob: 1
			})

def eval_hit(predictions, sim_vec, scale, sample):

    top_k = [1, 5, 10]
    top_features = []
    top_labels = []
    # mrr = 0
    top_k_metric = np.array([0 for k in top_k])
    if (len(predictions)!= scale * sample):
        print("predictions: {} don't match {}".format(len(predictions), scale * sample))
        exit()

    predictions = np.array(predictions).reshape((scale, sample))

    for i in range(len(predictions)):
        rank = np.argsort(-predictions[i,:])
        # print(rank)
        # print(sim_vec[i])
        # print("r",sim_vec[i])
        true_index = np.where(rank == 0)[0][0]
        # if(true_index == 0):
        # 	sim_vec[i] = sim_vec[i][rank]
        # 	top_features.append(sim_vec[i][0])
        # 	top_labels.append(1)
        # 	top_features.append(sim_vec[i][1])
        # 	top_labels.append(0)


        # mrr += 1/(true_index + 1)

        if(true_index != 0):
        	# tmp = rank[0]
        	# print(rank)
        	for i in np.arange(true_index-1, -1 ,-1):
        		rank[i+1] = rank[i]

        	rank[0] = 0
        	# rank[true_index] = tmp
        	# print("re: ",rank)
        	# exit()
        sim_vec[i] = sim_vec[i][rank]
        top_features.append(sim_vec[i][0])
        top_labels.append(1)
        top_features.append(sim_vec[i][1])
        top_labels.append(0)

        # top_k:[1, 5, 10, 50]
        for k in range(len(top_k)):
            if true_index < top_k[k]:
                top_k_metric[k] += 1


    ratio_top_k = np.array([0 for i in top_k], dtype = np.float32)

    for i in range(len(ratio_top_k)):
        ratio_top_k[i] = round(top_k_metric[i] / scale, 3)

    # mrr /= scale
    return ratio_top_k, np.array(top_features), np.array(top_labels), sim_vec


def test_eval_hit(predictions, sim_vec, scale, sample):

    top_k = [1, 5, 10]
    top_features = []
    top_labels = []
    # mrr = 0
    top_k_metric = np.array([0 for k in top_k])
    if (len(predictions)!= scale * sample):
        print("predictions: {} don't match {}".format(len(predictions), scale * sample))
        exit()

    predictions = np.array(predictions).reshape((scale, sample))

    for i in range(len(predictions)):
        rank = np.argsort(-predictions[i,:])
        # print(rank)
        # print(sim_vec[i])
        # print("r",sim_vec[i])
        # print(sim_vec[i])
        # print(rank)
        sim_vec[i] = sim_vec[i][rank]
        # print(sim_vec[i])
        true_index = np.where(rank == 0)[0][0]
        # mrr += 1/(true_index + 1)
        if(true_index != 0):
        	top_features.append(sim_vec[i][0])
        	top_labels.append(0)
        elif(true_index == 0):
        	top_features.append(sim_vec[i][0])
        	top_labels.append(1)
        	top_features.append(sim_vec[i][1])
        	top_labels.append(0)
        # print(top_features)
        # print(top_labels)
        # exit()
        	# sim_vec[i] = sim_vec[i][rank]
	        # top_features.append(sim_vec[i][0])
	        # top_labels.append(1)
	        # top_features.append(sim_vec[i][1])
	        # top_labels.append(0)
        	# rank[true_index] = tmp
        	# print("re: ",rank)
        	# exit()

    # mrr /= scale
    return _, np.array(top_features), np.array(top_labels), sim_vec



def rank_eval_hit(predictions, scale, sample):

    top_k = [1, 3, 5]
    top_k_metric = np.array([0 for k in top_k])
    mrr = 0
    if (len(predictions)!= scale * sample):
        print("predictions: {} don't match {}".format(len(predictions), scale * sample))
        exit()

    predictions = np.array(predictions).reshape((scale, sample))

    for i in range(len(predictions)):
        rank = np.argsort(-predictions[i,:])

        true_index = np.where(rank == 0)[0][0]
        mrr += 1/(true_index + 1)
        # top_k:[1, 5, 10, 50]
        for k in range(len(top_k)):
            if true_index < top_k[k]:
                top_k_metric[k] += 1


    ratio_top_k = np.array([0 for i in top_k], dtype = np.float32)

    for i in range(len(ratio_top_k)):
        ratio_top_k[i] = round(top_k_metric[i] / scale, 3)

    mrr /= scale
    print("hits@{} = {} mrr: {}".format(top_k, ratio_top_k, mrr))
    return ratio_top_k

def get_metric(true_set, predict_set):
	assert len(true_set) == len(predict_set) 
	true_num = 0
	pre_num = 0
	pre_total = 0
	for i in range(len(true_set)):
		if(predict_set[i] == 1):
			pre_total += 1

		if(true_set[i] == predict_set[i]):
			true_num +=1
			if(true_set[i] == 1):
				pre_num += 1
	acc = round(true_num/5600, 6)
	precision = round(pre_num/pre_total, 6)
	recall = round(pre_num/2800, 6)
	f1 = round(2 * precision * recall / (precision + recall), 6)

	return acc, precision, recall, f1




def batch_iterator(x, y, batch_size):
	data_len = len(x)
	for i in range(0, data_len, batch_size):
		yield (x[i:min(i + batch_size, data_len)], y[i:min(i + batch_size, data_len)]) 

if __name__ == '__main__':

	# load_path = "./checkpoints/"
	load_path = "./new_checkpoints_god_tw/"
	cl_save_path = './final_cl/'
	# rank_model_save_path = './final_rank/'

	use_rank_checkpoint = True
	use_classifier_checkpoint = False

	iter_num = 30

	top_k = [1, 5, 10]

	epoch_num = 100

	test_data = p_d.generate_data_batches("name_to_pubs_train.json", 'name_to_pubs_test.json', c_s.TEST_SCALE, c_s.TEST_SAMPLE, c_s.TEST_BATCH_SIZE, 'Train')

	# test_data_t = p_d.generate_data_batches("name_to_pubs_train.json", "name_to_pubs_test.json", c_s.TEST_SCALE_T, c_s.TEST_SAMPLE_T, c_s.TEST_BATCH_SIZE_T, 'Test')

	test_data_rank = p_d.generate_data_batches("name_to_pubs_train.json", "name_to_pubs_test.json", c_s.RANK_TEST_SCALE, c_s.RANK_TEST_SAMPLE, c_s.RANK_TEST_BATCH_SIZE, 'rank_test')

	TE_person_author_ids, TE_person_word_ids, TE_per_person_author_ids, TE_per_person_word_ids, TE_person_papers,\
	TE_new_paper_author_ids, TE_new_paper_author_idfs, TE_new_paper_word_ids, TE_new_paper_word_idfs, TE_labels, TE_str_features\
	= test_data[0], test_data[1], test_data[2], test_data[3], test_data[4],\
	test_data[5], test_data[6], test_data[7],test_data[8], test_data[9], test_data[10]


	# TT_person_author_ids, TT_person_word_ids, TT_per_person_author_ids, TT_per_person_word_ids, TT_person_papers,\
	# TT_new_paper_author_ids, TT_new_paper_author_idfs, TT_new_paper_word_ids, TT_new_paper_word_idfs, TT_labels, TT_str_features\
	# = test_data_t[0], test_data_t[1], test_data_t[2], test_data_t[3], test_data_t[4],\
	# test_data_t[5], test_data_t[6], test_data_t[7],test_data_t[8], test_data_t[9], test_data_t[10]



	TR_person_author_ids, TR_person_word_ids, TR_per_person_author_ids, TR_per_person_word_ids, TR_person_papers,\
	TR_new_paper_author_ids, TR_new_paper_author_idfs, TR_new_paper_word_ids, TR_new_paper_word_idfs, TR_labels, TR_str_features\
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
			cl_model = classifier_model(classifier_sess, 0.001, 88, 150)
			if use_classifier_checkpoint:
				ckpt = tf.train.get_checkpoint_state(cl_save_path)
				if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
					print('Reloading classifier model parameters...')
					print(ckpt.model_checkpoint_path)
					cl_model.saver.restore(classifier_sess, ckpt.model_checkpoint_path)
			else:
				print("Create new classifier model parameters...")
				classifier_sess.run(tf.global_variables_initializer())


	# print("Begin test raw hits")
	test_begin = time()

	# predictions = rank_model.predict(
	# 	TE_new_paper_author_ids, TE_new_paper_author_idfs, TE_new_paper_word_ids, TE_new_paper_word_idfs,\
	# 	TE_person_author_ids, TE_person_word_ids, TE_per_person_author_ids, TE_per_person_word_ids, TE_person_papers)
	# # print("ed")
	# # print(predictions)
	# hits = eval_hit(predictions)
	# print("hits@{} = {} time cost: {}".format(top_k, hits, round(time() - test_begin, 3)))
	max_hits = 0
	max_acc = 0
	for i in range(iter_num):
		print("==========Iteration: {}==========".format(i+1))

		print("Test ranking model")
		predictions = rank_model.predict(TR_new_paper_author_ids, TR_new_paper_author_idfs, TR_new_paper_word_ids, TR_new_paper_word_idfs,\
                        TR_person_author_ids, TR_person_word_ids, TR_per_person_author_ids, TR_per_person_word_ids, TR_person_papers, TR_str_features)


		hits = rank_eval_hit(predictions, c_s.RANK_TEST_SCALE, c_s.RANK_TEST_SAMPLE)
		if(hits[0] > max_hits):
			rank_model.saver.save(rank_model.sess, './final_rank_cl/global_model.ckpt', global_step=i)
			# cl_model.saver.save(cl_model.sess, cl_save_path + 'cl_model.ckpt', global_step=epoch_count)
			max_hits = hits[0]


		print("Get train top_features")

		pre_cat = np.empty([0, 1])
		sim_vec = []
		for batch_index in np.arange(len(TE_person_author_ids)):
			pre_score, feature = rank_model.get_top_feature(TE_new_paper_author_ids[batch_index], TE_new_paper_author_idfs[batch_index], TE_new_paper_word_ids[batch_index], TE_new_paper_word_idfs[batch_index], 
				TE_person_author_ids[batch_index], TE_person_word_ids[batch_index], TE_per_person_author_ids[batch_index], TE_per_person_word_ids[batch_index], TE_person_papers[batch_index], TE_str_features[batch_index])

			pre_cat = np.concatenate((pre_cat, np.reshape(pre_score,[-1,1])))
			# print(np.array(feature).shape)
			# sim_vec = np.concatenate((sim_vec, np.reshape(feature, [-1,22])))
			sim_vec.extend(feature)
			# train_top_labels.extend(TE_labels[batch_index])
		sim_vec = np.array(sim_vec)
		# print(sim_vec)
		# print(np.array(sim_vec).shape)

		sim_vec = np.reshape(sim_vec, [int(c_s.TEST_SCALE/2), c_s.TEST_SAMPLE, c_s.FEATURE_SIZE])
		_, train_top_features, train_top_labels, train_sim_vec = eval_hit(pre_cat, sim_vec,int(c_s.TEST_SCALE/2), c_s.TEST_SAMPLE)
		# print("reshape:", sim_vec)
		# print(top_features)
		# exit()

		true_candidate_num = np.sum(np.array(train_top_labels))
		faker_candidate_num = len(train_top_labels) - true_candidate_num
		print("train true candidate: {} faker candidate: {}".format(true_candidate_num, faker_candidate_num))

		print("Get test top_features")
		pre_cat = np.empty([0, 1])
		sim_vec = []
		test_top_features = []
		test_top_labels = []
		for batch_index in np.arange(len(TR_person_author_ids)):
			pre_score, feature = rank_model.get_top_feature(TR_new_paper_author_ids[batch_index], TR_new_paper_author_idfs[batch_index], TR_new_paper_word_ids[batch_index], TR_new_paper_word_idfs[batch_index], 
				TR_person_author_ids[batch_index], TR_person_word_ids[batch_index], TR_per_person_author_ids[batch_index], TR_per_person_word_ids[batch_index], TR_person_papers[batch_index], TR_str_features[batch_index])

			pre_cat = np.concatenate((pre_cat, np.reshape(pre_score,[-1,1])))
			# print(np.array(feature).shape)
			# sim_vec = np.concatenate((sim_vec, np.reshape(feature, [-1,22])))
			sim_vec.extend(feature)
			# test_top_labels.extend(TT_labels[batch_index])
		sim_vec = np.array(sim_vec)
		# print(sim_vec)
		# print(np.array(sim_vec).shape)
		sim_vec = np.reshape(sim_vec, [int(c_s.TEST_SCALE_T/2), c_s.TEST_SAMPLE_T, c_s.FEATURE_SIZE])
		_, test_top_features, test_top_labels, _ = test_eval_hit(pre_cat, sim_vec, int(c_s.TEST_SCALE_T/2), c_s.TEST_SAMPLE_T)
			
		# for i in range(len(test_sim_vec)):
		# 	test_top_features.append(test_sim_vec[i][0])
		# 	test_top_features.extend(test_sim_vec[i][1:])

		# 	test_top_labels.append(1)
		# 	test_top_labels.extend([0]*4)

		true_candidate_num = np.sum(np.array(test_top_labels))
		faker_candidate_num = len(test_top_labels) - true_candidate_num
		print("test true candidate: {} faker candidate: {}".format(true_candidate_num, faker_candidate_num))

		# print("hits@{} = {} time cost: {}".format(top_k, hits, round(time() - test_begin, 3)))
		# exit()

		# Train classifier model
		print("Train classifier model")
		train_batch_x, train_batch_y = [], []
		test_batch_x, test_batch_y = [], []
		# print(train_top_labels)
		for x, y in batch_iterator(train_top_features, train_top_labels, 128):
			train_batch_x.append(x)
			train_batch_y.append(y)

		for x, y in batch_iterator(test_top_features, test_top_labels, 128):
			test_batch_x.append(x)
			test_batch_y.append(y)


		print("train_batch num: {}".format(len(train_batch_x)))
		print("test_batch num: {}".format(len(test_batch_x)))

		min_loss = 100
		for epoch_count in range(epoch_num):
			total_train_loss = 0
			for batch_index in range(len(train_batch_x)):

				train_loss, _= cl_model.train(train_batch_x[batch_index], train_batch_y[batch_index])
				total_train_loss += train_loss
			train_avg_loss = round(total_train_loss/len(train_batch_x), 6)

			
			if((epoch_count+1) % 10 == 0):
				print("epoch :{} train_loss: {}".format(epoch_count, train_avg_loss))
				test_avg_loss, predict_set, true_set, _ = cl_model.predict(test_batch_x, test_batch_y)
				# test_avg_loss, predict_set, true_set = cl_model.predict(train_batch_x, train_batch_y)
				# print("true_set_len: ", len(true_set))
				# t1p0 = 0
				# t0p1 = 1
				# for i in range(len(true_set)):
				# 	if true_set[i] != predict_set[i]:
				# 		if(true_set[i] == 1):
				# 			t1p0 += 1
				# 		elif(true_set[i] == 0 ):
				# 			t0p1 += 1

				# print("t1p0: {} t0p1: {}".format(t1p0, t0p1))

				# acc = accuracy_score(true_set, predict_set)
				# performance = metrics.precision_recall_fscore_support(true_set, predict_set, average='binary')
				# precision = performance[0]
				# recall = performance[1]
				# f1 = performance[2]
				acc, precision, recall, f1 = get_metric(true_set, predict_set)
				print("test loss: {} accuracy: {} precision: {} recall: {} f1: {}".format(test_avg_loss, round(acc, 4), round(precision, 4), round(recall, 4), round(f1, 4)))
				
				# if (test_avg_loss < min_loss) or (acc > max_acc):
				# 	if(test_avg_loss < min_loss):
				# 		min_loss = test_avg_loss
				# 	if max_acc < acc:
				# 		max_acc = acc
				# else:
				# 	print("max_acc :{} min_loss: {}".format(max_acc, min_loss))
				# 	break

				if(acc > max_acc):
					cl_model.saver.save(cl_model.sess, cl_save_path + 'cl_model.ckpt', global_step=epoch_count)
					max_acc = acc

		# Create error case
		# print("Create error case")
		test_avg_loss, predict_set, true_set, train_prob = cl_model.predict(train_batch_x, train_batch_y)
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
		# For 1/1 case:
		for index in t1p1_list:
			_11_neg_person_author_ids, _11_neg_person_word_ids, _11_neg_per_person_author_ids, _11_neg_per_person_word_ids, _11_neg_person_papers = [], [], [], [], []
			_11_new_pos_person_author_ids, _11_new_pos_person_word_ids, _11_new_pos_per_person_author_ids, _11_new_pos_per_person_word_ids, _11_new_pos_person_papers = [], [], [], [], []
			_11_new_paper_author_ids, _11_new_paper_author_idfs, _11_new_paper_word_ids, _11_new_paper_word_idfs, _11_pos_str_features, _11_neg_str_features = [], [], [], [], [], []

			if((index+1) % 2 != 1):
				print("error 1/1 index: ", index)
			vec_num = int(index/2)
			batch_num = int((vec_num+1) / 20)
			ins_num = ((vec_num+1) % 20)
			if(ins_num == 0):
				batch_num = batch_num - 1
				ins_num = 19
			else:
				ins_num = ins_num - 1

			tmp_feature = [TE_new_paper_word_ids[batch_num][ins_num * 9]]
			tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_new_paper_word_ids.extend(tmp_feature)

			tmp_feature = [TE_new_paper_word_idfs[batch_num][ins_num * 9]]
			tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_new_paper_word_idfs.extend(tmp_feature)

			tmp_feature = [TE_new_paper_author_ids[batch_num][ins_num * 9]]
			tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_new_paper_author_ids.extend(tmp_feature)

			tmp_feature = [TE_new_paper_author_idfs[batch_num][ins_num * 9]]
			tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_new_paper_author_idfs.extend(tmp_feature)



			tmp_feature = [TE_person_word_ids[batch_num][ins_num * 9]]
			tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_new_pos_person_word_ids.extend(tmp_feature)

			# print("TE_person_author_ids")
			# print(TE_person_author_ids[batch_num])
			# print(TE_person_author_ids[batch_num][ins_num * 9:])
			tmp_feature = [TE_person_author_ids[batch_num][ins_num * 9]]
			tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_new_pos_person_author_ids.extend(tmp_feature)
			# exit()
			# print("re")
			# print(re_new_pos_person_author_ids)
			# exit()
			tmp_feature = [TE_per_person_word_ids[batch_num][ins_num * 9]]
			tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_new_pos_per_person_word_ids.extend(tmp_feature)

			# print("TE_person_author_ids")
			# print(TE_per_person_author_ids[batch_num])
			tmp_feature = [TE_per_person_author_ids[batch_num][ins_num * 9]]
			tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_new_pos_per_person_author_ids.extend(tmp_feature)

			tmp_feature = [TE_person_papers[batch_num][ins_num * 9]]
			tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_new_pos_person_papers.extend(tmp_feature)

			tmp_feature = [TE_str_features[batch_num][ins_num * 9]]
			tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_pos_str_features.extend(tmp_feature)


			tmp_feature = TE_person_word_ids[batch_num][(ins_num * 9) + 1: (ins_num+1) * 9]
			# tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_neg_person_word_ids.extend(tmp_feature)

			tmp_feature = TE_person_author_ids[batch_num][(ins_num * 9) + 1: (ins_num+1) * 9]
			# tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_neg_person_author_ids.extend(tmp_feature)
			# exit()
			# print("re")
			# print(re_new_pos_person_author_ids)
			# exit()
			tmp_feature = TE_per_person_word_ids[batch_num][(ins_num * 9) + 1: (ins_num+1) * 9]
			# tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_neg_per_person_word_ids.extend(tmp_feature)

			# print("TE_person_author_ids")
			# print(TE_per_person_author_ids[batch_num])
			tmp_feature = TE_per_person_author_ids[batch_num][(ins_num * 9) + 1: (ins_num+1) * 9]
			# tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_neg_per_person_author_ids.extend(tmp_feature)

			tmp_feature = TE_person_papers[batch_num][(ins_num * 9) + 1: (ins_num+1) * 9]
			# tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_neg_person_papers.extend(tmp_feature)

			tmp_feature = TE_str_features[batch_num][(ins_num * 9) + 1: (ins_num+1) * 9]
			# tmp_feature = np.repeat(np.array(tmp_feature), 8, axis = 0).tolist()
			_11_neg_str_features.extend(tmp_feature)


			train_loss, _ = rank_model.train(_11_new_paper_word_ids, _11_new_paper_word_idfs, \
				_11_new_pos_person_word_ids, _11_new_pos_per_person_word_ids, _11_new_pos_person_papers, \
				_11_neg_person_word_ids, _11_neg_per_person_word_ids, _11_neg_person_papers,\
				_11_new_paper_author_ids, _11_new_paper_author_idfs,\
				_11_new_pos_person_author_ids, _11_new_pos_per_person_author_ids,\
				_11_neg_person_author_ids, _11_neg_per_person_author_ids,\
				_11_pos_str_features, _11_neg_str_features)



