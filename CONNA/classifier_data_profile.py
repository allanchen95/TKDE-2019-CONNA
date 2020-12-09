from os.path import join
import sys
sys.path.append("..")
import os
import multiprocessing as mp
import random
from datetime import datetime
from utils import data_utils
import classifier_settings as settings
from utils import multithread_utils
from collections import defaultdict
import numpy as np
from gensim.models import Word2Vec
import copy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re

start_time = datetime.now()

name2aidpid = None
name2pid = None
pid_dict = None
pid_list = None
pub_feature_dict = None
author_num = None
word_num = None


def generate_test_data(train_data_file, test_data_file, data_size, sample_num, batch_size, info):
    print("INFO: ", info)
    print("data_size: {} sample_num: {} batch_size: {}".format(data_size, sample_num, batch_size))
    global name2aidpid
    global pid_dict
    global pid_list
    global neg_sample
    global pub_feature_dict
    global author_num
    global word_num
    # global index_list
    neg_sample = sample_num


  
    pub_feature_dict = data_utils.load_data(settings.EMB_DATA_DIR, "pub_feature.ids")
    print("Loaded pub features")
    # author_num = len(data_utils.load_data('/root/zhangjing/NA/emb/', "author_emb.array"))
    author_num = len(data_utils.load_data(settings.EMB_DATA_DIR, "author_emb.array"))
    print("#author = %d" %author_num)

    # word_num = len(data_utils.load_data('/root/zhangjing/NA/emb/', "title_emb.array"))
    word_num = len(data_utils.load_data(settings.EMB_DATA_DIR, "word_emb.array"))
    print("#Word = %d" %word_num)

    train_name2aidpid = data_utils.load_json(settings.NEW_DATA_DIR, train_data_file)
    test_name2aidpid = data_utils.load_json(settings.NEW_DATA_DIR, test_data_file)

    name2aidpid = train_name2aidpid

	#split train/test
    name_valid = set()
    for name in train_name2aidpid:
        person_id = train_name2aidpid[name]
        if(len(person_id) > neg_sample):
            name_valid.add((name))

    pid_dict = {}
    pid_list = []
    
    print('#name = %d' % len(name_valid))

    for name in name_valid:
        aid2pid = test_name2aidpid[name]
        for aid in aid2pid:
            for pid in aid2pid[aid]:
                # may be twice
                pid_dict[pid] = (name, aid)
                pid_list.append(pid)

    print("Processed name2aidpid #paper_list: ", len(pid_list))

    use_list = random.sample(pid_list, min(data_size, len(pid_list)))

    print("Get data!!!")
    print("pid_list:{}".format(len(use_list))) 
    # index_list = np.arange(len(use_list))
    # random.shuffle(index_list)
    return get_test_data_batches(use_list, neg_sample, batch_size, info)


def get_valid_index(paper_list, neg_sample):
    valid_index = []
    print("neg_sample:", neg_sample)
    for pos_pid in paper_list:
        (name, aid) = pid_dict[pos_pid]

        # sample_list = name2plist[name]
        sample_list = list(name2aidpid[name])

        if len(sample_list) <= neg_sample:
            print("error! person num <= :", neg_sample)
            exit()

        copy_sample_list = copy.deepcopy(sample_list)
        copy_sample_list.remove(aid)
        random.shuffle(copy_sample_list)
        assert len(copy_sample_list) == (len(sample_list) - 1)

        neg_author_list = random.sample(copy_sample_list, neg_sample)
        valid_index.append((pos_pid, neg_author_list))
    # print(valid_index)
    return valid_index


def get_test_data_batches(paper_list, neg_sample, batch_size, info):
    valid_index = get_valid_index(paper_list, neg_sample)
    if(info == 'Train'):
        print("train info")
        print("train len_valid_index:",len(valid_index))
    elif(info == 'rank_test'):
        print("rank test info!")
        print("test len_valid_index:",len(valid_index))

    # res = multithread_utils.processed_by_multi_thread(_test_gen_pos_and_neg_pair, valid_index)
    res = []
    res_id = []
    for ins in valid_index:
        res.append(_test_gen_pos_and_neg_pair(ins, neg_sample))
        res_id.append(ins)
        if(len(res) % 1000 == 0):
            print('now process: ', len(res))

    person_author_ids, person_word_ids, per_person_author_ids, per_person_word_ids, author_len, word_len, author_whole,  word_whole, person_papers = [], [], [], [], [], [], [], [], [] 
    new_paper_author_ids, new_paper_author_idfs, new_paper_word_ids, new_paper_word_idfs, new_paper_authors, new_paper_words = [], [], [], [], [], []
    neg_str_features = []
    ins_id = []

    b_person_author_ids, b_person_word_ids, b_per_person_author_ids, b_per_person_word_ids,b_person_papers = [], [], [], [], []
    b_new_paper_author_ids, b_new_paper_author_idfs, b_new_paper_word_ids, b_new_paper_word_idfs = [], [], [], []
    b_neg_str_features = []
    b_ins_id = []
    for i, t in enumerate(res):
        if i % 1000 == 0:
            print(i, datetime.now()-start_time)
        if t == None:
            continue
        tmp_id = res_id[i]
        person_author_id_list, person_word_id_list, per_person_author_id_list, per_person_word_id_list, author_len_list, word_len_list, author_whole_len, word_whole_len, person_paper_num = t[0], t[1], t[2], t[3], t[4],t[5],t[6], t[7], t[8]
        new_paper_author_id_list, new_paper_author_idf_list, new_paper_word_id_list, new_paper_word_idf_list, new_paper_author_num, new_paper_word_num = t[9],t[10],t[11],t[12],t[13],t[14]
        neg_str_features_list = t[15]
        # print("ttt:",paper_num)

        person_author_ids.extend(person_author_id_list) 
        person_word_ids.extend(person_word_id_list)
        per_person_author_ids.extend(per_person_author_id_list)
        per_person_word_ids.extend(per_person_word_id_list)
        author_len.extend(author_len_list)
        word_len.extend(word_len_list)
        author_whole.extend(author_whole_len)
        word_whole.extend(word_whole_len)
        # print(person_paper_num)
        person_papers.extend(person_paper_num)
        # print(person_papers)

        new_paper_author_ids.extend(new_paper_author_id_list)
        new_paper_author_idfs.extend(new_paper_author_idf_list)
        new_paper_word_ids.extend(new_paper_word_id_list)
        new_paper_word_idfs.extend(new_paper_word_idf_list)
        new_paper_authors.extend(new_paper_author_num)
        new_paper_words.extend(new_paper_word_num)

        neg_str_features.extend(neg_str_features_list)
        ins_id.append(tmp_id)

        if(len(person_author_ids) % (neg_sample + 1) != 0):
            print("NEG NUM DO NOT MATCH!!!")
            exit()
        if(len(person_author_ids) == batch_size):

            new_paper_author_ids = _add_mask(author_num, new_paper_author_ids, min(max(new_paper_authors),settings.MAX_AUTHOR))
            new_paper_author_idfs = _add_mask(0, new_paper_author_idfs, min(max(new_paper_authors),settings.MAX_AUTHOR))
            new_paper_word_ids = _add_mask(word_num, new_paper_word_ids, min(max(new_paper_words),settings.MAX_WORD))
            new_paper_word_idfs = _add_mask(0, new_paper_word_idfs, min(max(new_paper_words),settings.MAX_WORD))

            person_author_ids = _add_mask(author_num, person_author_ids, min(max(author_whole),settings.MAX_AUTHOR))
            person_word_ids = _add_mask(word_num, person_word_ids, min(max(word_whole),settings.MAX_WORD))


            tmp_author_id, tmp_word_id = [], []

            pad_paper_num = np.max(np.array(person_papers))
            # print(np.array(neg_person_papers))

            pad_author = min(np.max(np.array(author_len)), settings.MAX_PER_AUTHOR)
            # print(np.array(neg_author_len))

            pad_word = min(np.max(np.array(word_len)), settings.MAX_WORD)


            for i in range(len(per_person_author_ids)):
                per_author_id_list = per_person_author_ids[i]
                per_word_id_list = per_person_word_ids[i]
   
                # print("pad_paper: {} pad_author: {} pad_word: {}".format(pad_paper_num, pad_author, pad_word))

                per_author_id_list = _add_paper_mask(author_num, per_author_id_list, pad_author, min(pad_paper_num, settings.MAX_PAPER))
                per_word_id_list = _add_paper_mask(word_num, per_word_id_list, pad_word, min(pad_paper_num, settings.MAX_PAPER))

                
                tmp_author_id.append(per_author_id_list)
                tmp_word_id.append(per_word_id_list)


            b_person_author_ids.append(person_author_ids) 
            b_person_word_ids.append(person_word_ids)
            b_per_person_author_ids.append(tmp_author_id)
            b_per_person_word_ids.append(tmp_word_id)
            # print(np.array(neg_person_papers))
            
            person_papers = np.array(person_papers)
            person_papers[person_papers > settings.MAX_PAPER] = settings.MAX_PAPER
            b_person_papers.append(person_papers.tolist())
            # print(neg_person_papers)
            # print(np.array(neg_tmp_author_id).shape)

            b_new_paper_author_ids.append(new_paper_author_ids)
            b_new_paper_author_idfs.append(new_paper_author_idfs)
            b_new_paper_word_ids.append(new_paper_word_ids)
            b_new_paper_word_idfs.append(new_paper_word_idfs)

            b_neg_str_features.append(neg_str_features)

            b_ins_id.append(ins_id)

            # print(np.array(b_new_pos_per_person_author_ids).shape)


            person_author_ids, person_word_ids, per_person_author_ids, per_person_word_ids, author_len, word_len, author_whole,  word_whole, person_papers = [], [], [], [], [], [], [], [], [] 
            new_paper_author_ids, new_paper_author_idfs, new_paper_word_ids, new_paper_word_idfs, new_paper_authors, new_paper_words = [], [], [], [], [], []
            neg_str_features = []
            ins_id = []

    if person_author_ids:
        new_paper_author_ids = _add_mask(author_num, new_paper_author_ids, min(max(new_paper_authors),settings.MAX_AUTHOR))
        new_paper_author_idfs = _add_mask(0, new_paper_author_idfs, min(max(new_paper_authors),settings.MAX_AUTHOR))
        new_paper_word_ids = _add_mask(word_num, new_paper_word_ids, min(max(new_paper_words),settings.MAX_WORD))
        new_paper_word_idfs = _add_mask(0, new_paper_word_idfs, min(max(new_paper_words),settings.MAX_WORD))

        person_author_ids = _add_mask(author_num, person_author_ids, min(max(author_whole),settings.MAX_AUTHOR))
        person_word_ids = _add_mask(word_num, person_word_ids, min(max(word_whole),settings.MAX_WORD))


        tmp_author_id, tmp_word_id = [], []

        pad_paper_num = np.max(np.array(person_papers))
        # print(np.array(neg_person_papers))

        pad_author = min(np.max(np.array(author_len)), settings.MAX_PER_AUTHOR)
        # pad_author = np.max(np.array(author_len))
        # print(np.array(neg_author_len))

        pad_word = min(np.max(np.array(word_len)), settings.MAX_WORD)
        # pad_word = np.max(np.array(word_len))


        for i in range(len(per_person_author_ids)):
            per_author_id_list = per_person_author_ids[i]
            per_word_id_list = per_person_word_ids[i]

            # print("pad_paper: {} pad_author: {} pad_word: {}".format(pad_paper_num, pad_author, pad_word))

            per_author_id_list = _add_paper_mask(author_num, per_author_id_list, pad_author, min(pad_paper_num, settings.MAX_PAPER))
            per_word_id_list = _add_paper_mask(word_num, per_word_id_list, pad_word, min(pad_paper_num, settings.MAX_PAPER))

            
            tmp_author_id.append(per_author_id_list)
            tmp_word_id.append(per_word_id_list)


        b_person_author_ids.append(person_author_ids) 
        b_person_word_ids.append(person_word_ids)
        b_per_person_author_ids.append(tmp_author_id)
        b_per_person_word_ids.append(tmp_word_id)
        # print(np.array(neg_person_papers))
        
        person_papers = np.array(person_papers)
        person_papers[person_papers > settings.MAX_PAPER] = settings.MAX_PAPER
        b_person_papers.append(person_papers.tolist())
        # print(neg_person_papers)
        # print(np.array(neg_tmp_author_id).shape)

        b_new_paper_author_ids.append(new_paper_author_ids)
        b_new_paper_author_idfs.append(new_paper_author_idfs)
        b_new_paper_word_ids.append(new_paper_word_ids)
        b_new_paper_word_idfs.append(new_paper_word_idfs)

        b_neg_str_features.append(neg_str_features)
        b_ins_id.append(ins_id)

    return b_person_author_ids, b_person_word_ids, b_per_person_author_ids, b_per_person_word_ids, b_person_papers,\
    b_new_paper_author_ids, b_new_paper_author_idfs, b_new_paper_word_ids, b_new_paper_word_idfs, b_neg_str_features, b_ins_id





def get_paper_neg_instances(pid_list):

    person_author_ids, person_word_ids = [], []
    per_person_author_ids, per_person_word_ids = [], [] 
    author_len, word_len = [], []


    for pid in pid_list:
        
        # author_id_list, author_idf_list, word_id_list, word_idf_list, _, _, _,_ ,_ ,_ = pub_feature_dict[pid]
        author_id_list, author_idf_list, word_id_list, word_idf_list = pub_feature_dict[pid]
        person_author_ids += author_id_list
        person_word_ids += word_id_list

        per_person_author_ids.append(author_id_list)
        per_person_word_ids.append(word_id_list)

        # assert (len(author_id_list) == len(author_idf_list)) & (len(word_id_list) == len(word_idf_list))

        author_len.extend([len(author_id_list)])
        word_len.extend([len(word_id_list)])

    return person_author_ids, person_word_ids, per_person_author_ids, per_person_word_ids, author_len, word_len, len(person_author_ids), len(person_word_ids)

def get_paper_pos_instances(pid_list, self_id):

    person_author_ids, person_word_ids = [], []
    per_person_author_ids, per_person_word_ids = [], [] 
    author_len, word_len = [], []

    for pid in pid_list:

        if(pid != self_id):
        
            # author_id_list, author_idf_list, word_id_list, word_idf_list, _, _, _,_ ,_ ,_= pub_feature_dict[pid]
            author_id_list, author_idf_list, word_id_list, word_idf_list = pub_feature_dict[pid]
            person_author_ids += author_id_list
            person_word_ids += word_id_list

            per_person_author_ids.append(author_id_list)
            per_person_word_ids.append(word_id_list)

            # assert (len(author_id_list) == len(author_idf_list)) & (len(word_id_list) == len(word_idf_list))

            author_len.extend([len(author_id_list)])
            word_len.extend([len(word_id_list)])

    return person_author_ids, person_word_ids, per_person_author_ids, per_person_word_ids, author_len, word_len, len(person_author_ids), len(person_word_ids)




def _test_gen_pos_and_neg_pair(index, neg_sample):

    # print("iL: ",i)
    # i = index[0]
    pos_pid = index[0]
    neg_idx = index[1]
    # pos_pid = pid_list[i]
    (name, aid) = pid_dict[pos_pid]
    pos_person_pids = name2aidpid[name][aid]

    if (len(neg_idx) != neg_sample):
        print("error: ",len(neg_idx))
        exit()
    # neg person


    pos_str_feature, neg_str_feature = [], []

    neg_person_author_id_list, neg_person_word_id_list = [],[]
    neg_per_person_author_id_list, neg_per_person_word_id_list = [],[]
    neg_author_len_list, neg_word_len_list = [], []
    neg_author_whole_len, neg_word_whole_len = [], []
    neg_person_paper_num = []


    for neg_pid in neg_idx:
        neg_person_pids = name2aidpid[name][neg_pid]

        neg_pairs = (pos_pid, neg_pid)
        
        neg_str_feature.append([])
        

        person_author_ids, person_word_ids, \
        per_person_author_ids, per_person_word_ids, \
        author_len, word_len,\
        whole_author_len, whole_word_len = get_paper_neg_instances(neg_person_pids)

        neg_person_author_id_list.append(person_author_ids)
        neg_person_word_id_list.append(person_word_ids)
        neg_per_person_author_id_list.append(per_person_author_ids)    
        neg_per_person_word_id_list.append(per_person_word_ids)

        neg_person_author_maxlen = np.max(np.array(author_len))
        neg_person_word_maxlen = np.max(np.array(word_len))

        neg_author_len_list.append(neg_person_author_maxlen)
        neg_word_len_list.append(neg_person_word_maxlen)

        neg_author_whole_len.append(whole_author_len)
        neg_word_whole_len.append(whole_word_len)
        assert len(author_len) == len(word_len)

        neg_person_paper_num.append(len(author_len))
    # pos person
    # print(neg_str_feature)
    pos_person_author_id_list, pos_person_word_id_list = [],[]
    pos_per_person_author_id_list, pos_per_person_word_id_list = [],[]
    pos_author_len_list, pos_word_len_list = [], []
    pos_author_whole_len, pos_word_whole_len =[], []
    pos_person_paper_num = []

    pos_pairs = (pos_pid, aid)
    pos_str_feature.append([])
    # print(pos_str_feature)
    # exit()
    person_author_ids, person_word_ids, \
    per_person_author_ids, per_person_word_ids, \
    author_len, word_len, \
    whole_author_len, whole_word_len = get_paper_pos_instances(pos_person_pids, pos_pid)

    pos_person_author_id_list.append(person_author_ids)
    pos_person_word_id_list.append(person_word_ids)
    pos_per_person_author_id_list.append(per_person_author_ids)    
    pos_per_person_word_id_list.append(per_person_word_ids)

    pos_person_author_maxlen = np.max(np.array(author_len))
    pos_person_word_maxlen = np.max(np.array(word_len))

    pos_author_len_list.append(pos_person_author_maxlen)
    pos_word_len_list.append(pos_person_word_maxlen)

    pos_author_whole_len.append(whole_author_len)
    pos_word_whole_len.append(whole_word_len)

    assert len(author_len) == len(word_len)

    pos_person_paper_num.append(len(author_len))


    neg_person_author_id_list.extend(pos_person_author_id_list)
    neg_person_word_id_list.extend(pos_person_word_id_list)
    neg_per_person_author_id_list.extend(pos_per_person_author_id_list)
    neg_per_person_word_id_list.extend(pos_per_person_word_id_list)
    neg_author_len_list.extend(pos_author_len_list)
    neg_word_len_list.extend(pos_word_len_list)
    neg_author_whole_len.extend(pos_author_whole_len)
    neg_word_whole_len.extend(pos_word_whole_len) 
    neg_person_paper_num.extend(pos_person_paper_num)
    neg_str_feature.extend(pos_str_feature)
    


    
    # author_id_list, author_idf_list, word_id_list, word_idf_list, _, _, _,_ ,_ ,_ = pub_feature_dict[pos_pid]
    author_id_list, author_idf_list, word_id_list, word_idf_list= pub_feature_dict[pos_pid]
    # neg_author_id_list, neg_author_idf_list, neg_word_id_list, neg_word_idf_list = pub_feature_dict[neg_pid]

    paper_author_id_list, paper_author_idf_list, paper_word_id_list, paper_word_idf_list = [],[],[],[]

    paper_author_id_list.append(author_id_list)
    # paper_author_id_list.append(neg_author_id_list)

    paper_author_idf_list.append(author_idf_list)
    # paper_author_idf_list.append(neg_author_idf_list)

    paper_word_id_list.append(word_id_list)
    # paper_word_id_list.append(neg_word_id_list)

    paper_word_idf_list.append(word_idf_list)

    paper_author_num = [len(author_id_list)]
    paper_word_num = [len(word_id_list)]

    new_paper_author_id_list = np.repeat(np.array(paper_author_id_list), neg_sample + 1, axis = 0).tolist()
    new_paper_author_idf_list = np.repeat(np.array(paper_author_idf_list), neg_sample + 1, axis = 0).tolist()
    new_paper_word_id_list = np.repeat(np.array(paper_word_id_list), neg_sample + 1, axis = 0).tolist()
    new_paper_word_idf_list = np.repeat(np.array(paper_word_idf_list), neg_sample + 1, axis = 0).tolist()   

    new_paper_author_num = np.repeat(np.array(paper_author_num), neg_sample + 1, axis = 0).tolist()
    new_paper_word_num = np.repeat(np.array(paper_word_num), neg_sample + 1, axis = 0).tolist()
    # print(new_paper_word_num)

    assert len(new_paper_author_id_list) == len(neg_person_author_id_list)


    return neg_person_author_id_list, neg_person_word_id_list, neg_per_person_author_id_list, neg_per_person_word_id_list, neg_author_len_list, neg_word_len_list, neg_author_whole_len, neg_word_whole_len, neg_person_paper_num,\
    new_paper_author_id_list, new_paper_author_idf_list, new_paper_word_id_list, new_paper_word_idf_list, new_paper_author_num, new_paper_word_num, neg_str_feature
       
    



def _add_mask(feature_mask, features, num_max):
    for i in range(len(features)):
        if len(features[i]) <= num_max:
            features[i] = features[i] + [feature_mask] * (num_max + 1 - len(features[i]))
        else:
            features[i] = features[i][:num_max] + [feature_mask]
    return features

def _add_paper_mask(feature_mask, features, num_max, paper_max):
    for i in range(len(features)):
        if len(features[i]) <= num_max:
            features[i] = features[i] + [feature_mask] * (num_max + 1 - len(features[i]))
        else:
            features[i] = features[i][:num_max] + [feature_mask]

    if len(features) <= paper_max:
        paper_mask_len = paper_max - len(features)
        paper_mask_mat = (np.ones((paper_mask_len, num_max + 1), dtype = int) * feature_mask).tolist()

        features.extend(paper_mask_mat)
        # print("11",paper_mask_mat)
        # exit(0)
    else:
        features = features[:paper_max]
        # print("22", features)
        # exit(0)
    return features









    
