from os.path import join
import sys
sys.path.append("..")
import os
import multiprocessing as mp
import random
from datetime import datetime
from utils import data_utils
# from utils import settings
import classifier_settings as settings
from utils import multithread_utils
from collections import defaultdict
import numpy as np
from gensim.models import Word2Vec
import copy
import random
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

start_time = datetime.now()

name2aidpid = None
name2pid = None
pid_dict = None
pid_list = None
pub_feature_dict = None
author_num = None
word_num = None


def generate_data_batches(train_data_file, test_data_file, data_size, sample_num, batch_size, info):

    print("data_size: {} sample_num: {} batch_size: {}".format(data_size, sample_num, batch_size))
    global name2aidpid
    global name2pid
    global name2plist
    global pid_dict
    global pid_list
    global pub_feature_dict
    global author_num
    global word_num
    global index_list
    global pubs_dict

    name2pid = defaultdict(list)
    name2plist = defaultdict(list)
    pid_dict = {}
    pid_list = []

    pubs_dict = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'pubs_raw.json')

    name2aidpid = data_utils.load_json(settings.GLOBAL_DATA_DIR, train_data_file)  
    print('Train #data=%d' % len(name2aidpid))
    for name in name2aidpid:
        aid2pid = name2aidpid[name]
        for aid in aid2pid:
            name2pid[name] += aid2pid[aid]
            name2plist[name].append(aid)
            for pid in aid2pid[aid]:
                # may be twice
                pid_dict[pid] = (name, aid)
                pid_list.append(pid)
        random.shuffle(name2pid[name])
    # random.shuffle(pid_list)
    print("Processed name2aidpid")



    pid_dict_test = {}
    pid_list_test = []

    name2aidpid_test = data_utils.load_json(settings.GLOBAL_DATA_DIR, test_data_file)  
    print('test #data=%d' % len(name2aidpid_test))
    for name in name2aidpid_test:
        aid2pid = name2aidpid_test[name]
        for aid in aid2pid:
            for pid in aid2pid[aid]:
                pid_dict_test[pid] = (name, aid)
                pid_list_test.append(pid)

    pub_feature_dict = data_utils.load_data(settings.EMB_DATA_DIR, "pub_feature.ids")
    print("Loaded pub features")
    # count = 0
    # for i ,j in pub_feature_dict.items():
    #     print("key: ",i)
    #     print('value: ',j)
    #     count+=1
    #     if(count == 6):
    #         exit()

    author_num = len(data_utils.load_data(settings.EMB_DATA_DIR, "author_emb.array"))
    print("#author = %d" %author_num)

    word_num = len(data_utils.load_data(settings.EMB_DATA_DIR, "word_emb.array"))
    print("#Word = %d" %word_num)

    # print("Get test data!!!")
    if info == 'Train':
        print("Get train data!!!")
        print("train_pid_list:{}".format(len(pid_list))) 
        # index_list = np.arange(len(pid_list))
        # random.shuffle(index_list)
        # return get_test_data_batches(min(len(pid_list),data_size), sample_num, batch_size, info)
    elif info == 'Test' or info == 'rank_test':
        print("Get {} data!!!".format(info))
        pid_list = pid_list_test
        pid_dict = pid_dict_test
        print("test_pid_list:{}".format(len(pid_list))) 
        # index_list = np.arange(len(pid_list))
        # random.shuffle(index_list)
        # return get_test_data_batches(min(len(pid_list),data_size), sample_num, batch_size, info)
    else:
        print("NO INFO!!!")
        exit()



    index_list = np.arange(len(pid_list))
    random.shuffle(index_list)
    return get_test_data_batches(min(len(pid_list),data_size), sample_num, batch_size, info)


def pairwise_feature(pair, name, author_dict, pubs_dict):
    feature_list = []
    paper = pair[0] 
    author = pair[1]

    paper_num = len(author_dict[name][author])
    feature_list.append(paper_num)

    author_authors = []
    for doc in author_dict[name][author]:
        if doc != paper:
            doc_dict = pubs_dict[doc[:24]]
            for aut in doc_dict["authors"]:
                author_authors.append(aut["name"].lower().translate(str.maketrans('','','._- ')))
    author_authors = list(set(author_authors))


    paper_authors = []
    for aut in pubs_dict[paper[:24]]["authors"]:
        paper_authors.append(aut["name"].lower().translate(str.maketrans('','','._- ')))

    common_name = [x for x in paper_authors if x in author_authors]

    feature_list.append(len(common_name))
    feature_list.append(len(common_name)/len(author_authors))
    feature_list.append(len(common_name)/len(paper_authors))

    author_orgs = []
    for doc in author_dict[name][author]:
        if doc != paper:
            doc_dict = pubs_dict[doc[:24]]
            for aut in doc_dict['authors']:
                if aut['name'].lower().translate(str.maketrans('','','._- ')) == name.lower().translate(str.maketrans('','','._- ')) and 'org' in aut.keys():
                        author_orgs.append(aut['org'].lower())

    paper_dict = pubs_dict[paper[:24]]
    for aut in paper_dict['authors']:
        if aut['name'].lower().translate(str.maketrans('','','._- ')) == name.lower().translate(str.maketrans('','','._- ')):
            if 'org' in aut.keys():
                paper_org = aut['org'].lower()
            else:
                paper_org = ''

    org_num = author_orgs.count(paper_org)
    feature_list.append(org_num)
    feature_list.append(org_num/(len(author_orgs)+0.001))

    author_orgs = list(set(author_orgs))
    author_org = ' '.join(author_orgs)

    if len(paper_org) < 2 or len(author_org) < 2:
        feature_list.append(-1)
        feature_list.append(-1)
    else:
        corus4 = [author_org, paper_org]
        a1 = TfidfVectorizer()
        b1 = a1.fit_transform(corus4)
        c1 = [b1.toarray()[0],b1.toarray()[1]]
        feature_list.append(cosine_similarity(c1)[0][1])
        set1 = set(author_org.split(' '))
        set2 = set(paper_org.split(' '))
        jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
        feature_list.append(jaccard)

    author_venues = []
    for doc in author_dict[name][author]:
        if doc != paper:
            doc_dict = pubs_dict[doc[:24]]
            author_venues.append(doc_dict["venue"].lower())
    
    paper_venue = pubs_dict[paper[:24]]["venue"].lower()

    venue_num = author_venues.count(paper_venue)
    feature_list.append(venue_num)
    feature_list.append(venue_num/len(author_venues))


    author_venues = list(set(author_venues))
    feature_list.append(len(author_venues))
    author_venue = ' '.join(author_venues)

    corus3 = [author_venue, paper_venue]
    a1 = TfidfVectorizer()
    b1 = a1.fit_transform(corus3)
    c1 = [b1.toarray()[0],b1.toarray()[1]]
    feature_list.append(cosine_similarity(c1)[0][1])

    set1 = set(author_venue.split(' '))
    set2 = set(paper_venue.split(' '))
    jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
    feature_list.append(jaccard)

    author_keyword = ''
    author_keywords = []
    for doc in author_dict[name][author]:
        if doc != paper:
            doc_dict = pubs_dict[doc[:24]]
            if "keywords" in doc_dict:
                author_keywords.extend(doc_dict["keywords"])
    author_keyword = ' '.join(author_keywords)

    paper_keyword = ''
    paper_keywords = []
    if "keywords" in pubs_dict[paper[:24]]:
        paper_keywords = pubs_dict[paper[:24]]["keywords"]
    paper_keyword = ' '.join(paper_keywords)

    if len(paper_keyword) < 2 or len(author_keyword) < 2:
        feature_list.append(-1)
        feature_list.append(-1)
    else:   
        corus1 = [author_keyword, paper_keyword]
        a1 = TfidfVectorizer()
        b1 = a1.fit_transform(corus1)
        c1 = [b1.toarray()[0],b1.toarray()[1]]
        feature_list.append(cosine_similarity(c1)[0][1])

        set1 = set(author_keyword.split(' '))
        set2 = set(paper_keyword.split(' '))
        jaccard1 = len(set1.intersection(set2)) / len(set1.union(set2))
        feature_list.append(jaccard1)

    author_title = ''
    author_titles = []
    for doc in author_dict[name][author]:
        if doc != paper:
            doc_dict = pubs_dict[doc[:24]]
            author_titles.append(doc_dict["title"])
    author_title = ' '.join(author_titles)

    paper_title = pubs_dict[paper[:24]]["title"]

    corus2 = [author_title, paper_title]
    a1 = TfidfVectorizer()
    b1 = a1.fit_transform(corus2)
    c1 = [b1.toarray()[0],b1.toarray()[1]]
    feature_list.append(cosine_similarity(c1)[0][1])

    set1 = set(author_title.split(' '))
    set2 = set(paper_title.split(' '))
    jaccard2 = len(set1.intersection(set2)) / len(set1.union(set2))
    feature_list.append(jaccard2)

    return feature_list 

def get_train_valid_index(data_size, samples, train_index):
    print(len(pid_list))
    # train_index = pickle.load(open('./train_valid_index.p', "rb"))
    # print(len(train_index))
    # exit()
    # valid_index = set()
    # faker_index = set()
    total_index = []
    total_num = 0

    _1valid_index =random.sample(train_index, data_size)

    for ins in _1valid_index:
        index = ins[0]
        neg_list = ins[1]
        pos_pid = pid_list[index]
        (name, aid) = pid_dict[pos_pid]
        pos_person_pids = name2aidpid[name][aid]

         
        # sample_list = name2plist[name]

        # copy_sample_list = copy.deepcopy(sample_list)
        # copy_sample_list.remove(aid)
        # random.shuffle(copy_sample_list)

        # neg_num = 0
        # neg_list= []
        # for pid in copy_sample_list:
            
        #     if(len(name2aidpid[name][pid]) > 1):
        #         neg_num += 1
        #         neg_list.append(pid)
        #     if (neg_num == (samples - 1)):
        #         break
        # if (neg_num != (samples-1)):
        #     continue
        # valid_index.append(i)
        total_index.append((index,samples,neg_list))
        total_num += 1      
    print("Train pos num: {}/ {}".format(len(total_index), total_num))

    if(len(total_index) != data_size):
        print("not enough train_nums: {} / {}".format(len(total_index), data_size))
        exit()
    random.shuffle(total_index)
    # print(total_index)
    return total_index

def get_valid_index(data_size, samples, test_index):
    print(len(pid_list))

    total_index = []
    total_num = 0

    return_list = []
    # _0valid_index =random.sample(test_index, data_size)

    for ins in test_index:
        index = ins[0]
        neg_list = ins[1]
        pos_pid = pid_list[index]
        (name, aid) = pid_dict[pos_pid]
        pos_person_pids = name2aidpid[name][aid]

        
        total_index.append((index,samples,neg_list))
        return_list.append((pos_pid, aid, name, neg_list))
        total_num += 1      
    print("test pos num: {}/ {}".format(len(total_index), total_num))

    assert len(total_index) == len(return_list)
    if(len(total_index) != data_size):
        print("not enough train_nums: {} / {}".format(len(total_index), data_size))
        exit()
    # random.shuffle(total_index)
    return total_index, return_list
    # valid_index = set()
    # faker_index = set()
    # total_index = []
    # total_num = 0
    # for i in index_list:
    #     pos_pid = pid_list[i]
    #     (name, aid) = pid_dict[pos_pid]
    #     pos_person_pids = name2aidpid[name][aid]

    #     sample_list = name2plist[name]

    #     if (len(pos_person_pids) <=1) :
    #         continue

    #     if (len(sample_list) - 1) < (samples - 1):
    #         continue

    #     copy_sample_list = copy.deepcopy(sample_list)
    #     copy_sample_list.remove(aid)
    #     random.shuffle(copy_sample_list)
    #     neg_num = 0
    #     neg_list = []
    #     for pid in copy_sample_list:
            
    #         if(len(name2aidpid[name][pid]) > 1):
    #             neg_num += 1
    #             neg_list.append(pid)
    #         if (neg_num == (samples - 1)):
    #             break
    #     if (neg_num != (samples-1)):
    #         continue
    #     # valid_index.append(i)
    #     total_index.append((i, samples, neg_list))
    #     total_num += 1
        


    #     if(len(total_index) == data_size):
    #         # print(i)
    #         break
    # # if len(set(faker_index) & set(valid_index)) != 0 :
    # #     print("faker&valid error!")
    # if(len(total_index) != data_size):
    #     print("not enough train_nums: {} / {}".format(len(total_index), data_size))
    #     exit()
    # random.shuffle(total_index)

def get_test_valid_index(data_size, samples):
    print(len(pid_list))
    # valid_index = set()
    # faker_index = set()
    total_index = []
    total_num = 0
    for i in index_list:
        pos_pid = pid_list[i]
        (name, aid) = pid_dict[pos_pid]
        pos_person_pids = name2aidpid[name][aid]

        sample_list = name2plist[name]

        if (len(pos_person_pids) <= 1) :
            continue

        if (len(sample_list) - 1) < (samples - 1):
            continue

        copy_sample_list = copy.deepcopy(sample_list)
        copy_sample_list.remove(aid)
        random.shuffle(copy_sample_list)
        neg_num = 0
        neg_list = []
        for pid in copy_sample_list:
            
            if(len(name2aidpid[name][pid]) > 1):
                neg_num += 1
                neg_list.append(pid)
            if (neg_num == (samples - 1)):
                break
        if (neg_num != (samples-1)):
            continue
        # valid_index.append(i)
        total_index.append((i, samples, neg_list))
        total_num += 1
        


        if(len(total_index) == data_size):
            # print(i)
            break
    # if len(set(faker_index) & set(valid_index)) != 0 :
    #     print("faker&valid error!")
    if(len(total_index) != data_size):
        print("not enough train_nums: {} / {}".format(len(total_index), data_size))
        exit()
    random.shuffle(total_index)
    return total_index


def get_test_data_batches(data_size, sample_num, batch_size, info):
    print("test_instances: {}".format(sample_num))
    if(info == 'Train'):
        # train_index = pickle.load(open('./godtw_train_valid_index.p', "rb"))
        # print("Train Info!")
        # total_index = get_train_valid_index(int(data_size/2), sample_num, train_index)
        # print("train len_valid_index:",len(total_index))
        # pickle.dump(total_index, open("./cl_godtw_train_valid_index.p", "wb"))
        total_index = pickle.load(open('./cl_godtw_train_valid_index.p', "rb"))
        print("train_total_index:",len(total_index))
        # exit()
    # elif (info == 'Test'):
    #     # total_index = pickle.load(open('./cl_test_valid_index.p', "rb"))
    #     print("Test Info!")
    #     # total_index = get_test_valid_index(int(data_size/2), sample_num)
    #     # print("test len_valid_index:",len(total_index))
    #     # pickle.dump(total_index, open("./cl_godtw_test_valid_index.p", "wb"))
    #     total_index = pickle.load(open('./cl_godtw_test_valid_index.p', "rb"))
    #     # exit()
    elif (info == 'rank_test'):
        print("rank_test info!")
        test_index = pickle.load(open('./godtw_test_valid_index.p', "rb"))[:800]
        total_index, return_list = get_valid_index(data_size, sample_num, test_index)
        print("rank test len_valid_index:",len(total_index))
        pickle.dump(return_list, open("./print_test_result/return_list.p", "wb"))

    else:
        print("Error info!")
        exit()
    print("total_index: ",len(total_index))
    # print(faker_index)
    # exit()
    res = multithread_utils.processed_by_multi_thread(_test_gen_pos_and_neg_pair, total_index)
    # exit()
    person_author_ids, person_word_ids, per_person_author_ids, per_person_word_ids, author_len, word_len, person_papers = [], [], [], [], [], [], [] 
    new_paper_author_ids, new_paper_author_idfs, new_paper_word_ids, new_paper_word_idfs, new_paper_authors, new_paper_words, pos_str_features  = [], [], [], [], [], [], []
    labels = []

    b_person_author_ids, b_person_word_ids, b_per_person_author_ids, b_per_person_word_ids, b_person_papers = [], [], [], [], []
    b_new_paper_author_ids, b_new_paper_author_idfs, b_new_paper_word_ids, b_new_paper_word_idfs, b_new_paper_authors, b_new_paper_words = [], [], [], [], [], []
    b_labels = []
    for i, t in enumerate(res):
        if i % 1000 == 0:
            print(i, datetime.now()-start_time)
        if t == None:
            continue

        person_author_id_list, person_word_id_list, per_person_author_id_list, per_person_word_id_list, author_len_list, word_len_list, person_paper_num = t[0], t[1], t[2], t[3], t[4],t[5],t[6]
        new_paper_author_id_list, new_paper_author_idf_list, new_paper_word_id_list, new_paper_word_idf_list, new_paper_author_num, new_paper_word_num, label_list = t[7],t[8],t[9],t[10],t[11],t[12], t[13]
        pos_str_features_list = t[14]
        # total_ins_dict = {**total_ins_dict, **t[14]}
        # instance = t[15]

        # print("ttt:",paper_num)
        # print(label)
        # print(neg_person_paper_num)
        # print(new_pos_person_paper_num)
        # print(new_paper_author_num)
        # exit(0)
        # print(len(person_author_id_list))
        # exit()
        person_author_ids.extend(person_author_id_list) 
        person_word_ids.extend(person_word_id_list)
        per_person_author_ids.extend(per_person_author_id_list)
        per_person_word_ids.extend(per_person_word_id_list)
        author_len.extend(author_len_list)
        word_len.extend(word_len_list)
        person_papers.extend(person_paper_num)

        new_paper_author_ids.extend(new_paper_author_id_list)
        new_paper_author_idfs.extend(new_paper_author_idf_list)
        new_paper_word_ids.extend(new_paper_word_id_list)
        new_paper_word_idfs.extend(new_paper_word_idf_list)
        new_paper_authors.extend(new_paper_author_num)
        new_paper_words.extend(new_paper_word_num)
        labels.extend(label_list)
        pos_str_features.extend(pos_str_features_list)
        # instance_list.extend(instance)

        # print(person_papers)
        # print(new_paper_words)
        # exit()
        if(len(person_author_ids) % (sample_num) != 0):
            print("NEG NUM DO NOT MATCH!!!")
            exit()
        # HERE HERE s
        if(len(person_author_ids) == batch_size):

            new_paper_author_ids = _add_mask(author_num, new_paper_author_ids, min(max(new_paper_authors),settings.MAX_AUTHOR))
            new_paper_author_idfs = _add_mask(0, new_paper_author_idfs, min(max(new_paper_authors),settings.MAX_AUTHOR))
            new_paper_word_ids = _add_mask(word_num, new_paper_word_ids, min(max(new_paper_words),settings.MAX_WORD))
            new_paper_word_idfs = _add_mask(0, new_paper_word_idfs, min(max(new_paper_words),settings.MAX_WORD))

            person_author_ids = _add_mask(author_num, person_author_ids, min(max(author_len),settings.MAX_AUTHOR))
            person_word_ids = _add_mask(word_num, person_word_ids, min(max(word_len),settings.MAX_WORD))

            # print(np.array(new_paper_author_idfs).shape)
            # exit()

            # print(len(per_person_word_ids))
            # print(len(word_len))
            # exit(0)
            tmp_author_id, tmp_word_id = [], []

            pad_paper_num = np.max(np.array(person_papers))
            # print(np.array(neg_person_papers))

            pad_author = np.max(np.array(author_len))
            # print(np.array(neg_author_len))

            pad_word = np.max(np.array(word_len))


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

            b_labels.append(labels)
            # b_instance_list.append(instance_list)

            # print(np.array(b_new_pos_per_person_author_ids).shape)
            # print(b_labels)


            person_author_ids, person_word_ids, per_person_author_ids, per_person_word_ids, author_len, word_len, person_papers = [], [], [], [], [], [], [] 
            new_paper_author_ids, new_paper_author_idfs, new_paper_word_ids, new_paper_word_idfs, new_paper_authors, new_paper_words = [], [], [], [], [], []
            labels= []


    if person_author_ids:
        # print("NO MORE PERSON IDS")
        # exit()
        new_paper_author_ids = _add_mask(author_num, new_paper_author_ids, min(max(new_paper_authors),settings.MAX_AUTHOR))
        new_paper_author_idfs = _add_mask(0, new_paper_author_idfs, min(max(new_paper_authors),settings.MAX_AUTHOR))
        new_paper_word_ids = _add_mask(word_num, new_paper_word_ids, min(max(new_paper_words),settings.MAX_WORD))
        new_paper_word_idfs = _add_mask(0, new_paper_word_idfs, min(max(new_paper_words),settings.MAX_WORD))

        person_author_ids = _add_mask(author_num, person_author_ids, min(max(author_len),settings.MAX_AUTHOR))
        person_word_ids = _add_mask(word_num, person_word_ids, min(max(word_len),settings.MAX_WORD))

        # print(np.array(new_paper_author_idfs).shape)
        # exit()

        # print(len(per_person_word_ids))
        # print(len(word_len))
        # exit(0)
        tmp_author_id, tmp_word_id = [], []

        pad_paper_num = np.max(np.array(person_papers))
        # print(np.array(neg_person_papers))

        pad_author = np.max(np.array(author_len))
        # print(np.array(neg_author_len))

        pad_word = np.max(np.array(word_len))


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
        b_labels.append(labels)
        # b_instance_list.append(instance_list)

        # print(b_labels)

    pos_str_features = np.array(pos_str_features)
    # neg_str_features = np.array(neg_str_features)

    pos_str_backup = copy.deepcopy(pos_str_features)
    # neg_str_backup = copy.deepcopy(neg_str_features)

    pos_str_features[pos_str_features == -1] = 0
    # neg_str_features[neg_str_features == -1] = 0

    # total_str = np.concatenate((pos_str_features, neg_str_features), axis=0)
    
    total_mean = np.mean(pos_str_features, 0)

    for i in [6,7,13,14]:
        pos_str_backup[:,i][pos_str_backup[:,i] == -1] = total_mean[i]

        # neg_str_backup[:,i][neg_str_backup[:,i] == -1] = total_mean[i]

    # assert len(pos_str_backup) == len(neg_str_backup)
    b_pos_str_feature = []

    data_len = len(pos_str_backup)

    for i in range(0, data_len, batch_size):
        b_pos_str_feature.append(pos_str_backup[i:min(i + batch_size, data_len)]) 
        # b_neg_str_feature.append(neg_str_backup[i:min(i + settings.BATCH_SIZE, data_len)])

    assert len(b_pos_str_feature) == len(b_person_author_ids)

    return b_person_author_ids, b_person_word_ids, b_per_person_author_ids, b_per_person_word_ids, b_person_papers,\
    b_new_paper_author_ids, b_new_paper_author_idfs, b_new_paper_word_ids, b_new_paper_word_idfs,b_labels, b_pos_str_feature







    # return b_person_author_ids, b_person_author_idfs, b_person_author_nums, b_person_word_ids, b_person_word_idfs, b_person_word_nums,\
    #     b_paper_author_ids, b_paper_author_idfs, b_paper_author_nums, b_paper_word_ids, b_paper_word_idfs, b_paper_word_nums,\
    #     b_person_pids_list, b_pid_list, b_labels,\
    #     b_per_person_author_ids, b_per_person_author_idfs, b_per_person_word_ids, b_per_person_word_idfs, b_per_person_paper_num

def get_paper_neg_instances(pid_list):

    person_author_ids, person_word_ids = [], []
    per_person_author_ids, per_person_word_ids = [], [] 
    author_len, word_len = [], []

    for pid in pid_list:
        
        author_id_list, author_idf_list, word_id_list, word_idf_list = pub_feature_dict[pid]
        person_author_ids += author_id_list
        person_word_ids += word_id_list

        per_person_author_ids.append(author_id_list)
        per_person_word_ids.append(word_id_list)

        # assert (len(author_id_list) == len(author_idf_list)) & (len(word_id_list) == len(word_idf_list))

        author_len.extend([len(author_id_list)])
        word_len.extend([len(word_id_list)])

    return person_author_ids, person_word_ids, per_person_author_ids, per_person_word_ids, author_len, word_len

def get_paper_pos_instances(pid_list, self_id):

    person_author_ids, person_word_ids = [], []
    per_person_author_ids, per_person_word_ids = [], [] 
    author_len, word_len = [], []

    for pid in pid_list:

        if(pid != self_id):
        
            author_id_list, author_idf_list, word_id_list, word_idf_list = pub_feature_dict[pid]
            person_author_ids += author_id_list
            person_word_ids += word_id_list

            per_person_author_ids.append(author_id_list)
            per_person_word_ids.append(word_id_list)

            # assert (len(author_id_list) == len(author_idf_list)) & (len(word_id_list) == len(word_idf_list))

            author_len.extend([len(author_id_list)])
            word_len.extend([len(word_id_list)])

    return person_author_ids, person_word_ids, per_person_author_ids, per_person_word_ids, author_len, word_len

def _test_gen_pos_and_neg_pair(index):

    # print("iL: ",index)
    i = index[0]
    tag = '1'
    sample_num = index[1]
    neg_list = index[2]
    pos_pid = pid_list[i]
    (name, aid) = pid_dict[pos_pid]
    pos_person_pids = name2aidpid[name][aid]

    # sample_list = name2plist[name]
    # # print("aid:{} name:{} sample_list:{}".format(aid, name, sample_list))
    # copy_sample_list = copy.deepcopy(sample_list)
    # copy_sample_list.remove(aid)
    # random.shuffle(copy_sample_list)
    # print("tag:", tag)
    if tag == '1':
        threshold = sample_num - 1
    elif tag == '0':
        threshold = sample_num
    else:
        print("NO tag")
        exit()


    # neg_num = 0
    # neg_idx = []
    # for pid in copy_sample_list:
        
    #     if(len(name2aidpid[name][pid]) > 1):
    #         neg_idx.append(pid)

    #         neg_num += 1
    #     if (neg_num == threshold):
    #         break

    if (len(neg_list) != threshold):
        print("error: ",len(neg_list))
        exit()

    pos_str_feature, neg_str_feature = [], []

    # neg person
    neg_person_author_id_list, neg_person_word_id_list = [],[]
    neg_per_person_author_id_list, neg_per_person_word_id_list = [],[]
    neg_author_len_list, neg_word_len_list = [], []
    neg_person_paper_num = []
    
    # ins_dict = {}
    # ins_dict[pos_pid] = {}
    # ins_dict[pos_pid]['label'] = tag
    # ins_dict[pos_pid]['name'] = name
    # ins_dict[pos_pid]['author_id'] = aid
    # ins_dict[pos_pid]['neg_id'] = []


    for neg_pid in neg_list:
        neg_person_pids = name2aidpid[name][neg_pid]

        neg_pairs = (pos_pid, neg_pid)
        neg_str_feature.append(pairwise_feature(neg_pairs, name, name2aidpid, pubs_dict))

        person_author_ids, person_word_ids, per_person_author_ids, per_person_word_ids, author_len, word_len = get_paper_neg_instances(neg_person_pids)

        neg_person_author_id_list.append(person_author_ids)
        neg_person_word_id_list.append(person_word_ids)
        neg_per_person_author_id_list.append(per_person_author_ids)    
        neg_per_person_word_id_list.append(per_person_word_ids)

        neg_person_author_maxlen = np.max(np.array(author_len))
        neg_person_word_maxlen = np.max(np.array(word_len))

        neg_author_len_list.append(neg_person_author_maxlen)
        neg_word_len_list.append(neg_person_word_maxlen)
        
        assert len(author_len) == len(word_len)

        neg_person_paper_num.append(len(author_len))
        # ins_dict[pos_pid]['neg_id'].append(neg_pid)
    # pos person
    pos_person_author_id_list, pos_person_word_id_list = [],[]
    pos_per_person_author_id_list, pos_per_person_word_id_list = [],[]
    pos_author_len_list, pos_word_len_list = [], []
    pos_person_paper_num = []
    label = []

    if tag == '1':
        pos_pairs=(pos_pid, aid)

        pos_str_feature.append(pairwise_feature(pos_pairs, name, name2aidpid, pubs_dict))
        person_author_ids, person_word_ids, per_person_author_ids, per_person_word_ids, author_len, word_len = get_paper_pos_instances(pos_person_pids, pos_pid)

        pos_person_author_id_list.append(person_author_ids)
        pos_person_word_id_list.append(person_word_ids)
        pos_per_person_author_id_list.append(per_person_author_ids)    
        pos_per_person_word_id_list.append(per_person_word_ids)

        pos_person_author_maxlen = np.max(np.array(author_len))
        pos_person_word_maxlen = np.max(np.array(word_len))

        pos_author_len_list.append(neg_person_author_maxlen)
        pos_word_len_list.append(neg_person_word_maxlen)

        assert len(author_len) == len(word_len)

        pos_person_paper_num.append(len(author_len))

        # expend to neg_num
        pos_person_author_id_list.extend(neg_person_author_id_list)
        pos_person_word_id_list.extend(neg_person_word_id_list)
        pos_per_person_author_id_list.extend(neg_per_person_author_id_list)
        pos_per_person_word_id_list.extend(neg_per_person_word_id_list)
        pos_author_len_list.extend(neg_author_len_list)
        pos_word_len_list.extend(neg_word_len_list)
        pos_person_paper_num.extend(neg_person_paper_num)
        pos_str_feature.extend(neg_str_feature)
        label.append(1)

        # print(pos_person_paper_num)
        # print(len(pos_per_person_word_id_list))
        # pos person
        # print(aid)
        # print(len(new_pos_person_word_id_list))
        # print(len(new_pos_per_person_word_id_list))
        # print(new_pos_author_len_list)
        # print(new_pos_word_len_list)
        # print(new_pos_person_paper_num)
    elif tag == '0':
        pos_person_author_id_list.extend(neg_person_author_id_list)
        pos_person_word_id_list.extend(neg_person_word_id_list)
        pos_per_person_author_id_list.extend(neg_per_person_author_id_list)
        pos_per_person_word_id_list.extend(neg_per_person_word_id_list)
        pos_author_len_list.extend(neg_author_len_list)
        pos_word_len_list.extend(neg_word_len_list)
        pos_person_paper_num.extend(neg_person_paper_num)
        label.append(0)
        # print(pos_person_paper_num)
        # print(len(pos_per_person_word_id_list))
        # pos person
        # print(aid)
        # print(len(new_pos_person_word_id_list))
        # print(len(new_pos_per_person_word_id_list))
        # print(new_pos_author_len_list)
        # print(new_pos_word_len_list)
        # print(new_pos_person_paper_num)

    # instance = []
    # instance.append(pos_pid)

    author_id_list, author_idf_list, word_id_list, word_idf_list = pub_feature_dict[pos_pid]
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

    new_paper_author_id_list = np.repeat(np.array(paper_author_id_list), sample_num, axis = 0).tolist()
    new_paper_author_idf_list = np.repeat(np.array(paper_author_idf_list), sample_num, axis = 0).tolist()
    new_paper_word_id_list = np.repeat(np.array(paper_word_id_list), sample_num, axis = 0).tolist()
    new_paper_word_idf_list = np.repeat(np.array(paper_word_idf_list), sample_num, axis = 0).tolist()   

    new_paper_author_num = np.repeat(np.array(paper_author_num), sample_num, axis = 0).tolist()
    new_paper_word_num = np.repeat(np.array(paper_word_num), sample_num, axis = 0).tolist()
    # print(len(new_paper_word_num))



    return pos_person_author_id_list, pos_person_word_id_list, pos_per_person_author_id_list, pos_per_person_word_id_list, pos_author_len_list, pos_word_len_list, pos_person_paper_num,\
    new_paper_author_id_list, new_paper_author_idf_list, new_paper_word_id_list, new_paper_word_idf_list, new_paper_author_num, new_paper_word_num, label, pos_str_feature
    # return neg_person_author_id_list, neg_person_word_id_list, neg_per_person_author_id_list, neg_per_person_word_id_list, neg_author_len_list, neg_word_len_list, neg_person_paper_num,\
    # new_pos_person_author_id_list, new_pos_person_word_id_list, new_pos_per_person_author_id_list, new_pos_per_person_word_id_list, new_pos_author_len_list, new_pos_word_len_list, new_pos_person_paper_num,\
    # new_paper_author_id_list, new_paper_author_idf_list, new_paper_word_id_list, new_paper_word_idf_list, new_paper_author_num, new_paper_word_num
    



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










    
