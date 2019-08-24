from os.path import join
import sys
sys.path.append("..")
import os
import multiprocessing as mp
import random
from datetime import datetime
from utils import data_utils
from utils import settings
from utils import multithread_utils
from collections import defaultdict
import numpy as np
from gensim.models import Word2Vec
import copy
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


def generate_data_batches(train_data_file, test_data_file, data_size, info):

    global name2aidpid
    global name2aidpid_test
    global name2pid
    global name2plist
    global pid_dict
    global pid_list
    global pub_feature_dict
    global author_num
    global word_num
    global neg_sample
    global index_list
    global pubs_dict


    #For train 
    neg_sample = settings.NEG_SAMPLE
    name2pid = defaultdict(list)
    name2plist = defaultdict(list)
    pid_dict = {}
    pid_list = []


    pubs_dict = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'pubs_raw.json')

    name2aidpid = data_utils.load_json(settings.GLOBAL_DATA_DIR, train_data_file)  
    print('train #data=%d' % len(name2aidpid))
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

    # # random.shuffle(pid_list)
    # print(pid_list[:10])
    # exit()
    print("Processed name2aidpid")

    # pub_feature_dict = data_utils.load_data('/root/zhangjing/NA/emb', "pub_feature.ids")
    pub_feature_dict = data_utils.load_data(settings.EMB_DATA_DIR, "pub_feature.ids")
    print("Loaded pub features")
    # count = 0
    # for i ,j in pub_feature_dict.items():
    #     print("key: ",i)
    #     print('value: ',j)
    #     count+=1
    #     if(count == 6):
    #         exit()

    # author_num = len(data_utils.load_data('/root/zhangjing/NA/emb/', "author_emb.array"))
    author_num = len(data_utils.load_data(settings.EMB_DATA_DIR, "author_emb.array"))
    print("#author = %d" %author_num)

    # word_num = len(data_utils.load_data('/root/zhangjing/NA/emb/', "title_emb.array"))
    word_num = len(data_utils.load_data(settings.EMB_DATA_DIR, "word_emb.array"))
    print("#Word = %d" %word_num)
    if info == 'TRAIN':
        print("Get train data!!!")
        print("train_pid_list:{}".format(len(pid_list))) 
        index_list = np.arange(len(pid_list))
        random.shuffle(index_list)
        return get_data_batches(min(len(pid_list),data_size))
    elif info == 'TEST':
        print("Get test data!!!")
        pid_list = pid_list_test
        pid_dict = pid_dict_test
        print("test_pid_list:{}".format(len(pid_list))) 
        index_list = np.arange(len(pid_list))
        random.shuffle(index_list)
        return get_test_data_batches(min(len(pid_list), data_size))
    else:
        print("NO INFO!!!")
        exit()

def get_valid_index(data_size, samples):
    print(len(pid_list))
    valid_index = []
    for i in index_list:
        pos_pid = pid_list[i]
        (name, aid) = pid_dict[pos_pid]
        pos_person_pids = name2aidpid[name][aid]

        sample_list = name2plist[name]

        if (len(pos_person_pids) <=1) :
            continue

        if (len(sample_list) - 1) < samples:
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
            if (neg_num == samples):
                break

        if (neg_num != samples):
            continue
        valid_index.append((i, neg_list))
        # valid_index.append(i)
        if(len(valid_index) == data_size):
            # print(i)
            break
    # print(valid_index)
    return valid_index

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


def get_test_data_batches(data_size):
    valid_index = get_valid_index(data_size, settings.TEST_SAMPLE)
    print("test len_valid_index:",len(valid_index))
    pickle.dump(valid_index, open("./godtw_test_valid_index.p", "wb"))
    # print(valid_index)
    # exit()
    res = multithread_utils.processed_by_multi_thread(_test_gen_pos_and_neg_pair, valid_index)

    person_author_ids, person_word_ids, per_person_author_ids, per_person_word_ids, author_len, word_len, person_papers = [], [], [], [], [], [], [] 
    new_paper_author_ids, new_paper_author_idfs, new_paper_word_ids, new_paper_word_idfs, new_paper_authors, new_paper_words, pos_str_features = [], [], [], [], [], [], []

    b_person_author_ids, b_person_word_ids, b_per_person_author_ids, b_per_person_word_ids, b_person_papers = [], [], [], [], []
    b_new_paper_author_ids, b_new_paper_author_idfs, b_new_paper_word_ids, b_new_paper_word_idfs, b_new_paper_authors, b_new_paper_words = [], [], [], [], [], []
    for i, t in enumerate(res):
        if i % 1000 == 0:
            print(i, datetime.now()-start_time)
        if t == None:
            continue

        person_author_id_list, person_word_id_list, per_person_author_id_list, per_person_word_id_list, author_len_list, word_len_list, person_paper_num = t[0], t[1], t[2], t[3], t[4],t[5],t[6]
        new_paper_author_id_list, new_paper_author_idf_list, new_paper_word_id_list, new_paper_word_idf_list, new_paper_author_num, new_paper_word_num = t[7],t[8],t[9],t[10],t[11],t[12]
        pos_str_features_list = t[13]
        # print("ttt:",paper_num)

        # print(neg_person_paper_num)
        # print(new_pos_person_paper_num)
        # print(new_paper_author_num)
        # exit(0)

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
        pos_str_features.extend(pos_str_features_list)
        # print(person_papers)
        # print(new_paper_words)
        # exit()
        if(len(person_author_ids) % (settings.TEST_SAMPLE + 1) != 0):
            print("NEG NUM DO NOT MATCH!!!")
            exit()
        # HERE HERE s
        if(len(person_author_ids) == settings.TEST_BATCH_SIZE):

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



            # print(np.array(b_new_pos_per_person_author_ids).shape)



            person_author_ids, person_word_ids, per_person_author_ids, per_person_word_ids, author_len, word_len, person_papers = [], [], [], [], [], [], [] 
            new_paper_author_ids, new_paper_author_idfs, new_paper_word_ids, new_paper_word_idfs, new_paper_authors, new_paper_words = [], [], [], [], [], []


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

    for i in range(0, data_len, settings.TEST_BATCH_SIZE):
        b_pos_str_feature.append(pos_str_backup[i:min(i + settings.TEST_BATCH_SIZE, data_len)]) 
        # b_neg_str_feature.append(neg_str_backup[i:min(i + settings.BATCH_SIZE, data_len)])

    assert len(b_pos_str_feature) == len(b_person_author_ids)

    return b_person_author_ids, b_person_word_ids, b_per_person_author_ids, b_per_person_word_ids, b_person_papers,\
    b_new_paper_author_ids, b_new_paper_author_idfs, b_new_paper_word_ids, b_new_paper_word_idfs, b_pos_str_feature





 
def get_data_batches(data_size):
    valid_index = get_valid_index(data_size, settings.NEG_SAMPLE)
    print("train len_valid_index:",len(valid_index))
    # print('11111')
    # exit()
    # pickle.dump(valid_index, open("./godtw_train_valid_index.p", "wb"))
    # valid_index = pickle.load(open('./train_valid_index.p', "rb"))

    print("train len: ", len(valid_index))
    # exit()
    res = multithread_utils.processed_by_multi_thread(_gen_pos_and_neg_pair, valid_index)

    neg_person_author_ids, neg_person_word_ids, neg_per_person_author_ids, neg_per_person_word_ids, neg_author_len, neg_word_len, neg_person_papers = [], [], [], [], [], [], [] 
    new_pos_person_author_ids, new_pos_person_word_ids, new_pos_per_person_author_ids, new_pos_per_person_word_ids, new_pos_author_len, new_pos_word_len, new_pos_person_papers = [], [], [], [], [], [], []
    new_paper_author_ids, new_paper_author_idfs, new_paper_word_ids, new_paper_word_idfs, new_paper_authors, new_paper_words, pos_str_features, neg_str_features = [], [], [], [], [], [], [], []

    b_neg_person_author_ids, b_neg_person_word_ids, b_neg_per_person_author_ids, b_neg_per_person_word_ids, b_neg_person_papers = [], [], [], [], []
    b_new_pos_person_author_ids, b_new_pos_person_word_ids, b_new_pos_per_person_author_ids, b_new_pos_per_person_word_ids,b_new_pos_person_papers = [], [], [], [], []
    b_new_paper_author_ids, b_new_paper_author_idfs, b_new_paper_word_ids, b_new_paper_word_idfs, b_new_paper_authors, b_new_paper_words = [], [], [], [], [], []
    for i, t in enumerate(res):
        if i % 1000 == 0:
            print(i, datetime.now()-start_time)
        if t == None:
            continue

        neg_person_author_id_list, neg_person_word_id_list, neg_per_person_author_id_list, neg_per_person_word_id_list, neg_author_len_list, neg_word_len_list, neg_person_paper_num = t[0], t[1], t[2], t[3], t[4],t[5],t[6]
        new_pos_person_author_id_list, new_pos_person_word_id_list, new_pos_per_person_author_id_list, new_pos_per_person_word_id_list, new_pos_author_len_list, new_pos_word_len_list, new_pos_person_paper_num = t[7], t[8], t[9], t[10], t[11],t[12], t[13]
        new_paper_author_id_list, new_paper_author_idf_list, new_paper_word_id_list, new_paper_word_idf_list, new_paper_author_num, new_paper_word_num = t[14],t[15],t[16],t[17],t[18],t[19]
        pos_str_feature_list, neg_str_feature_list = t[20], t[21]
        assert len(pos_str_features) == len(neg_str_features)
        # print(neg_str_feature_list)
        # print('---------------')
        # print(pos_str_feature_list)
        # exit()
        # print("ttt:",paper_num)

        # print(neg_person_paper_num)
        # print(new_pos_person_paper_num)
        # print(new_paper_author_num)
        # exit(0)

        neg_person_author_ids.extend(neg_person_author_id_list) 
        neg_person_word_ids.extend(neg_person_word_id_list)
        neg_per_person_author_ids.extend(neg_per_person_author_id_list)
        neg_per_person_word_ids.extend(neg_per_person_word_id_list)
        neg_author_len.extend(neg_author_len_list)
        neg_word_len.extend(neg_word_len_list)
        neg_person_papers.extend(neg_person_paper_num)

        new_pos_person_author_ids.extend(new_pos_person_author_id_list)
        new_pos_person_word_ids.extend(new_pos_person_word_id_list)
        new_pos_per_person_author_ids.extend(new_pos_per_person_author_id_list)
        new_pos_per_person_word_ids.extend(new_pos_per_person_word_id_list)
        new_pos_author_len.extend(new_pos_author_len_list)
        new_pos_word_len.extend(new_pos_word_len_list)
        new_pos_person_papers.extend(new_pos_person_paper_num)

        new_paper_author_ids.extend(new_paper_author_id_list)
        new_paper_author_idfs.extend(new_paper_author_idf_list)
        new_paper_word_ids.extend(new_paper_word_id_list)
        new_paper_word_idfs.extend(new_paper_word_idf_list)
        new_paper_authors.extend(new_paper_author_num)
        new_paper_words.extend(new_paper_word_num)

        pos_str_features.extend(pos_str_feature_list)
        neg_str_features.extend(neg_str_feature_list)

        if(len(neg_person_author_ids) % settings.NEG_SAMPLE != 0):
            print("NEG NUM DO NOT MATCH!!!")
            exit()

        if(len(neg_person_author_ids) == settings.BATCH_SIZE):

            new_paper_author_ids = _add_mask(author_num, new_paper_author_ids, min(max(new_paper_authors),settings.MAX_AUTHOR))
            new_paper_author_idfs = _add_mask(0, new_paper_author_idfs, min(max(new_paper_authors),settings.MAX_AUTHOR))
            new_paper_word_ids = _add_mask(word_num, new_paper_word_ids, min(max(new_paper_words),settings.MAX_WORD))
            new_paper_word_idfs = _add_mask(0, new_paper_word_idfs, min(max(new_paper_words),settings.MAX_WORD))

            neg_person_author_ids = _add_mask(author_num, neg_person_author_ids, min(max(neg_author_len),settings.MAX_AUTHOR))
            neg_person_word_ids = _add_mask(word_num, neg_person_word_ids, min(max(neg_word_len),settings.MAX_WORD))

            new_pos_person_author_ids = _add_mask(author_num, new_pos_person_author_ids, min(max(new_pos_author_len), settings.MAX_AUTHOR)) 
            new_pos_person_word_ids = _add_mask(word_num, new_pos_person_word_ids, min(max(new_pos_word_len), settings.MAX_WORD)) 

            # print(np.array(new_paper_author_idfs).shape)
            # exit()

            # print(len(per_person_word_ids))
            # print(len(word_len))
            # exit(0)
            neg_tmp_author_id, neg_tmp_word_id = [], []
            pos_tmp_author_id, pos_tmp_word_id = [], []

            neg_pad_paper_num = np.max(np.array(neg_person_papers))
            # print(np.array(neg_person_papers))

            neg_pad_author = np.max(np.array(neg_author_len))
            # print(np.array(neg_author_len))

            neg_pad_word = np.max(np.array(neg_word_len))

            pos_pad_paper_num = np.max(np.array(new_pos_person_papers))
            # print(np.array(neg_person_papers))

            pos_pad_author = np.max(np.array(new_pos_author_len))
            # print(np.array(neg_author_len))

            pos_pad_word = np.max(np.array(new_pos_word_len))
            # print(np.array(neg_word_len))
            # exit()
            for i in range(len(neg_per_person_author_ids)):
                neg_per_author_id_list = neg_per_person_author_ids[i]
                neg_per_word_id_list = neg_per_person_word_ids[i]

                pos_per_author_id_list = new_pos_per_person_author_ids[i]
                pos_per_word_id_list = new_pos_per_person_word_ids[i]
   
                # print("neg pad_paper: {} pad_author: {} pad_word: {}".format(neg_pad_paper_num, neg_pad_author, neg_pad_word))
                # print("pos pad_paper: {} pad_author: {} pad_word: {}".format(pos_pad_paper_num, pos_pad_author, pos_pad_word))

                neg_per_author_id_list = _add_paper_mask(author_num, neg_per_author_id_list, neg_pad_author, min(neg_pad_paper_num, settings.MAX_PAPER))
                neg_per_word_id_list = _add_paper_mask(word_num, neg_per_word_id_list, neg_pad_word, min(neg_pad_paper_num, settings.MAX_PAPER))

                pos_per_author_id_list = _add_paper_mask(author_num, pos_per_author_id_list, pos_pad_author, min(pos_pad_paper_num, settings.MAX_PAPER))
                pos_per_word_id_list = _add_paper_mask(word_num, pos_per_word_id_list, pos_pad_word, min(pos_pad_paper_num, settings.MAX_PAPER))

                
                neg_tmp_author_id.append(neg_per_author_id_list)
                neg_tmp_word_id.append(neg_per_word_id_list)
                
                pos_tmp_author_id.append(pos_per_author_id_list)
                pos_tmp_word_id.append(pos_per_word_id_list)


            b_neg_person_author_ids.append(neg_person_author_ids) 
            b_neg_person_word_ids.append(neg_person_word_ids)
            b_neg_per_person_author_ids.append(neg_tmp_author_id)
            b_neg_per_person_word_ids.append(neg_tmp_word_id)
            # print(np.array(neg_person_papers))
            
            neg_person_papers = np.array(neg_person_papers)
            neg_person_papers[neg_person_papers > settings.MAX_PAPER] = settings.MAX_PAPER
            b_neg_person_papers.append(neg_person_papers.tolist())
            # print(neg_person_papers)
            # print(np.array(neg_tmp_author_id).shape)

            b_new_pos_person_author_ids.append(new_pos_person_author_ids) 
            b_new_pos_person_word_ids.append(new_pos_person_word_ids)
            b_new_pos_per_person_author_ids.append(pos_tmp_author_id)
            b_new_pos_per_person_word_ids.append(pos_tmp_word_id)
            
            new_pos_person_papers = np.array(new_pos_person_papers)
            new_pos_person_papers[new_pos_person_papers > settings.MAX_PAPER] = settings.MAX_PAPER
            b_new_pos_person_papers.append(new_pos_person_papers.tolist())

            b_new_paper_author_ids.append(new_paper_author_ids)
            b_new_paper_author_idfs.append(new_paper_author_idfs)
            b_new_paper_word_ids.append(new_paper_word_ids)
            b_new_paper_word_idfs.append(new_paper_word_idfs)



            # print(np.array(b_new_pos_per_person_author_ids).shape)



            neg_person_author_ids, neg_person_word_ids, neg_per_person_author_ids, neg_per_person_word_ids, neg_author_len, neg_word_len, neg_person_papers = [], [], [], [], [], [], [] 
            new_pos_person_author_ids, new_pos_person_word_ids, new_pos_per_person_author_ids, new_pos_per_person_word_ids, new_pos_author_len, new_pos_word_len, new_pos_person_papers = [], [], [], [], [], [], []
            new_paper_author_ids, new_paper_author_idfs, new_paper_word_ids, new_paper_word_idfs, new_paper_authors, new_paper_words = [], [], [], [], [], []


    if neg_person_author_ids:

        new_paper_author_ids = _add_mask(author_num, new_paper_author_ids, min(max(new_paper_authors),settings.MAX_AUTHOR))
        new_paper_author_idfs = _add_mask(0, new_paper_author_idfs, min(max(new_paper_authors),settings.MAX_AUTHOR))
        new_paper_word_ids = _add_mask(word_num, new_paper_word_ids, min(max(new_paper_words),settings.MAX_WORD))
        new_paper_word_idfs = _add_mask(0, new_paper_word_idfs, min(max(new_paper_words),settings.MAX_WORD))

        neg_person_author_ids = _add_mask(author_num, neg_person_author_ids, min(max(neg_author_len),settings.MAX_AUTHOR))
        neg_person_word_ids = _add_mask(word_num, neg_person_word_ids, min(max(neg_word_len),settings.MAX_WORD))

        new_pos_person_author_ids = _add_mask(author_num, new_pos_person_author_ids, min(max(new_pos_author_len), settings.MAX_AUTHOR)) 
        new_pos_person_word_ids = _add_mask(word_num, new_pos_person_word_ids, min(max(new_pos_word_len), settings.MAX_WORD)) 

        # print(np.array(new_paper_author_idfs).shape)
        # exit()

        # print(len(per_person_word_ids))
        # print(len(word_len))
        # exit(0)
        neg_tmp_author_id, neg_tmp_word_id = [], []
        pos_tmp_author_id, pos_tmp_word_id = [], []

        neg_pad_paper_num = np.max(np.array(neg_person_papers))
        # print(np.array(neg_person_papers))

        neg_pad_author = np.max(np.array(neg_author_len))
        # print(np.array(neg_author_len))

        neg_pad_word = np.max(np.array(neg_word_len))

        pos_pad_paper_num = np.max(np.array(new_pos_person_papers))
        # print(np.array(neg_person_papers))

        pos_pad_author = np.max(np.array(new_pos_author_len))
        # print(np.array(neg_author_len))

        pos_pad_word = np.max(np.array(new_pos_word_len))
        # print(np.array(neg_word_len))
        # exit()
        for i in range(len(neg_per_person_author_ids)):
            neg_per_author_id_list = neg_per_person_author_ids[i]
            neg_per_word_id_list = neg_per_person_word_ids[i]

            pos_per_author_id_list = new_pos_per_person_author_ids[i]
            pos_per_word_id_list = new_pos_per_person_word_ids[i]

            # print("neg pad_paper: {} pad_author: {} pad_word: {}".format(neg_pad_paper_num, neg_pad_author, neg_pad_word))
            # print("pos pad_paper: {} pad_author: {} pad_word: {}".format(pos_pad_paper_num, pos_pad_author, pos_pad_word))

            neg_per_author_id_list = _add_paper_mask(author_num, neg_per_author_id_list, neg_pad_author, min(neg_pad_paper_num, settings.MAX_PAPER))
            neg_per_word_id_list = _add_paper_mask(word_num, neg_per_word_id_list, neg_pad_word, min(neg_pad_paper_num, settings.MAX_PAPER))

            pos_per_author_id_list = _add_paper_mask(author_num, pos_per_author_id_list, pos_pad_author, min(pos_pad_paper_num, settings.MAX_PAPER))
            pos_per_word_id_list = _add_paper_mask(word_num, pos_per_word_id_list, pos_pad_word, min(pos_pad_paper_num, settings.MAX_PAPER))

            
            neg_tmp_author_id.append(neg_per_author_id_list)
            neg_tmp_word_id.append(neg_per_word_id_list)
            
            pos_tmp_author_id.append(pos_per_author_id_list)
            pos_tmp_word_id.append(pos_per_word_id_list)


        b_neg_person_author_ids.append(neg_person_author_ids) 
        b_neg_person_word_ids.append(neg_person_word_ids)
        b_neg_per_person_author_ids.append(neg_tmp_author_id)
        b_neg_per_person_word_ids.append(neg_tmp_word_id)
        # print(np.array(neg_person_papers))
        
        neg_person_papers = np.array(neg_person_papers)
        neg_person_papers[neg_person_papers > settings.MAX_PAPER] = settings.MAX_PAPER
        b_neg_person_papers.append(neg_person_papers.tolist())
        # print(neg_person_papers)
        # print(np.array(neg_tmp_author_id).shape)

        b_new_pos_person_author_ids.append(new_pos_person_author_ids) 
        b_new_pos_person_word_ids.append(new_pos_person_word_ids)
        b_new_pos_per_person_author_ids.append(pos_tmp_author_id)
        b_new_pos_per_person_word_ids.append(pos_tmp_word_id)
        
        new_pos_person_papers = np.array(new_pos_person_papers)
        new_pos_person_papers[new_pos_person_papers > settings.MAX_PAPER] = settings.MAX_PAPER
        b_new_pos_person_papers.append(new_pos_person_papers.tolist())

        b_new_paper_author_ids.append(new_paper_author_ids)
        b_new_paper_author_idfs.append(new_paper_author_idfs)
        b_new_paper_word_ids.append(new_paper_word_ids)
        b_new_paper_word_idfs.append(new_paper_word_idfs)
   

    pos_str_features = np.array(pos_str_features)
    neg_str_features = np.array(neg_str_features)

    pos_str_backup = copy.deepcopy(pos_str_features)
    neg_str_backup = copy.deepcopy(neg_str_features)

    pos_str_features[pos_str_features == -1] = 0
    neg_str_features[neg_str_features == -1] = 0

    total_str = np.concatenate((pos_str_features, neg_str_features), axis=0)
    
    total_mean = np.mean(total_str, 0)

    for i in [6,7,13,14]:
        pos_str_backup[:,i][pos_str_backup[:,i] == -1] = total_mean[i]

        neg_str_backup[:,i][neg_str_backup[:,i] == -1] = total_mean[i]

    assert len(pos_str_backup) == len(neg_str_backup)
    b_pos_str_feature, b_neg_str_feature = [], []

    data_len = len(pos_str_backup)

    for i in range(0, data_len, settings.BATCH_SIZE):
        b_pos_str_feature.append(pos_str_backup[i:min(i + settings.BATCH_SIZE, data_len)]) 
        b_neg_str_feature.append(neg_str_backup[i:min(i + settings.BATCH_SIZE, data_len)])

    assert len(b_pos_str_feature) == len(b_neg_person_author_ids)
    return b_neg_person_author_ids, b_neg_person_word_ids, b_neg_per_person_author_ids, b_neg_per_person_word_ids, b_neg_person_papers,\
    b_new_pos_person_author_ids, b_new_pos_person_word_ids, b_new_pos_per_person_author_ids, b_new_pos_per_person_word_ids,b_new_pos_person_papers,\
    b_new_paper_author_ids, b_new_paper_author_idfs, b_new_paper_word_ids, b_new_paper_word_idfs, b_pos_str_feature, b_neg_str_feature


    # return b_person_author_ids, b_person_author_idfs, b_person_author_nums, b_person_word_ids, b_person_word_idfs, b_person_word_nums,\
    #     b_paper_author_ids, b_paper_author_idfs, b_paper_author_nums, b_paper_word_ids, b_paper_word_idfs, b_paper_word_nums,\
    #     b_person_pids_list, b_pid_list, b_labels,\
    #     b_per_person_author_ids, b_per_person_author_idfs, b_per_person_word_ids, b_per_person_word_idfs, b_per_person_paper_num

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

    return person_author_ids, person_word_ids, per_person_author_ids, per_person_word_ids, author_len, word_len

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

    return person_author_ids, person_word_ids, per_person_author_ids, per_person_word_ids, author_len, word_len

def _gen_pos_and_neg_pair(index):

    # print("iL: ",i)

    i = index[0]
    neg_idx = index[1]
    pos_pid = pid_list[i]
    (name, aid) = pid_dict[pos_pid]
    pos_person_pids = name2aidpid[name][aid]

    # sample_list = name2plist[name]
    # # print("aid:{} name:{} sample_list:{}".format(aid, name, sample_list))
    # copy_sample_list = copy.deepcopy(sample_list)
    # copy_sample_list.remove(aid)
    # random.shuffle(copy_sample_list)

    # neg_num = 0
    # neg_idx = []
    # for pid in copy_sample_list:
        
    #     if(len(name2aidpid[name][pid]) > 1):
    #         neg_idx.append(pid)

    #         neg_num += 1
    #     if (neg_num == neg_sample):
    #         break

    if (len(neg_idx) != neg_sample):
        print("error")
        exit()

    pos_str_feature, neg_str_feature = [], []


    # neg person
    neg_person_author_id_list, neg_person_word_id_list = [],[]
    neg_per_person_author_id_list, neg_per_person_word_id_list = [],[]
    neg_author_len_list, neg_word_len_list = [], []
    neg_person_paper_num = []

    for neg_pid in neg_idx:
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
    # pos person
    pos_pairs=(pos_pid, aid)

    pos_str_feature.append(pairwise_feature(pos_pairs, name, name2aidpid, pubs_dict))


    pos_person_author_id_list, pos_person_word_id_list = [],[]
    pos_per_person_author_id_list, pos_per_person_word_id_list = [],[]
    pos_author_len_list, pos_word_len_list = [], []
    pos_person_paper_num = []


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
    new_pos_person_author_id_list = np.repeat(np.array(pos_person_author_id_list), neg_sample, axis = 0).tolist()
    new_pos_person_word_id_list = np.repeat(np.array(pos_person_word_id_list), neg_sample, axis = 0).tolist()
    new_pos_per_person_author_id_list = np.repeat(np.array(pos_per_person_author_id_list), neg_sample, axis = 0).tolist()   
    new_pos_per_person_word_id_list = np.repeat(np.array(pos_per_person_word_id_list), neg_sample, axis = 0).tolist() 
    new_pos_author_len_list = np.repeat(np.array(pos_author_len_list), neg_sample, axis = 0).tolist() 
    new_pos_word_len_list = np.repeat(np.array(pos_word_len_list), neg_sample, axis = 0).tolist()
    new_pos_person_paper_num = np.repeat(np.array(pos_person_paper_num), neg_sample, axis = 0).tolist()


    pos_str_feature = np.repeat(np.array(pos_str_feature), neg_sample, axis = 0).tolist()
    # pos person
    # print(aid)
    # print(len(new_pos_person_word_id_list))
    # print(len(new_pos_per_person_word_id_list))
    # print(new_pos_author_len_list)
    # print(new_pos_word_len_list)
    # print(new_pos_person_paper_num)



    # author_id_list, author_idf_list, word_id_list, word_idf_list, _, _, _,_ ,_ ,_ = pub_feature_dict[pos_pid]
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

    new_paper_author_id_list = np.repeat(np.array(paper_author_id_list), neg_sample, axis = 0).tolist()
    new_paper_author_idf_list = np.repeat(np.array(paper_author_idf_list), neg_sample, axis = 0).tolist()
    new_paper_word_id_list = np.repeat(np.array(paper_word_id_list), neg_sample, axis = 0).tolist()
    new_paper_word_idf_list = np.repeat(np.array(paper_word_idf_list), neg_sample, axis = 0).tolist()   

    new_paper_author_num = np.repeat(np.array(paper_author_num), neg_sample, axis = 0).tolist()
    new_paper_word_num = np.repeat(np.array(paper_word_num), neg_sample, axis = 0).tolist()

    return neg_person_author_id_list, neg_person_word_id_list, neg_per_person_author_id_list, neg_per_person_word_id_list, neg_author_len_list, neg_word_len_list, neg_person_paper_num,\
    new_pos_person_author_id_list, new_pos_person_word_id_list, new_pos_per_person_author_id_list, new_pos_per_person_word_id_list, new_pos_author_len_list, new_pos_word_len_list, new_pos_person_paper_num,\
    new_paper_author_id_list, new_paper_author_idf_list, new_paper_word_id_list, new_paper_word_idf_list, new_paper_author_num, new_paper_word_num, pos_str_feature, neg_str_feature

def _test_gen_pos_and_neg_pair(index):

    # print("iL: ",i)
    i = index[0]
    neg_idx = index[1]
    pos_pid = pid_list[i]
    (name, aid) = pid_dict[pos_pid]
    pos_person_pids = name2aidpid[name][aid]

    # sample_list = name2plist[name]
    # # print("aid:{} name:{} sample_list:{}".format(aid, name, sample_list))
    # copy_sample_list = copy.deepcopy(sample_list)
    # copy_sample_list.remove(aid)
    # random.shuffle(copy_sample_list)

    # neg_num = 0
    # neg_idx = []
    # for pid in copy_sample_list:
        
    #     if(len(name2aidpid[name][pid]) > 1):
    #         neg_idx.append(pid)

    #         neg_num += 1
    #     if (neg_num == settings.TEST_SAMPLE):
    #         break

    if (len(neg_idx) != settings.TEST_SAMPLE):
        print("error: ",len(neg_idx))
        exit()
    pos_str_feature, neg_str_feature = [], []
    # neg person
    neg_person_author_id_list, neg_person_word_id_list = [],[]
    neg_per_person_author_id_list, neg_per_person_word_id_list = [],[]
    neg_author_len_list, neg_word_len_list = [], []
    neg_person_paper_num = []

    for neg_pid in neg_idx:
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
    # pos person
    pos_person_author_id_list, pos_person_word_id_list = [],[]
    pos_per_person_author_id_list, pos_per_person_word_id_list = [],[]
    pos_author_len_list, pos_word_len_list = [], []
    pos_person_paper_num = []

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

    # print(pos_person_paper_num)
    # print(len(pos_per_person_word_id_list))
    # pos person
    # print(aid)
    # print(len(new_pos_person_word_id_list))
    # print(len(new_pos_per_person_word_id_list))
    # print(new_pos_author_len_list)
    # print(new_pos_word_len_list)
    # print(new_pos_person_paper_num)


    
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

    new_paper_author_id_list = np.repeat(np.array(paper_author_id_list), settings.TEST_SAMPLE + 1, axis = 0).tolist()
    new_paper_author_idf_list = np.repeat(np.array(paper_author_idf_list), settings.TEST_SAMPLE + 1, axis = 0).tolist()
    new_paper_word_id_list = np.repeat(np.array(paper_word_id_list), settings.TEST_SAMPLE + 1, axis = 0).tolist()
    new_paper_word_idf_list = np.repeat(np.array(paper_word_idf_list), settings.TEST_SAMPLE + 1, axis = 0).tolist()   

    new_paper_author_num = np.repeat(np.array(paper_author_num), settings.TEST_SAMPLE + 1, axis = 0).tolist()
    new_paper_word_num = np.repeat(np.array(paper_word_num), settings.TEST_SAMPLE + 1, axis = 0).tolist()
    # print(new_paper_word_num)

    return pos_person_author_id_list, pos_person_word_id_list, pos_per_person_author_id_list, pos_per_person_word_id_list, pos_author_len_list, pos_word_len_list, pos_person_paper_num,\
    new_paper_author_id_list, new_paper_author_idf_list, new_paper_word_id_list, new_paper_word_idf_list, new_paper_author_num, new_paper_word_num, pos_str_feature 
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

# tmp: for test
def get_feature_by_id(feature_ids, feature_padding, emb_model):
    features = []
    for feature_id in feature_ids:
        if feature_id == feature_padding:
            break
        feature = emb_model.wv.index2word[feature_id]
        features.append(feature)
    return features
# tmp: for test
def check_feature_emb(feature_ids, feature_padding, emb_model, emb_array):
    features = []
    for feature_id in feature_ids:
        if feature_id == feature_padding:
            break
        emb = emb_array[feature_id]
        feature = emb_model.wv.index2word[feature_id]
        feature_emb = emb_model.wv[feature]
        print("emb array:",emb)
        print("emb w2v:",feature_emb)
    

# tmp: for test
def get_pub_by_id(pub_id, _pubs_dict):
    return _pubs_dict[pub_id[:pub_id.index("-")]]

if __name__=='__main__':
    train = generate_data_batches("name_to_pubs_train_500.json", settings.TRAIN_SCALE)
    print("generated train batches: ",settings.TRAIN_SCALE)
    test = generate_data_batches("name_to_pubs_test_100.json", settings.TEST_SCALE)
    print("generated test batches: ",settings.TEST_SCALE)
    


    ################### Check the triplets #################################

    person_author_id_list = train[0]
    person_word_id_list = train[3]
    pos_author_id_list = train[6]
    pos_word_id_list = train[9]
    neg_author_id_list = train[12]
    neg_word_id_list = train[15]
    person_pids = train[18]
    pos_pid = train[19]
    neg_pid = train[20]

    author_model = Word2Vec.load(join(settings.EMB_DATA_DIR, 'author_name_backup.emb'))
    word_model = Word2Vec.load(join(settings.EMB_DATA_DIR, 'word_backup.emb'))

    author_emb = data_utils.load_data(settings.EMB_DATA_DIR, "author_emb_backup.array")
    word_emb = data_utils.load_data(settings.EMB_DATA_DIR, "word_emb_backup.array")

    pubs_dict = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'pubs_raw.json')

    # author_names = get_feature_by_id(person_author_id_list[0][0], author_num, author_model)
    # print(author_names)

    # words = get_feature_by_id(person_word_id_list[0][0], word_num, word_model)
    # print(words)

    # for pid in person_pids[0][0]:
    #     pub = get_pub_by_id(pid, pubs_dict)
    #     print(pub)    

    # pos_author_names = get_feature_by_id(neg_author_id_list[0][0], author_num, author_model)
    # print(pos_author_names)

    # pos_words = get_feature_by_id(neg_word_id_list[0][0], word_num, word_model)
    # print(pos_words)

    # pub = get_pub_by_id(neg_pid[0][0], pubs_dict)
    # print(pub)    


    # check_feature_emb(person_author_id_list[0][0], author_num, author_model,author_emb)
    # check_feature_emb(pos_author_id_list[0][0], author_num, author_model,author_emb)
    # check_feature_emb(neg_author_id_list[0][0], author_num, author_model,author_emb)

    # check_feature_emb(person_word_id_list[0][0], word_num, word_model,word_emb)
    # check_feature_emb(pos_word_id_list[0][0], word_num, word_model,word_emb)
    # check_feature_emb(neg_word_id_list[0][0], word_num, word_model,word_emb)


    # check triplets
    for pid in person_pids[0][0]:
        pub = get_pub_by_id(pid, pubs_dict)
        print(pub)

    print("===== pos =====")
    pub = get_pub_by_id(pos_pid[0][0], pubs_dict)
    print(pub)    

    print("===== neg ======")
    pub = get_pub_by_id(neg_pid[0][0], pubs_dict)
    print(pub)    










    
