from os.path import join
import sys
sys.path.append("..")
import codecs
import math
from collections import defaultdict as dd
from embedding import EmbeddingModel
from datetime import datetime
# from utils.cache import LMDBClient
from utils import data_utils
from utils import feature_utils
from utils import settings
from utils import multithread_utils
import numpy as np
from nltk.corpus import stopwords

start_time = datetime.now()


_pubs_dict = None

def get_pub_feature(i):
    if i % 1000 == 0:
        print("The %dth paper"%i)
    pid = list(_pubs_dict)[i]
    paper = _pubs_dict[pid]
    if "title" not in paper or "authors" not in paper:
        return None
    if len(paper["authors"]) > 300:
        return None
    if len(paper["authors"]) > 30:
        print(i, pid, len(paper["authors"]))
    n_authors = len(paper.get('authors', []))
    authors = []
    for j in range(n_authors):
        author_features, word_features = feature_utils.extract_author_features(paper, j)
        aid = '{}-{}'.format(pid, j)
        authors.append((aid, author_features, word_features))
    return authors


def dump_pub_features_to_file():
    """
    generate author features by raw publication data and dump to files
    author features are defined by his/her paper attributes excluding the author's name
    """
    global _pubs_dict

    # Load publication features
    _pubs_dict = data_utils.load_json('data_dir', 'pub_files')
    res = multithread_utils.processed_by_multi_thread(get_pub_feature, range(len(_pubs_dict)))
    data_utils.dump_data(res, "Essential_Embeddings/", "pub.features")
    # _pubs_dict = data_utils.load_json(settings.GLOBAL_DATA_DIR, 'pubs_raw.json')
    # res = multithread_utils.processed_by_multi_thread(get_pub_feature, range(len(_pubs_dict)))
    # data_utils.dump_data(res, settings.GLOBAL_DATA_DIR, "pub.features")


def cal_feature_idf():
    """
    calculate word IDF (Inverse document frequency) using publication data
    """
    features = data_utils.load_data('Essential_Embeddings/', "pub.features")
    feature_dir = join('Essential_Embeddings/', 'global')
    index = 0
    author_counter = dd(int)
    author_cnt = 0
    word_counter = dd(int)
    word_cnt = 0
    none_count = 0
    for pub_index in range(len(features)):
        pub_features = features[pub_index]
        # print(pub_features)
        if (pub_features == None):
            none_count += 1
            continue
        for author_index in range(len(pub_features)):
            aid, author_features, word_features = pub_features[author_index]

            if index % 100000 == 0:
                print(index, aid)
            index += 1
            
            for af in author_features:
                author_cnt += 1
                author_counter[af] += 1

            for wf in word_features:
                word_cnt +=1
                word_counter[wf] +=1        

    author_idf = {}
    for k in author_counter:
        author_idf[k] = math.log(author_cnt / author_counter[k])

    word_idf = {}
    for k in word_counter:
        word_idf[k] = math.log(word_cnt / word_counter[k])

    data_utils.dump_data(dict(author_idf), feature_dir, "author_feature_idf.pkl")
    data_utils.dump_data(dict(word_idf), feature_dir, "word_feature_idf.pkl")
    print("None count: ", none_count)

_emb_model = None

def get_feature_index(i):
    word = _emb_model.wv.index2word[i]
    embedding = _emb_model.wv[word]
    return (i, embedding)



def dump_emb_array(emb_model, output_name):
	global _emb_model
	_emb_model = emb_model
	# transform the feature embeddings from embedding to (id, embedding)
	res = multithread_utils.processed_by_multi_thread(get_feature_index, range(len(_emb_model.wv.vocab)))
	sorted_embeddings = sorted(res, key=lambda x:x[0])
	word_embeddings = list(list(zip(*sorted_embeddings))[1])
	data_utils.dump_data(np.array(word_embeddings), 'Essential_Embeddings/emb/', output_name)

def get_feature_ids_idfs_for_one_pub(features, emb_model, idfs):
    id_list = []
    idf_list = []
    for feature in features:
        if not feature in emb_model.wv:
            continue
        id = emb_model.wv.vocab[feature].index
        idf = 1
        if idfs and feature in idfs:
            idf = idfs[feature]
        id_list.append(id)
        idf_list.append(idf)
    return id_list, idf_list


def dump_feature_id_to_file():
    """
    transform a publication into a set of author and word IDs, dump it to csv
    """
    model = EmbeddingModel.Instance()
    author_emb_model = model.load_author_name_emb()
    author_emb_file = "author_emb.array"
    word_emb_model = model.load_word_name_emb()
    word_emb_file = "word_emb.array"
    dump_emb_array(author_emb_model, author_emb_file)
    dump_emb_array(word_emb_model, word_emb_file)

    features = data_utils.load_data('Essential_Embeddings/', "pub.features")
    author_idfs = data_utils.load_data('Essential_Embeddings/global/', 'author_feature_idf.pkl')
    word_idfs = data_utils.load_data('Essential_Embeddings/global/', 'word_feature_idf.pkl')
    index = 0
    feature_dict = {}
    for pub_index in range(len(features)):
    	pub_features = features[pub_index]
    	if (pub_features == None):
            continue
    	for author_index in range(len(pub_features)):
    		aid, author_features, word_features = pub_features[author_index]
    		if index % 100000 == 0:
    			print(index, author_features, word_features)
    		index += 1
    		author_id_list, author_idf_list = get_feature_ids_idfs_for_one_pub(author_features, author_emb_model, author_idfs)
    		word_id_list, word_idf_list = get_feature_ids_idfs_for_one_pub(word_features, word_emb_model, word_idfs)

    		if author_id_list is not None or word_id_list is not None:
    			feature_dict[aid] = (author_id_list, author_idf_list, word_id_list, word_idf_list)
    data_utils.dump_data(feature_dict, 'Essential_Embeddings/emb/', "pub_feature.ids")


if __name__ == '__main__':
    # Processing raw data as follows to generate essential word embeddings.


    #1. dump_pub_features_to_file()   # extract features of author name and words from publications
    #2. cal_feature_idf()                # calculate idf for each author name or word

    #3. emb_model = EmbeddingModel.Instance()   
    #4. emb_model.train()                # train embeddings for author names and words
    
    #5. dump_feature_id_to_file()        
    