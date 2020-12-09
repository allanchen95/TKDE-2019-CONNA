from multiprocessing import Pool
from datetime import datetime
from itertools import chain
from utils import string_utils
from utils import data_utils


def transform_feature(data):
    if type(data) is str:
        data = data.split()
    assert type(data) is list
    return data


def extract_common_features(item):
    title_features = string_utils.clean_sentence(item["title"], stemming=True)
    keywords_features = []
    keywords = item.get("keywords")
    if keywords:
        for k in keywords:
            keywords_features.extend(string_utils.clean_sentence(k, stemming=True))
    venue_features = []
    venue_name = item.get('venue', '')
    if len(venue_name) > 2:
        venue_features = string_utils.clean_sentence(venue_name.lower(), stemming=True)
    return title_features, keywords_features, venue_features


def extract_author_features(item, order=None):
    title_features, keywords_features, venue_features = extract_common_features(item)
    word_features = title_features + keywords_features + venue_features
    author_features = []

    for i, author in enumerate(item["authors"]):
        if order is not None and i != order:
            continue
        org_name = string_utils.clean_sentence(author.get("org", ""), stemming=True)
        if len(org_name) > 2:
            word_features += org_name

        for j, coauthor in enumerate(item["authors"]):
            if i == j:
                continue
            coauthor_name = coauthor.get("name", "")
            if(coauthor_name == None):
                continue
            if(len(coauthor_name.strip()) > 0):
                if len(coauthor_name.strip()) > 2:
                    author_features.append(string_utils.clean_name(coauthor_name))
                else:
                    author_features.append(coauthor_name.lower())
                
    return author_features, word_features
