from __future__ import division
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from os.path import join
from utils import settings
import os
from collections import defaultdict
from utils import data_utils
from utils import multithread_utils


_person_pubIDs = None
_pubIDs = None
_pubs_dict = None
_test_preds = None
_test_grnds = None

def eval_hit(predictions):

    top_k = [1, 3, 5]
    mrr = 0
    top_k_metric = np.array([0 for k in top_k])
    if (len(predictions)!= settings.TEST_SCALE * (settings.TEST_SAMPLE + 1)):
        print("predictions: {} don't match {}".format(len(predictions), settings.TEST_SCALE * (settings.TEST_SAMPLE + 1)))
        exit()
    per_d = settings.TEST_SCALE * (settings.TEST_SAMPLE + 1)

    predictions = np.array(predictions).reshape((settings.TEST_SCALE, settings.TEST_SAMPLE + 1))

    for i in range(len(predictions)):
        rank = np.argsort(-predictions[i,:])

        true_index = np.where(rank == 0)[0][0]
        mrr += 1/(true_index +1)
        # top_k:[1, 5, 10, 50]
        for k in range(len(top_k)):
            if true_index < top_k[k]:
                top_k_metric[k] += 1

    mrr = round(mrr/settings.TEST_SCALE, 3)
    ratio_top_k = np.array([0 for i in top_k], dtype = np.float32)

    for i in range(len(ratio_top_k)):
        ratio_top_k[i] = round(top_k_metric[i] / settings.TEST_SCALE, 3)

    print("hits@{} = {} mrr: {}".format(top_k, ratio_top_k, mrr))
    return ratio_top_k





def eval(pubs_dict, test_person_pubIDs, test_pubIDs, labels, predictions):
    
    global _person_pubIDs
    global _pubIDs
    global _pubs_dict
    global _test_preds
    global _test_grnds
    
    _person_pubIDs = test_person_pubIDs
    _pubIDs = test_pubIDs
    _pubs_dict = pubs_dict
    _test_preds = predictions
    _test_grnds = labels

    res = multithread_utils.processed_by_multi_thread(case_study, range(len(_test_grnds)))
    res = list(zip(*res))
    test_pred_labels = list(res[0])
    # print(labels.shape, predictions.shape)
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels, predictions)

    out_dir = join(settings.OUT_DIR, 'error_case')
    os.makedirs(out_dir, exist_ok=True)
    data_utils.dump_json_text(list(res[1]), out_dir, 'fn.json',4)
    data_utils.dump_json_text(list(res[2]), out_dir, 'fp.json',4)

    return precision, recall, f1_score

def casestudy(i):

    result = {}
    person_pids = _person_pubIDs[i]
    target_pid = _pubIDs[i]
    person_pubs = {}
    for j in range(len(person_pids)):
        pid = person_pids[j]
        if pid == target_pid:
            break
        person_pubs[pid] = {"content":_pubs_dict[pid[:pid.index("-")]]}
    pub = _pubs_dict[target_pid[:target_pid.index("-")]]
    result['person'] = person_pubs
    result['paper'] = {'pid':target_pid,'content':pub}
    return result


def case_study(i):
    test_grnd = _test_grnds[i]
    test_pred = _test_preds[i]
    fn_result = {}
    fp_result= {}

    # The paper belongs to the person, but is predicted not to be his paper
    if test_pred == 0 and test_grnd == 1:
        fn_result = casestudy(i)
   
    # The paper does not belong to the person, but is predicted to be his paper
    elif test_pred == 1 and test_grnd == 0:
        fp_result = casestudy(i)
    return test_pred, fn_result, fp_result



