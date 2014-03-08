import numpy as np
from natsort import natsorted
import cPickle as pickle

def fileAddress(folder, title):
    return folder + title +'.txt'

def loadFile(folder, kind, qids = None, q = None):
    #global data, y, qid
    data, relevance, qid = [], [], []

    file = open(fileAddress(folder, kind))
    #print fileAddress(folder, kind)

    for line in file:
        # apply condition
        if q is not None and not('qid:{} '.format(q) in line):
            continue
        ## do not parse the same query twice
        #if qids is not None and 'qid:{} '.format(q) in qids:
        #    continue

        # process line
        features = []
        first = True
        for word in line.split():
            if first:
                relevance.append(int(word))
                first = False
            elif word.startswith('qid'):
                qid.append(word.split(':')[1])
            elif word.startswith('#'):
                break
            else:
                features.append(float(word.split(':')[1]))

        data.append(features)
    if q is None:
        return data, relevance, qid
    else:
        return data, relevance

# reference_feature 24 is bm25
def get_letor_3(total_features, nr_rel_docs = 30, reference_feature = 24):

    data_total = np.zeros((1,2*total_features))

    data_label_dic = {}
    data_features = np.zeros((1, total_features))
    data_labels = np.zeros(1)

    for i in range(1,7):
        #folder = '~/VariousDATA/OHSUMED/QueryLevelNorm/Fold{}/'.format(i)
        folder ='~/VariousDATA/TestClustering/Fold{}/'.format(i)
        print 'Parsing Fold{}'.format(i)+'...'

        # load qids to be clustered
        data_in, relevance, qid = loadFile(folder, 'vali')
        #qids.append(set(qid[:]))
        qids_dum = set(qid[:])
        data_per_q, relevance_per_q = {}, {}
        for q in qids_dum:
            data_per_q[q], relevance_per_q[q] = loadFile(folder, 'vali', None, q)

        data = np.zeros((len(data_per_q.keys()), 2*total_features))
        cnt = 0
        for q in natsorted(data_per_q.keys()):
            ranking_feat = np.asfarray(data_per_q[q])
            data_features = np.vstack((data_features, ranking_feat))
            data_labels = np.hstack((data_labels, relevance_per_q[q]))
            print data_features.shape
            print data_labels.shape
            sortedarray = np.argsort(ranking_feat[:, reference_feature])[::-1]
            useful = ranking_feat[sortedarray[0:nr_rel_docs],:]
            data[cnt] = np.hstack((np.mean(useful, axis = 0), np.std(useful, axis = 0) **2))
            cnt += 1

        print data.shape
        data_total = np.vstack((data_total, data))
        data_per_q.clear()
        relevance_per_q.clear()

    # remove the first dummy line
    data_total = data_total[1:,:]
    data_label_dic['data'] = data_features[1:, :]
    data_label_dic['labels'] = data_labels[1:]

    with open('data_cluster.pickle', 'wb') as handle:
        pickle.dump(data_total, handle)
    with open('data_label.pickle', 'wb') as handle:
        pickle.dump(data_label_dic, handle)

def load_pickle(path, filename):
    with open(path+filename, 'rb') as handle:
        data = pickle.load(handle)
    return data

