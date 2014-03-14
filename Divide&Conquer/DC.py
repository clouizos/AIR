# -*- coding: utf-8 -*-
import numpy as np
from sklearn.mixture import DPGMM
import load_data as ld
import scipy.spatial.distance as sc
import matplotlib.pyplot as plt
from rankSVM import RankSVM
from sklearn.decomposition import DictionaryLearning
from sklearn.cross_validation import train_test_split
from natsort import natsorted
import multiprocessing as mpc

# estimate the query features
# according to the top documents
# from a reference feature
# reference 24 is the bm25 of the document
def query_features(queries, reference, nr_rel_docs, total_features, dataq):
    data = np.zeros((queries.shape[0], 2*total_features))
    cnt = 0
    for q in queries:
        ranking_feat = np.asfarray(dataq[q])
        sortedarray = np.argsort(ranking_feat[:, reference])[::-1]
        useful = ranking_feat[sortedarray[0:nr_rel_docs],:]
        data[cnt] = np.hstack((np.mean(useful, axis = 0), np.std(useful, axis = 0) **2))
        cnt += 1
    return data

def get_dic_per_cluster(clust_q, data_cluster, dataq, i, out_q = None):
    if out_q is not None:
        name = mpc.current_process().name
        print name, 'Starting'
    else:
        print 'Starting estimation of dic %i...' % i
    # parse the feature vectors for each cluster
    for q in clust_q:
        data_cluster = np.vstack((data_cluster,dataq[q]))
    # remove useless first line
    data_cluster = data_cluster[1:,:]
    # learn the sparse code for that cluster
    dict_learn = DictionaryLearning()
    dict_learn.fit(data_cluster)
    if out_q is not None:
        out_q.put(dict(i = dict_learn))
        print name, 'Exiting'
    else:
        return dict(i = dict_learn)


# parse the set of atoms for each cluster
# in order to represent the data under that cluster
# with a cluster specific sparse code
def create_dics(each_cluster, dataq, train, total_features = 64, parallel = False):
    dics = []
    if parallel:
        processes = []
        out_q = mpc.Queue()

    result = {}
    for i in xrange(len(each_cluster)):
        # get the queries in that cluster
        clust_q = train[each_cluster[i][:]]
        data_cluster = np.zeros((1, total_features))
        if parallel:
            args = (clust_q, data_cluster, dataq, i, out_q)

            p = mpc.Process(name='dic_cluster_%i' % i, target=get_dic_per_cluster, args=args)
            processes.append(p)
            p.start()
        else:
            result.update(get_dic_per_cluster(clust_q, data_cluster, dataq, i))

    if parallel:
        # wait for the processes
        for process in processes:
            process.join()

        # parse the results
        for i in xrange(len(each_cluster)):
            result.update(out_q.get())

    # add the dictionaries with the order we have for the clusters
    for key in natsorted(result.keys()):
        dics.append(result[key])

    return dics

def train(data_cluster, labels_cluster, dic, i, out_q = None):
    if out_q is not None:
        name = mpc.current_process().name
        print name, 'Starting'
    else:
        print 'Training for cluster %i...' % i
    # encode the features according to the dictionary of that cluster
    data_cluster = dic.transform(data_cluster)
    # train the classifier and add it to the list of the ensemble
    ranksvm = RankSVM().fit(data_cluster, labels_cluster)
    if out_q is not None:
        out_q.put(dict(i = ranksvm))
        print name, 'Exiting'
    else:
        return dict(i = ranksvm)


# train an ensemble of classifiers
# one according to each cluster
def train_ensemble(each_cluster, dataq, labelq, train, dics, total_features = 64, parallel = False):
    ensemble = []
    if parallel:
        processes = []
        out_q = mpc.Queue()
    result = {}
    for i in xrange(len(each_cluster)):
        # get the queries for that cluster
        clust_q = train[each_cluster[i][:]]
        data_cluster = np.zeros((1, total_features))
        labels_cluster = np.zeros(1)
        # get the data and labels for that cluster
        for q in clust_q:
            data_cluster = np.vstack((data_cluster, dataq[q]))
            labels_cluster = np.hstack((labels_cluster, labelq[q] ))

        # remove the useless first entries
        data_cluster = data_cluster[1:,:]
        labels_cluster = labels_cluster[1:]
        dic_i = dics[i]
        if parallel:
            args = (data_cluster, labels_cluster, dic_i, i, out_q)

            p = mpc.Process(name='train_cluster_%i' % i, target=train, args=args)
            processes.append(p)
            p.start()
        else:
            result.update(train(data_cluster, labels_cluster, dic_i, i))

    if parallel:
        # wait for the processes
        for process in processes:
            process.join()

        # parse the results
        for i in xrange(len(each_cluster)):
            result.update(out_q.get())

    # add the dictionaries with the order we have for the clusters
    for key in natsorted(result.keys()):
        ensemble.append(result[key])

    return ensemble

# parse the dissimilarity of the testing queries
# according to the training ones
def parse_test_dis(data_cluster_train, data_cluster_test):
    test_dis = np.zeros((data_cluster_test.shape[0], data_cluster_train.shape[0]))
    for i in xrange(data_cluster_test.shape[0]):
        check = np.vstack((data_cluster_train, data_cluster_test[i,:]))
        diss = sc.pdist(check, 'mahalanobis')
        diss = sc.squareform(diss)
        test_dis[i,:] = diss[:-1, -1]
    return test_dis

# parse the probability distribution over the clusters
# for the testing queries
def estimate_cluster_prob(test_dis, dpgmm, clusters):
    estimates = np.zeros((test_dis.shape[0], clusters.shape[0]))
    for i in xrange(estimates.shape[0]):
        tot_probs = dpgmm.predict_proba(test_dis[i,:].reshape((1,test_dis[i,:].shape[0])))[0,clusters[:]]
        tot_probs /= tot_probs.sum()
        estimates[i, :] = tot_probs
    return estimates

# find the predicted rankings from each classifier
def predict_ensemble(ensemble, dics, estimates, test, dataq):
    # dict containing a list of predictions from each classifier
    # indexed by the query string
    predictions_per_q = {}
    for q in test:
        data_test = dataq[q]
        data_encode = np.zeros(data_test.shape)
        predictions = []
        # encode the testing query with a weighted encoding
        # by the probability distribution over the clusters
        for i in xrange(len(dics)):
            data_encode += estimates[i] * dics[i].transform(data_test)
        # get the predictions from each classifier
        for predictor in ensemble:
            predictions.append(predictor.predict(data_encode))
        predictions_per_q[q] = predictions
    return predictions_per_q

# aggregate the results and produce the final ranking
def find_final_ranking(predictions_per_q):
    ranked_list_per_q = {}
    for q in predictions_per_q:
        rankings = []
        # get all the predictions
        predictions = np.asarray(predictions_per_q[q])
        # the unique ones are the documents
        docs = np.unique(predictions)
        for doc in docs:
            # find the index of the document in the prediction
            pred_ranks = np.where(predictions == doc)[1]
            # just average the indices from all the predictions
            new_rank = int(np.sum(pred_ranks)/pred_ranks.shape[0])
            # this is the new rank of the document
            rankings.append((doc, new_rank))
        # sort the list in place
        rankings.sort(key=lambda tup: tup[1])
        ranked_list_per_q[q] = rankings
    return ranked_list_per_q

def dcg(preds, k):
    return np.sum([preds[i]/np.log2(i+2) for i in range(0,k)])

def get_ndcg(ranked_list_per_q, labelsq, rel_k):
    ndcg = np.zeros(len(rel_k))
    for q in ranked_list_per_q:
        rankings = ranked_list_per_q[q]
        labels = labelsq[q]
        ideal_labels = np.sort(labels)[::-1]
        ranked_labels = labels[rankings[:]]
        for i in xrange(len(rel_k)):
            ideal_dcg = dcg(ideal_labels, rel_k[i])
            dcg_q = dcg(ranked_labels, rel_k[i])
            ndcg[i] += dcg_q/ideal_dcg
    return ndcg

if __name__ == '__main__':
    #ld.get_letor_3(64)
    #data_total_raw = ld.load_pickle('','data_cluster.pickle')
    print 'Parsing the data...'
    data, labels = ld.load_pickle_total('', 'data_per_q_total.pickle', 'relevance_per_q_total.pickle')

    queries = np.asanyarray(natsorted(data.keys()), dtype=object)

    training, testing = train_test_split(queries, test_size=0.33, random_state=42)
    print 'Training set size: %i, Testing set size: %i' % (training.shape[0], testing.shape[0])

    print 'Estimating query features...'
    data_cluster_train = query_features(training, 24, 50, 64, data)
    data_cluster_test = query_features(testing, 24, 50, 64, data)

    print 'Calculating dissimilarity space for training queries...'
    data_cluster_train_ds = sc.pdist(data_cluster_train, 'mahalanobis')
    data_cluster_train_ds = sc.squareform(data_cluster_train_ds)

    plt.figure(1)
    plt.imshow(data_cluster_train_ds)
    plt.colorbar()
    plt.title('Initial dissimilarity')

    print 'Training a Dirichlet Process Gaussian Mixture model...'
    dpgmm = DPGMM(covariance_type='diag', alpha=2.0, n_iter=1000, n_components=10)
    dpgmm.fit(data_cluster_train_ds)
    prediction = dpgmm.predict(data_cluster_train_ds)
    clusters = np.unique(prediction)

    print 'Found %i clusters!' % clusters.shape[0]
    print clusters

    # create the reordered input data according to the clusters
    data_cluster = np.zeros((1, data_cluster_train.shape[1]))
    each_cluster = []
    for i in xrange(clusters.shape[0]):
        cluster = data_cluster_train[prediction == clusters[i],:]
        each_cluster.append(np.where(prediction == clusters[i])[0])
        data_cluster = np.vstack((data_cluster, cluster))

    # remove the useless first element
    data_cluster = data_cluster[1:, :]

    # visuallize the clustering, it should be a block matrix
    data_vis = sc.pdist(data_cluster, 'mahalanobis')
    data_vis = sc.squareform(data_vis)
    plt.figure(2)
    plt.imshow(data_vis)
    plt.colorbar()
    plt.title('Clustered dissimilarity')
    plt.show()

    print 'Estimating dissimilarity of testing queries to the training ones...'
    test_dis = parse_test_dis(data_cluster_train, data_cluster_test)
    print 'Estimating probabilities of belonging to the clusters...'
    estimates = estimate_cluster_prob(test_dis, dpgmm, clusters)

    # here we will create a sparse coding representation
    # for each cluster
    print 'Create sparse vector dictionaries per cluster...'
    dics = create_dics(each_cluster, data, training)


    # train the ensemble according to those dictionaries
    print 'Training an ensemble of classifiers...'
    ensemble = train_ensemble(each_cluster, data, labels, training, dics)

    # prediction for the testing queries
    print 'Getting predictions for the testing queries from the ensemble...'
    predictions_per_q = predict_ensemble(ensemble, dics, estimates, testing, data)

    # aggregate the results to get the final rankings
    print 'Aggregating the results...'
    ranked_list_per_q = find_final_ranking(predictions_per_q)

    # get the total ndcg at rel_k places
    print 'Estimating the NDCG scores...'
    rel_k = [1,2,3,5,10]
    ndcg = get_ndcg(ranked_list_per_q, labels, rel_k)

    print ndcg
