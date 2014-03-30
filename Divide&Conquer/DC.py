# -*- coding: utf-8 -*-
import numpy as np
from sklearn.mixture import DPGMM
import load_data as ld
import scipy.spatial.distance as sc
import matplotlib
# running on the server gives exception about DISPLAY variable,
# use this in order to avoid them and successfully save figures
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rankSVM import RankSVM
from sklearn.decomposition import DictionaryLearning
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from natsort import natsorted
import multiprocessing as mpc
import cPickle as pickle
import shelve


# estimate the query features
# according to the top documents
# from a reference feature
def query_features(queries, reference, nr_rel_docs, total_features, dataq):
    data = np.zeros((queries.shape[0], 2*total_features))
    cnt = 0
    for q in queries:
        ranking_feat = np.asfarray(dataq[q])
        sortedarray = np.argsort(ranking_feat[:, reference])[::-1]
        useful = ranking_feat[sortedarray[0:nr_rel_docs], :]
        data[cnt] = np.hstack((np.mean(useful, axis=0), np.std(useful, axis=0)**2))
        cnt += 1
    return data


def get_dic_per_cluster(clust_q, data_cluster, dataq, i, out_q=None, kerPCA=False):
    if out_q is not None:
        name = mpc.current_process().name
        print name, 'Starting'
    else:
        print 'Starting estimation of dic %i...' % i
    # parse the feature vectors for each cluster
    for q in clust_q:
        data_cluster = np.vstack((data_cluster, dataq[q]))
    # remove useless first line
    data_cluster = data_cluster[1:, :]
    # learn the sparse code for that cluster
    if kerPCA is False:
        dict_learn = DictionaryLearning(n_jobs=10)
        dict_learn.fit(data_cluster)
    else:
        print 'Doing kernel PCA...'
        print data_cluster.shape
        dict_learn = KernelPCA(kernel="rbf", gamma=10, n_components=3)
        #dict_learn = PCA(n_components=10)
        dict_learn.fit(data_cluster)
    if out_q is not None:
        res = {}
        res[i] = dict_learn
        out_q.put(res)
        print name, 'Exiting'
    else:
        print 'Finished.'
        return dict_learn   # dict(i = dict_learn)


# parse the set of atoms for each cluster
# in order to represent the data under that cluster
# with a cluster specific sparse code
def create_dics(each_cluster, dataq, train, total_features=64, parallel=False, kerPCA=False):
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
            args = (clust_q, data_cluster, dataq, i, out_q, kerPCA)

            p = mpc.Process(name='dic_cluster_%i' % i, target=get_dic_per_cluster, args=args)
            processes.append(p)
            p.start()
        else:
            dics.append(get_dic_per_cluster(clust_q, data_cluster, dataq, i, kerPCA=kerPCA))

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


def train_cl(data_cluster, labels_cluster, i, dic=None, out_q=None):
    if out_q is not None:
        name = mpc.current_process().name
        print name, 'Starting'
    else:
        print 'Training for cluster %i...' % i
    # train the classifier and add it to the list of the ensemble
    if dic is not None:
        # encode the features according to the dictionary of that cluster
        # remove NaN or infinity values that might exist
        tr = dic.transform(data_cluster)
        tr[np.isnan(tr)] = 0
        tr[np.isinf(tr)] = 0
        # sometimes the queries in tat cluster can have only one
        # class e.g. only non-relevant
        # Hence training will throw exceptions, simply add an untrained
        # rankSVM and check when gathering the predictions
        # accordingly
        try:
            ranksvm = RankSVM().fit(tr, labels_cluster)
        except:
            ranksvm = RankSVM()
    else:
        try:
            ranksvm = RankSVM().fit(data_cluster, labels_cluster)
        except:
            ranksvm = RankSVM()
    if out_q is not None:
        res = {}
        res[i] = ranksvm
        out_q.put(res)
        print name, 'Exiting'
    else:
        return ranksvm   # dict(i = ranksvm)


# train an ensemble of classifiers
# one according to each cluster
def train_ensemble(each_cluster, dataq, labelq, train, dics=None, total_features=64, parallel=False):
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
            labels_cluster = np.hstack((labels_cluster, np.squeeze(labelq[q])))

        # remove the useless first entries
        data_cluster = data_cluster[1:, :]
        labels_cluster = labels_cluster[1:]
        if dics is not None:
            dic_i = dics[i]
        else:
            dic_i = None
        if parallel:
            args = (data_cluster, labels_cluster, i, dic_i, out_q)

            p = mpc.Process(name='train_cluster_%i' % i, target=train_cl, args=args)
            processes.append(p)
            p.start()
        else:
            #result.update(train_cl(data_cluster, labels_cluster, dic_i, i))
            ensemble.append(train_cl(data_cluster, labels_cluster, dic_i, i))

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
        check = np.vstack((data_cluster_train, data_cluster_test[i, :]))
        diss = sc.pdist(check, 'mahalanobis')
        diss = sc.squareform(diss)
        test_dis[i, :] = diss[:-1, -1]
    return test_dis


# parse the probability distribution over the clusters
# for the testing queries
def estimate_cluster_prob(test_dis, dpgmm, clusters):
    estimates = np.zeros((test_dis.shape[0], clusters.shape[0]))
    for i in xrange(estimates.shape[0]):
        tot_probs = dpgmm.predict_proba(test_dis[i, :].reshape((1, test_dis[i, :].shape[0])))[0, clusters[:]]
        tot_probs /= tot_probs.sum()
        estimates[i, :] = tot_probs
    return estimates


# find the predicted rankings from each classifier
def predict_ensemble(ensemble, estimates, test, dataq, dics=None, kerPCA=False):
    # dict containing a list of predictions from each classifier
    # indexed by the query string
    predictions_per_q = {}
    cnt = 0
    for q in test:
        print cnt
        data_test = np.asarray(dataq[q])
        predictions = []
        # encode the testing query with a weighted encoding
        # by the probability distribution over the clusters
        for i in xrange(len(ensemble)):
            if dics is not None:
                data_encode = estimates[cnt, i] * dics[i].transform(data_test)
                # since sometimes a model was not trained we might
                # get exceptions when gathering the predictions
                try:
                    pred = ensemble[i].predict(data_encode)
                    predictions.append(pred)
                except:
                    pass
            else:
                data_encode = estimates[cnt, i] * data_test
                try:
                    pred = ensemble[i].predict(data_encode)
                    predictions.append(pred)
                except:
                    pass
        predictions_per_q[q] = predictions
        cnt += 1
    return predictions_per_q


# aggregate the results and produce the final ranking
def find_final_ranking(predictions_per_q):
    ranked_list_per_q = {}
    for q in predictions_per_q:
        rankings = []
        # get all the predictions
        predictions = np.asarray(predictions_per_q[q])
        if np.sum(np.squeeze(predictions)) == 0:
            continue
        print 'pred:' + str(predictions)
        # the unique ones are the documents
        docs = np.unique(predictions)
        print 'doc:' + str(docs)
        for doc in docs:
            # find the index of the document in the prediction
            pred_ranks = np.where(predictions == doc)[1]
            # just average the indices from all the predictions
            new_rank = int(np.sum(pred_ranks)/pred_ranks.shape[0])
            # this is the new rank of the document
            rankings.append((doc, new_rank))
        # sort the list in place
        rankings.sort(key=lambda tup: tup[1])
        ranked_list_per_q[q] = np.asarray(rankings)[:, 0]
    return ranked_list_per_q


def dcg(preds, k):
    if preds.shape[0] > k:
        return np.sum([preds[i]/np.log2(i+2) for i in range(0, k)])
    else:
        return 0.0


def get_ndcg(ranked_list_per_q, labelsq, rel_k):
    ndcg = np.zeros(len(rel_k))
    for q in ranked_list_per_q:
        rankings = np.asarray(ranked_list_per_q[q], dtype=int)
        labels = np.squeeze(np.asarray(labelsq[q]))
        print 'labels:' + str(labels)
        print 'rankings:' + str(rankings)
        ideal_labels = np.sort(labels)[::-1]
        ranked_labels = labels[rankings[:]]
        for i in xrange(len(rel_k)):
            ideal_dcg = dcg(ideal_labels, rel_k[i])
            dcg_q = dcg(ranked_labels, rel_k[i])
            # in case we do not have k returned documents
            if ideal_dcg != 0:
                ndcg[i] += dcg_q/ideal_dcg
    ndcg /= len(ranked_list_per_q)
    return ndcg


# write the data in LambdaMART format
def create_txt_files(training, testing, data, labels, q_to_id, num_feat=64):
    towrite_train = np.zeros((1, num_feat + 3), dtype=object)
    towrite_test = np.zeros((1, num_feat + 3), dtype=object)

    for q in training:
        dataq = np.asfarray(data[q])
        qs = np.empty(dataq.shape[0], dtype=int).reshape((dataq.shape[0], 1))
        qs[:] = q_to_id[q]
        labels_ = np.asarray(labels[q], dtype=int).reshape((dataq.shape[0], 1))
        doc_ids = np.asarray(range(dataq.shape[0]), dtype=int).reshape((dataq.shape[0], 1))
        towrite_train = np.vstack((towrite_train, np.hstack((labels_, qs, doc_ids, dataq))))
    towrite_train = towrite_train[1:, :]

    for q in testing:
        dataq = np.asfarray(data[q])
        qs = np.empty(dataq.shape[0], dtype=object).reshape((dataq.shape[0], 1))
        qs[:] = q_to_id[q]
        labels_ = np.asarray(labels[q], dtype=int).reshape((dataq.shape[0], 1))
        doc_ids = np.asarray(range(dataq.shape[0]), dtype=int).reshape((dataq.shape[0], 1))
        towrite_test = np.vstack((towrite_test, np.hstack((labels_, qs, doc_ids, dataq))))
    towrite_test = towrite_test[1:, :]

    np.savetxt('train.txt', towrite_train, delimiter=',')
    np.savetxt('test.txt', towrite_test, delimiter=',')

if __name__ == '__main__':
    #ld.get_letor_3(64)
    #data_total_raw = ld.load_pickle('','data_cluster.pickle')
    print 'Parsing the data...'
    #data, labels = ld.load_pickle_total('', 'data_per_q_total.pickle', 'relevance_per_q_total.pickle')
    data = shelve.open('/virdir/Scratch/saveFR2_SVM_330/q_docs.db')
    labels = shelve.open('/virdir/Scratch/saveFR2_SVM_330/q_rels.db')
    docIDs = shelve.open('/virdir/Scratch/saveFR2_SVM_330/q_docIds.db')
    tmp = natsorted(data.keys())[0:int(0.3 * len(data.keys()))]
    print len(data.keys()), len(tmp)
    queries = np.asanyarray(tmp, dtype=object)

    training, testing = train_test_split(queries, test_size=0.33, random_state=42)

    print 'Training set size: %i, Testing set size: %i' % (training.shape[0], testing.shape[0])

    # create txt files for LambdaMART
    #q_to_id = dict(zip(queries.tolist(),range(queries.shape[0])))
    #create_txt_files(training, testing, data, labels, q_to_id, num_feat=23)

    # print 'Estimating query features...'
    data_cluster_train = query_features(training, 15, 10, 23, data)
    data_cluster_test = query_features(testing, 15, 10, 23, data)
    data_cluster_train_ds = data_cluster_train

    """if you want clustering on the dissimilarity space uncomment
       below and change accordingly"""
    # print 'Calculating dissimilarity space for training queries...'
    # data_cluster_train_ds = sc.pdist(data_cluster_train, 'euclidean')
    # data_cluster_train_ds = sc.squareform(data_cluster_train_ds)

    # # plt.figure(1)
    # # plt.imshow(data_cluster_train_ds)
    # # plt.colorbar()
    # # plt.title('Initial dissimilarity')

    print 'Training a Dirichlet Process Gaussian Mixture model...'
    dpgmm = DPGMM(alpha=1.0, n_iter=100, n_components=50)
    dpgmm.fit(data_cluster_train_ds)
    prediction = dpgmm.predict(data_cluster_train_ds)
    clusters = np.unique(prediction)

    print 'Found %i clusters!' % clusters.shape[0]
    print clusters

    """create the reordered input data according to the clusters
      it is only needed if you want to visuallize the clustering
      afterwards"""
    #data_cluster = np.zeros((1, data_cluster_train.shape[1]))

    # each cluster is a list of lists that contains the indices
    # of the queries for each cluster
    each_cluster = []
    for i in xrange(clusters.shape[0]):
        cluster = data_cluster_train[prediction == clusters[i], :]
        each_cluster.append(np.where(prediction == clusters[i])[0])
        #data_cluster = np.vstack((data_cluster, cluster))
    # remove the useless first element
    #data_cluster = data_cluster[1:, :]
    # visuallize the clustering, it should be a block matrix
    #data_vis = sc.pdist(data_cluster, 'euclidean')
    #data_vis = sc.squareform(data_vis)

    # plt.figure(1)
    # plt.imshow(data_vis)
    # plt.colorbar()
    # plt.title('Clustering results')
    # fig1 = plt.gcf()
    # fig1.savefig('cl_res_dis.png', dpi=100)

    # uncomment below only if you have a clustering on the dissimilarity space
    # print 'Estimating dissimilarity of testing queries to the training ones...'
    # test_dis = parse_test_dis(data_cluster_train, data_cluster_test)

    print 'Estimating probabilities of belonging to the clusters...'
    estimates = estimate_cluster_prob(data_cluster_test, dpgmm, clusters)

    """all of the below pickling stuff can be used to quickly test
    what you want from the pipeline"""

    # save the clustering results
    #with open('each_cluster.pickle', 'wb') as handle:
    #    pickle.dump(each_cluster, handle)

    # save the estimates of the testing queries
    #with open('estimates.pickle', 'wb') as handle:
    #    pickle.dump(estimates, handle)

    # with open('each_cluster.pickle', 'rb') as handle:
    #     each_cluster = pickle.load(handle)

    # with open('estimates.pickle', 'rb') as handle:
    #     estimates = pickle.load(handle)

    """here we will perform different feature extraction
       for each cluster"""

    # print 'Create sparse vector dictionaries per cluster...'
    # dics = create_dics(each_cluster, data, training, total_features=23, parallel=False, kerPCA=False)

    print 'Create kernelPCA per cluster...'
    dics = create_dics(each_cluster, data, training, total_features=23, parallel=False, kerPCA=True)

    # save the classes to disk, can take time to compute them
    # with open('kerPCA_dics_clusters.pickle', 'wb') as handle:
    #     pickle.dump(dics, handle)

    # with open('kerPCA_dics_clusters.pickle', 'rb') as handle:
    #     dics = pickle.load(handle)

    # with open('sparse_dics_clusters.pickle', 'wb') as handle:
    #     pickle.dump(dics, handle)

    # with open('sparse_dics_clusters.pickle', 'rb') as handle:
    #     dics = pickle.load(handle)


    # train the ensemble according to those dictionaries
    print 'Training an ensemble of classifiers...'
    ensemble = train_ensemble(each_cluster, data, labels, training, dics=dics, total_features=23, parallel=True)
    # ensemble = train_ensemble(each_cluster, data, labels, training, dics=None, total_features=23, parallel=True)

    # with open('ensemble_nodic.pickle', 'wb') as handle:
    #     pickle.dump(ensemble, handle)

    # with open('ensemble_nodic.pickle', 'rb') as handle:
    #     ensemble = pickle.load(handle)

    # with open('ensemble_kerPCA.pickle', 'rb') as handle:
    #     pickle.dump(ensemble, handle)

    # with open('ensemble_kerPCA.pickle', 'rb') as handle:
    #     ensemble = pickle.load(handle)

    # with open('ensemble_sparse.pickle', 'wb') as handle:
    #     pickle.dump(ensemble, handle)

    # with open('ensemble_sparse.pickle', 'rb') as handle:
    #     ensemble = pickle.load(handle)

    # prediction for the testing queries
    print 'Getting predictions for the testing queries from the ensemble...'
    predictions_per_q = predict_ensemble(ensemble, estimates, testing, data, dics=dics, kerPCA=False)
    # predictions_per_q = predict_ensemble(ensemble, estimates, testing, data)

    # with open('predictions_per_q_kerPCA.pickle', 'wb') as handle:
    #     pickle.dump(predictions_per_q, handle)

    # with open('predictions_per_q_nodic.pickle', 'wb') as handle:
    #     pickle.dump(predictions_per_q, handle)

    # with open('predictions_per_q_sparse.pickle', 'wb') as handle:
    #     pickle.dump(predictions_per_q, handle)

    # with open('predictions_per_q_kerPCA.pickle', 'rb') as handle:
    #     predictions_per_q = pickle.load(handle)

    # aggregate the results to get the final rankings
    print 'Aggregating the results...'
    ranked_list_per_q = find_final_ranking(predictions_per_q)

    # get the total ndcg at rel_k places
    print 'Estimating the NDCG scores...'
    rel_k = [1, 3, 5, 10]
    ndcg = get_ndcg(ranked_list_per_q, labels, rel_k)

    print 'Final NDCG: ' + str(ndcg)
