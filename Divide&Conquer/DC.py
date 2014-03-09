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

# parse the set of atoms for each cluster
# in order to represent the data under that cluster
# with a cluster specific sparse code
def create_dics(each_cluster, dataq, train, total_features = 64):
    dics = []
    for cluster in each_cluster:
        # get the queries in that cluster
        clust_q = train[cluster[:]]
        data_cluster = np.zeros((1, total_features))
        # parse the feature vectors for each cluster
        for q in clust_q:
            data_cluster = np.vstack((data_cluster,dataq[q]))
        # remove useless first line
        data_cluster = data_cluster[1:,:]
        # learn the sparse code for that cluster
        dict_learn = DictionaryLearning(transform_algorithm='lasso_cd')
        dict_learn.fit(data_cluster)
        dics.append(dict_learn)

    return dics

# train an ensemble of classifiers
# one according to each cluster
def train_ensemble(each_cluster, dataq, labelq, train, dics, total_features = 64):
    ensemble = []

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

        # encode the features according to the dictionary of that cluster
        data_cluster = dics[i].transform(data_cluster)

        # train the classifier and add it to the list of the ensemble
        ranksvm = RankSVM().fit(data_cluster, labels_cluster)
        ensemble.append(ranksvm)

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



#ld.get_letor_3(64)
#data_total_raw = ld.load_pickle('','data_cluster.pickle')

data, labels = ld.load_pickle_total('', 'data_per_q_total.pickle', 'relevance_per_q_total.pickle')

queries = np.asanyarray(natsorted(data.keys()), dtype=object)

train, test = train_test_split(queries, test_size=0.33, random_state=42)
print 'Training set size: %i, Testing set size: %i' % (train.shape[0], test.shape[0])

data_cluster_train = query_features(train, 24, 50, 64, data)
data_cluster_test = query_features(test, 24, 50, 64, data)

data_cluster_train_ds = sc.pdist(data_cluster_train, 'mahalanobis')
data_cluster_train_ds = sc.squareform(data_cluster_train_ds)

plt.figure(1)
plt.imshow(data_cluster_train_ds)
plt.colorbar()
plt.title('Initial dissimilarity')

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

test_dis = parse_test_dis(data_cluster_train, data_cluster_test)
estimates = estimate_cluster_prob(test_dis, dpgmm, clusters)

print test_dis.shape
print estimates.shape

# here we will create a sparse coding representation
# for each cluster
dics = create_dics(each_cluster, data, train)


# train the ensemble according to those dictionaries
ensemble = train_ensemble(each_cluster, data, labels, train, dics)

predictions_per_q = predict_ensemble(ensemble, dics, estimates, test, data)




