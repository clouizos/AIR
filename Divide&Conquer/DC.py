# -*- coding: utf-8 -*-
import numpy as np
from sklearn.mixture import DPGMM
import load_data as ld
import scipy.spatial.distance as sc
import matplotlib.pyplot as plt
from rankSVM import RankSVM

#ld.get_letor_3(64)
data_total_raw = ld.load_pickle('','data_cluster.pickle')
data_label_dic = ld.load_pickle('','data_label.pickle')

data = data_label_dic['data']
labels = data_label_dic['labels']

print data.shape
print labels.shape

train = data[0:int(0.7 * data.shape[0]), :]
train_labels = labels[0:int(0.7 * data.shape[0])]
test = data[int(0.7 * data.shape[0]):,:]
test_labels = labels[int(0.7 * data.shape[0]):]

print train.shape
print test.shape
print train_labels.shape
print test_labels.shape

ranksvm = RankSVM().fit(train[0:50,:], train_labels[0:50])
print ranksvm.coef_
print ranksvm.coef_.shape

# produce a dissimilarity space
data_total = sc.pdist(data_total_raw, 'mahalanobis')
data_total = sc.squareform(data_total)
plt.imshow(data_total)
plt.colorbar()
plt.title('Initial dissimilarity')
plt.show()

# fit the data according to that space with DPGMM
dpgmm = DPGMM(covariance_type='diag', alpha=2., n_iter=1000, n_components=10)
dpgmm.fit(data_total)
prediction = dpgmm.predict(data_total)
#print prediction
print 'Found %i clusters!' % np.unique(prediction).shape[0]
print np.unique(prediction)

# create the reordered input data according to the clusters
data_cluster = np.zeros((1, data_total_raw.shape[1]))
clusters = np.unique(prediction)
clustered_indices = []
for i in xrange(clusters.shape[0]):
    clustered_indices.append(prediction == clusters[i])
    data_cluster = np.vstack((data_cluster, data_total_raw[prediction == clusters[i],:]))

# remove the useless first element
data_cluster = data_cluster[1:, :]

# visuallize the clustering, it should be a block matrix
data_vis = sc.pdist(data_cluster, 'mahalanobis')
data_vis = sc.squareform(data_vis)
plt.imshow(data_vis)
plt.colorbar()
plt.title('Clustered dissimilarity')
plt.show()

# parse the probability distribution over the clusters
# for each query
prob_under_cluster = np.zeros((data_total.shape[0], clusters.shape[0]))
for x in xrange(prob_under_cluster.shape[0]):
    tot_probs = dpgmm.predict_proba(data_total[x,:].reshape((1,data_total[x,:].shape[0])))[0,clusters[:]]
    tot_probs /= tot_probs.sum()
    prob_under_cluster[x, :] = tot_probs

