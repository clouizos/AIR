import numpy as np
import cPickle as pickle
from dpRank import User


# dummy processing of the chain
# in order to produce some query clusters

with open('clean_total_50iter.pickle', 'rb') as handle:
    chain = pickle.load(handle)

tot_comp = 0
for elem in chain:
    if elem['gamma_k'].shape[0] > tot_comp:
        tot_comp = elem['gamma_k'].shape[0]

print tot_comp

cnt = 0
users_q = {}
q = {}
for elem in chain:
    user_obj = elem['users_objects']
    for user in user_obj:
        if cnt == 0:
            for i in xrange(len(user_obj[user].queries)):
                q_dist = np.zeros(tot_comp)
                q_dist[user_obj[user].assign[i]] += 1
                q[user_obj[user].queries[i]] = q_dist
                #print q_dist
                users_q[user] = q.copy()
        else:
            for i in xrange(len(user_obj[user].queries)):
                q_dist = users_q[user][user_obj[user].queries[i]]
                q_dist[user_obj[user].assign[i]] += 1
                users_q[user][user_obj[user].queries[i]] = q_dist
    cnt += 1

for user in users_q:
    for q in users_q[user]:
        dist = users_q[user][q]
        dist /= len(chain)
        users_q[user][q] = dist
            #user_obj[user].queries

clusters = []
for i in range(tot_comp):
    cluster = []
    clusters.append(cluster)

for user in users_q:
    for q in users_q[user]:
        clusters[users_q[user][q].argmax()].append((q, users_q[user][q].max()))

clusters_ = [cluster for cluster in clusters if cluster != []]

for cluster in clusters_:
    cluster.sort(key=lambda x: x[1], reverse=True)
    #cluster = cluster[::-1]


print clusters_
print len(clusters_)

for cl in clusters_:
    print cl
    print

with open('q_clusters.pickle', 'wb') as handle:
    pickle.dump(clusters_, handle)
