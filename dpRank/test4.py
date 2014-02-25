# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from genDummyData import gen_dummy
from natsort import natsorted
import math
import itertools
import utils as u

# hyperparameters
mu0 = 0
sigma0 = 0.5
alpha0 = 1.0
beta0 = 0.5
alpha = 1.0
eta = 0.1

# k is mentioned to infinite in the paper
# set to a very large number for now
# will create a trunctated DP that approximates
# the infinite one
# init number of groups, will change according to the data
K = 5

# nr of data points
nr_data = 1000

# nr of query features
T = 5
# nr of document features
V = 65
# nr of users
N = 20

    
nr_docs = 100
nr_queries = 20

# get some dummy data
docs, queries, user_q, q_doc, user_clicks = gen_dummy(nr_docs, nr_queries, N)

# gibbs sampler params
gibbs_iter = 1000
thinning = 5
burnin = 0.2*gibbs_iter
 
      
''' priors of the G0 '''
# prior means
mu_kt = u.random_normal(mu0, sigma0, T)
# prior precision
tau_kt = u.random_gamma(alpha0, beta0, T)
# prior beta_ku
beta_ku = u.random_normal(0, alpha0, V)


# params according to the draws from the truncated DP
# essentially each group has its own means, variances and betas
theta = np.zeros((K,2*T + V))    
for i in range(K):
    theta[i,0:T] = mu_kt.random()
    theta[i, T: 2*T] = tau_kt.random()
    theta[i, 2*T:] = beta_ku.random()

p = u.stick(alpha, K)
dp =  u.DP(K, p, theta)   

#joint_q_d(dp, user_clicks, user_q, docs, T)
        
# proof check, check how many times a specific parameter vector exists
#proof_check(theta, dp, 1)

# double layer of the DP
# needs to be done for each user
p2 = u.stick(eta, K)
theta_k = np.zeros((N, K))
for i in range(N):
    theta_k[i,:] = u.stick(alpha, K)

dp2 = u.DP(N, p2, theta_k)

#u.proof_check(theta_k, dp2, 2)

# initial gamma
gamma_k = u.stick(alpha, K)

# init random assignment of queries to groups 
group_query = np.random.choice(range(K), len(queries.keys()))
group_assign = dict(zip(natsorted(queries.keys()), group_query))

# sampling of c is performed for every query from the users in the given collection
def sample_c(user, user_q, K, eta, theta_c):
    q_assign = []
    # for every query
    for query in user_q[user]:
        assign = []
        # get probability of that query belonging to each group
        for i in range(K):
            p_q_d_k = u.joint_q_d(theta_c[i,:], queries[query], user_clicks[user+':'+query], docs, T)
            assign.append(p_q_d_k)
        q_assign.append(assign)
    
    return q_assign

for user in natsorted(user_q):
    
    print sample_c(user, user_q, K, eta, theta)
    
    break
  
