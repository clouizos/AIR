# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from genDummyData import gen_dummy
from natsort import natsorted
import math
import itertools
import utils as u
from pymc import dirichlet_expval

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

# nr of query features
T = 5
# nr of document features
V = 65
# nr of users
N = 20

# args for the dummy data generator   
nr_docs = 100
nr_queries = 20

# active components (components that have observations)
active_components = [[] for i in range(K)]

# get some dummy data
docs, queries, user_q, q_doc, user_clicks = gen_dummy(nr_docs, nr_queries, N)

# gibbs sampler params
gibbs_iter = 1000
thinning = 5
burnin = 0.2 * gibbs_iter
 
      
''' priors of the G0 '''
# prior means
mu_kt = u.random_normal(mu0, sigma0, T)
# prior precision
tau_kt = u.random_gamma(alpha0, beta0, T)
# prior beta_ku
beta_ku = u.random_normal(0, alpha0, V)
    
    
# double layer of the DP
# needs to be done for each user
p2 = u.stick(eta, K)
theta_k = np.zeros((N, K))
# for all users
for i in range(N):
    theta_k[i,:] = u.stick(alpha, K)

# generate the pi's
pis = u.DP(N, p2, theta_k)

#print pis
#u.proof_check(theta_k, pis, 2)


# initial gamma
gamma_k = u.stick(alpha, K)
# gamma_e auxiliary variable
gamma_e = 1 - np.sum(gamma_k)

# params according to the draws from the truncated DP
# essentially each group has its own means, variances and betas
# initial parameters
theta = np.zeros((K,2*T + V))
# active thetas    
for i in range(K):
    theta[i,0:T] = mu_kt.random()
    theta[i, T: 2*T] = tau_kt.random()
    theta[i, 2*T:] = beta_ku.random()

# theta_e auxiliary variable
theta_e = np.hstack((mu_kt.random(), tau_kt.random(), beta_ku.random()))

# remove queries that have no users assigned (only applicable to the dummy data)
for query in natsorted(queries):
    if len([i for i in user_q if query in user_q[i]]) == 0:
        del queries[query]
  
# sampling of c is performed for every query from the users in the given collection     
def sample_c(queries, user_q, eta, theta_c, theta_e, gamma_k, gamma_e, docs, active_components):
    # for all queries
    # get the number of active components
    K = theta_c.shape[0]
    
    for query in natsorted(queries):
        print 'checking query ' + query
        # users that have that query
        users = [i for i in user_q if query in user_q[i]]
        print 'users for that query: ' +str(users)
        # log probabilities of assignment of a query to a component
        assign = np.zeros(K+1)
        
        # for each user estimate the probability of the query belonging to group k
        for user in natsorted(users):
            # get the other queries of the user except the one estimated right now
            other_queries = set(natsorted(user_q[user])) - set([query])
            
            # for each component + the auxiliary one
            for i in range(K+1):
                mu_k = 0
                # find how many of the other queries of the user 
                # belong to the group k (only check active components)
                if i < K:
                    for other in other_queries:
                        if other in active_components[i]:
                            mu_k += 1
                if i < K: # checking assumed active components
                    term = np.log(eta*gamma_k[i] + mu_k)
                    p_q_d_k = u.joint_q_d(theta_c[i,:], queries[query], user_clicks[user+':'+query], docs, T)
                else:   # checking auxiliary component
                    term = np.log(eta*gamma_e + mu_k)
                    p_q_d_k = u.joint_q_d(theta_e, queries[query], user_clicks[user+':'+query], docs, T)
                
                # since we are use log probabilities, add the ones from each user
                assign[i] += term + p_q_d_k
        
        # maybe its better to normalize and pick one at random with weighted probability
        # by the assign vector?
        most_prob_k = assign.argmax() # get the most probable assignment
        
        print assign, most_prob_k
        
        # if we have a component that is considered active
        if most_prob_k < K:

            # in the existing active groups
            active_components[most_prob_k].append(query)
            
            # filter out the empty ones
            ind = []
            # counter to get the indices
            cnt = 0
            for component in active_components:
                # if empty active component
                if not component:
                    ind.append(cnt)
                cnt += 1
            # get the new non-empty active components
            active_components = [v for i, v in enumerate(active_components) if i not in ind]
            
            # set new K, delete params for the K's that got removed
            # and update gamma_e
            K = K - len(ind)
            theta_c = np.delete(theta_c, ind, axis = 0)
            gamma_k = np.delete(gamma_k, ind)
            gamma_e = 1 - np.sum(gamma_k)
        
            print active_components
            print np.shape(theta_c)
            print gamma_k, gamma_e
        # if we have the auxiliary component
        else:
            
            # add it to the active components
            active_components.append([query])
            
            # add the parameters of the auxiliary component to the active ones
            # and determine the new auxiliary theta_e
            theta_c = np.vstack((theta_c, theta_e))
            theta_e =  np.hstack((mu_kt.random(), tau_kt.random(), beta_ku.random()))
            # new gamma for the new component and new auxiliary gamma_e
            b = np.random.beta(1.0, alpha)
            gamma_k = np.append(gamma_k, b*gamma_e)
            gamma_e = (1 - b) * gamma_e
            # new K
            K += 1
        print active_components
        print theta_c.shape
        print gamma_k, gamma_e
            
    print active_components
    print theta_c.shape
    print gamma_k, gamma_e
    
    # return basically everything (need to check what is actually needed)
    return active_components, gamma_k, gamma_e, theta_c, theta_e


def sample_g(active_components, gamma_k, gamma_e, alpha, eta, user_q):
    # get the active components
    K = gamma_k.shape[0]
    # init the random variable h
    m_uik = np.zeros((len(user_q), K), dtype = int)
    
    # find the m_uik for every user
    for i in range(len(active_components)):
        comp_qs  = set(active_components[i])
        cnt = 0
        for user in natsorted(user_q.keys()):
            users_q = set(user_q[user])
            m_uik[cnt, i] = len(comp_qs.intersection(users_q))
            cnt += 1
    
             
    # it seems that we need to sample h first
    # and based on that we sample gamma  
    h_uik = np.zeros((len(user_q), K)) 
    for i in range(len(user_q)):
        for j in range(K):
            # do not keep the first value (0) since 
            # the stirling1 will produce 0 so it doesn't
            # affect the probabilities
            h_candidates = np.asfarray([u.stirling1(m_uik[i,j], h) * \
                            (eta * gamma_k[j])** h for h in range(m_uik[i,j])[1:]])
            # normalize and pick the maximum
            # again maybe we need to pick a random one according to those probabilities?
            h_candidates /= h_candidates.sum()
            if h_candidates.shape[0] > 0:
                h_uik[i,j] = h_candidates.argmax() + 1
            
    dir_params = np.zeros(K + 1)
    for k in range(K + 1):
        if k < K:
            dir_params[k] = np.sum(h_uik[:,k])
        else:
            dir_params[k] = alpha

    g = dirichlet_expval(dir_params)
    gamma_k = g[:-1]
    gamma_e = g[-1]
    
    print gamma_k, gamma_e
    return gamma_k, gamma_e


active_components, gamma_k, gamma_e, theta_c, theta_e = sample_c(queries, 
                                            user_q, eta, theta, theta_e, gamma_k, gamma_e, docs, active_components)

gamma_k, gamma_e = sample_g(active_components, gamma_k, gamma_e, alpha, eta, user_q)

