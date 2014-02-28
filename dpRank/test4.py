# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from genDummyData import gen_dummy
from natsort import natsorted
import utils as u
import pymc

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
gibbs_iter = 20
thinning = 5
burnin = int(0.2 * gibbs_iter)
 
      
''' priors of the G0 '''
# prior means
mu_kt = u.random_normal(mu0, sigma0, T)
# prior precision
tau_kt = u.random_gamma(alpha0, beta0, T)
# prior beta_ku
beta_ku = u.random_normal(0, alpha0, V)
    
'''   
# double layer of the DP
# needs to be done for each user
p2 = u.stick(eta, K)
theta_k = np.zeros((N, K))
# for all users
for i in range(N):
    theta_k[i,:] = u.stick(alpha, K)

# generate the pi's
pis = u.DP(N, p2, theta_k)
'''

#print pis
#u.proof_check(theta_k, pis, 2)


# initial gamma
gamma_k = u.stick(alpha, K)
# gamma_e auxiliary variable
gamma_e = 1 - np.sum(gamma_k)

# params according to the draws from the truncated DP
# essentially each group has its own means, variances and betas
# initial parameters
theta_c = np.zeros((K,2*T + V))
# active thetas    
for i in range(K):
    theta_c[i,0:T] = mu_kt.random()
    theta_c[i, T: 2*T] = tau_kt.random()
    theta_c[i, 2*T:] = beta_ku.random()

# theta_e auxiliary variable
theta_e = np.hstack((mu_kt.random(), tau_kt.random(), beta_ku.random()))

# remove queries that have no users assigned (only applicable to the dummy data)
for query in natsorted(queries):
    if len([i for i in user_q if query in user_q[i]]) == 0:
        del queries[query]
  
# sampling of c is performed for every query from the users in the given collection     
def sample_c(queries, user_q, eta, theta_c, theta_e, gamma_k, gamma_e, docs, active_components, user_clicks):
    # for all queries
    # get the number of active components
    K = theta_c.shape[0]
    
    for query in natsorted(queries):
        #print 'checking query ' + query
        # remove the query being checked right now if it exists
        # in a component
        for component in active_components:
            if query in component:
                component.remove(query)
            #print len(component)
            #component = [q for q in component if q!=query]
            #print len(component)
            
        # users that have that query
        users = [i for i in user_q if query in user_q[i]]
        #print 'users for that query: ' +str(users)
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
                    try:
                        term = np.log(eta * gamma_k[i] + mu_k)
                    except:
                        # sometimes it throws an exception about finding zero at 
                        # the log, can happen if gamma_k = 0  and mu_k = 0
                        #term = np.log(10 ** -200)
                        term = -1 * np.inf
                    p_q_d_k = u.joint_q_d(theta_c[i,:], queries[query], user_clicks[user+':'+query], docs, T)
                else:   # checking auxiliary component
                    term = np.log(eta*gamma_e + mu_k)
                    p_q_d_k = u.joint_q_d(theta_e, queries[query], user_clicks[user+':'+query], docs, T)
                
                # since we are use log probabilities, add the ones from each user
                assign[i] += term + p_q_d_k
        
        
        # since sometimes we have zero gamma, we dont want to assign a query there
        # set it to -inf and then find it here and replace it with 0 so that
        # that component wont be sampled
        assign[assign == (-1 * np.inf)] = 0.0
        # maybe its better to normalize and pick one at random with weighted probability
        # by the assign vector?
        assign /= assign.sum()
        # get a random assignment according to the assign probabilities
        most_prob_k = np.random.choice(range(len(assign)), 1, p=assign)[0]
        # get the most probable assignment
        #most_prob_k = assign.argmax()
        
        
        #print assign, most_prob_k
        
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
        
            #print active_components
            #print np.shape(theta_c)
            #print gamma_k, gamma_e
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
        #print active_components
        #print theta_c.shape
        #print gamma_k, gamma_e
            
    #print active_components
    #print theta_c.shape
    #print gamma_k, gamma_e
    
    # do a final check for empty components
    ind = []
    # counter to get the indices
    cnt = 0
    for component in active_components:
        # if empty active component
        if not component:
            ind.append(cnt)
        cnt += 1 
    
    active_components = [v for i, v in enumerate(active_components) if i not in ind]
        
    K = K - len(ind)
    theta_c = np.delete(theta_c, ind, axis = 0)
    gamma_k = np.delete(gamma_k, ind)
    gamma_e = 1 - np.sum(gamma_k)
    
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
                       
            if h_candidates.shape[0] > 0 and not np.array_equal(h_candidates, np.zeros(h_candidates.shape[0])):
                h_candidates /= h_candidates.sum()
                # max one
                #h_uik[i,j] = h_candidates.argmax() + 1
                # random one
                h_uik[i,j] = np.random.choice(range(m_uik[i,j])[1:], 1, p=h_candidates)[0]
    
    dir_params = np.zeros(K + 1)
    for k in range(K + 1):
        if k < K:
            dir_params[k] = np.sum(h_uik[:,k])
        else:
            dir_params[k] = alpha
    
    # take the expectation or a random one from that dirichlet?
    #g = np.random.dirichlet(dir_params)
    g = pymc.dirichlet_expval(dir_params)
    
    gamma_k = g[:-1]
    gamma_e = g[-1]
    
    #print gamma_k, gamma_e
    return gamma_k, gamma_e

def sample_theta(active_components, theta_c, queries):
    
    # I think this should be K (the number of active_components, but K != len(active_components))
    for k in range(len(active_components)):

        mu_k = theta_c[k, 0:T]
        sigma_k = theta_c[k, T: 2*T]
        nk = len(active_components[k])
        queries_of_k = active_components[k]

        # per feature we will update mu and sigma
        for t in range(T):    
            
            # can this be made more efficient?
            # is it actually correct?
            sum_of_user_queries = 0
            for q in queries_of_k:
                sum_of_user_queries += queries[q][t]

            current_sigma_kt = sigma_k[t]
            current_mu_kt = mu_k[t]
            lambda_kt = 1 / current_sigma_kt
            
            sigma_n_kt = 1 / ((nk*lambda_kt) + 1 / sigma0)
            mu_n_kt = sigma_n_kt * ((mu0 / sigma0) + (sum_of_user_queries / current_sigma_kt))

            # UPDATE mu here to use to compute sigma_kt
            current_mu_kt = np.random.normal(mu_n_kt, sigma_n_kt)

            sum_of_user_queries_minus_mu = 0
            for q in queries_of_k:
                sum_of_user_queries_minus_mu += (queries[q][t] - current_mu_kt)**2

            alpha_n_kt = alpha0 + nk/2
            beta_nk_t = beta0 + 1/2 * sum_of_user_queries_minus_mu

            current_sigma_kt = np.random.gamma(alpha_n_kt, beta_nk_t, 1)

            # write result to subvector
            mu_k[t] = current_mu_kt
            sigma_k[t] = current_sigma_kt

        # write vector update to theta
        theta_c[k, 0:T] = mu_k
        theta_c[k, T: 2*T] = sigma_k

    return theta_c


        
def sample_weights(active_components, theta_c, queries, user_q, docs, user_clicks, tot_iter = 20):

    prior_params = [alpha0, mu0, sigma0, T, V]
    new_theta_c = np.zeros(theta_c.shape)
    
    # for every group
    for k in range(len(active_components)):
        #beta_k = theta_c[k, 2*T:]
        users = []
        queries_check = set(active_components[k])
        for user in natsorted(user_q):
            if len(set(user_q[user]).intersection(queries_check)) > 1:
                users.append(user)
                
        chain = u.HMC(tot_iter, theta_c[k,:], queries, active_components[k], users, user_q, user_clicks, docs, prior_params)
        # Discard first 20% of MCMC chain
        clean = []
        for n in range(int(0.2 * tot_iter),len(chain)):
            # thin the samples
            if (n % 2 == 0):
                clean.append(chain[n])
                
        new_theta_c[k,:] = np.mean(clean, axis = 0)
    
    return new_theta_c


tot_chain = []
for i in range(gibbs_iter):
    print 'Doing iteration ' + str(i+1)+' ...'
    # make this iteratively?
    params = {}
    active_components, gamma_k, gamma_e, theta_c, theta_e = sample_c(queries,     # here you need to take the new theta_c because theta is the old one 
    # (according to the old components)
                                            user_q, eta, theta_c, theta_e, gamma_k, gamma_e, 
                                            docs, active_components, user_clicks)
    
    old_components = [list(component) for component in active_components]
    params['active_components'] = old_components
        
    # since K changes locally inside the sample_c, get the new one
    K = len(active_components)

    gamma_k, gamma_e = sample_g(active_components, gamma_k, gamma_e, alpha, eta, user_q)
    
    old_gamma_k = np.copy(gamma_k)
    old_gamma_e = np.copy(gamma_e)
    params['gamma_k'] = old_gamma_k
    params['gamma_e'] = old_gamma_e

    theta_c = sample_theta(active_components, theta_c, queries)

    theta_c = sample_weights(active_components, theta_c, queries, user_q, docs, user_clicks)
    
    old_theta_c = np.copy(theta_c)
    old_theta_e = np.copy(theta_e)
    params['theta_c'] = old_theta_c
    params['theta_e'] = old_theta_e
    
    print active_components
    print gamma_k, gamma_e
    print theta_c.shape
    
    
    tot_chain.append(params.copy())

print
print 'total chain:'
for elem in tot_chain:
    print elem['active_components']
print
    
 # Discard first 20% of MCMC chain (burnin)
clean = []
for n in range(burnin,len(tot_chain)):
    # thin the samples
    if (n % thinning == 0):
        clean.append(tot_chain[n])

print 'clean chain:'
for elem in clean:
    print elem['active_components']
    

