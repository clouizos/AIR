# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from genDummyData import gen_dummy
from natsort import natsorted
import math
import itertools

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
k = 10


def remaining_prop(gamma_k):
    return 1 - np.sum(gamma_k)

# nr of data points
nr_data = 1000

# nr of query features
T = 5
# nr of document features
V = 65
# nr of users
N = 10

    
nr_docs = 100
nr_queries = 20

# get some dummy data
docs, queries, user_q, q_doc, user_clicks = gen_dummy(nr_docs, nr_queries, N)

# gibbs sampler params
gibbs_iter = 1000
thinning = 5
burnin = 0.2*gibbs_iter

class random_normal:
    mean = 0
    variance = 1
    size = None
 
    def __init__(self, mu0, sigma0, T):
        self.mean = mu0
        self.variance = sigma0
        self.size = T
        
    def random(self):
        return np.random.normal(self.mean, self.variance, size = self.size)
        
class random_gamma:
    alpha = 0
    beta = 1
    size = None
    
    def __init__(self, alpha0, beta0, T):
        self.alpha = alpha0
        self.beta = beta0
        self.size = T
        
    def random(self):
        return np.random.gamma(self.alpha, self.beta, size = self.size)

# get the categorial results (each experiment directly and not the resulting sum)    
def generate_indices(x):
    b = np.zeros(k, dtype=int)
    upper = np.cumsum(x); lower = upper - x
    for value in range(len(x)):
	   b[lower[value]:upper[value]] = value
    # mix up the order, in-place, if you care about them not being sorted
    np.random.shuffle(b)
    return b     
    
# p is the probabilities from the stick-breaking construction
# k is the number of samples (experiments)  
def DP(k, p, theta):
    print p
    x = np.random.multinomial(k, p)
    print x
    b = generate_indices(x)
    print b
    return theta[b, :]

def stick(alpha, k):
    # beta distribution for constructing the stick
    betas = np.random.beta(1.0, alpha, k)
    remaining_pieces = np.append(1, np.cumprod(1 - betas[:-1]))
    p = betas * remaining_pieces
    print 'prv:' +str(p)
    print 'nxt:' + str(p/p.sum())
    return p/p.sum()
    
def proof_check(theta, dp, fig_n):
    bins = np.zeros(np.shape(theta)[0])
    for i in range(np.shape(theta)[0]):
        row = theta[i,:]
        indices = [j for j in range(np.shape(dp)[0]) if np.array_equal(dp[j, :], row) ]
        bins[i] = len(indices)

    # create a histogram of the draws of the parameters
    plt.figure(fig_n)
    plt.bar(np.arange(len(bins)), bins, width = 0.01)
    plt.show()
 
# eq.2 of the paper   
def joint_q_d(theta_c, user_clicks, user_q, docs, T):
    mu_c = theta_c[:,0:T]
    sigma_c = theta_c[:,T:2*T]
    beta_c = theta_c[:,2*T:]
    
    # p(q| mu_c, sigma_c)
    p_q = np.zeros((theta_c.shape[0], sigma_c[1,:].shape[0]))
    for i in range(theta_c.shape[0]):
        # generation of queries per each group, C x T matrix
        p_q[i,:] = np.random.multivariate_normal(mu_c[i,:], sigma_c[i,:] * np.eye(sigma_c[i,:].shape[0]))
    
    
    #prob_dict = {}
    # p(D|q, beta_c)
    #print pair_pref(beta_c)
    for user in natsorted(user_q.keys()):
        for q in natsorted(user_q[user]):
            D = user_clicks[user+':'+q]
        
            ind_clicked = [i for i, v in enumerate(D) if v[1] == 1]
            ind_not_clicked = [i for i, v in enumerate(D) if v[1] == 0]
            print ind_clicked, ind_not_clicked 
            
            comb = list(itertools.product(*[ind_clicked, ind_not_clicked]))
            pair_prefs = []
            pair_prefs2 = []
            # really, really small value for init prob
            tot_prob = 10 ** -208
            #for i in range(k):
            for elem in comb:
                # probably we need log probability in order to avoid underflows
                pair_prefs.append(np.log(pair_pref(beta_c[0,:], docs[D[elem[0]][0]], docs[D[elem[1]][0]])))
                pair_prefs2.append(pair_pref(beta_c[0,:], docs[D[elem[0]][0]], docs[D[elem[1]][0]]))
            
            if len(pair_prefs) > 1:
                #print pair_prefs   
                #print np.cumprod(pair_prefs2)[-1]
                tot_prob = np.prod(pair_prefs2)
                #print np.sum(pair_prefs)
                
            print tot_prob * p_q[0,:]
            #if len(ind_clicked) > 1:
            #    print pair_pref(beta_c[1,:], docs[D[ind_clicked[0]][0]], docs[D[ind_not_clicked[0]][0]])
            
            #print D
            #print q
            break
        break
    
# eq.1 of the paper
def pair_pref(beta_c, doc_i, doc_j):
    return 1/(1 + np.exp(- np.dot(beta_c.T, doc_i - doc_j )))

      
      
''' priors of the G0 '''
# prior means
mu_kt = random_normal(mu0, sigma0, T)
# prior precision
tau_kt = random_gamma(alpha0, beta0, T)
# prior beta_ku
beta_ku = random_normal(0, alpha0, V)


# params according to the draws from the truncated DP
# essentially each group has its own means, variances and betas
theta = np.zeros((k,2*T + V))    
for i in range(k):
    theta[i,0:T] = mu_kt.random()
    theta[i, T: 2*T] = tau_kt.random()
    theta[i, 2*T:] = beta_ku.random()

p = stick(alpha, k)
dp =  DP(k, p, theta)   

joint_q_d(dp, user_clicks, user_q, docs, T)
        
# proof check, check how many times a specific parameter vector exists
proof_check(theta, dp, 1)

# double layer of the DP
p2 = stick(eta, k)
theta_k = np.zeros((k, k))
for i in range(k):
    theta_k[i,:] = stick(alpha, k)
    
dp2 = DP(k, p2, theta_k)
print dp2.shape

proof_check(theta_k, dp2, 2)

# u: specific user
# pi_u: pi's for that user (double DP result)
# c_uj: latent group of user j
# sampling of c is performed for every query from the users in the given collection
init_gamma = stick(alpha, k)
# init random assignment of queries to groups 
group_query = np.random.choice(range(k), len(queries.keys()))
group_assign = dict(zip(natsorted(queries.keys()), group_query))

for user in natsorted(user_q):
    #print user
    #print user_q[user]
    # nr queries of the user
    mu = len(user_q[user])
    muk = np.zeros(k)
    for i in natsorted(user_q[user]):
        group = group_assign[i]
        muk[group] += 1 
        
    #print muk    
    norm_con = math.gamma(eta)/math.gamma(eta+mu)
    #print norm_con
    x = np.prod([math.gamma(eta*init_gamma[i] + muk[i])/math.gamma(eta*init_gamma[i]) for i in range(k)])

    #print norm_con * x
    for q in natsorted(user_q[user]):
        clicks = user_clicks[user+':'+q]
        #print q, user_clicks[user+':'+q]
