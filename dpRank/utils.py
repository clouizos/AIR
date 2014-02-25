from __future__ import division
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import math
import itertools
#from nzmath.combinatorial import stirling1



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
def generate_indices(x, k):
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
    x = np.random.multinomial(k, p)
    b = generate_indices(x, k)
    return theta[b, :]

def stick(alpha, k):
    # beta distribution for constructing the stick
    betas = np.random.beta(1.0, alpha, k)
    remaining_pieces = np.append(1, np.cumprod(1 - betas[:-1]))
    p = betas * remaining_pieces
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
 
 
def joint_q_d(theta_c, query, click_list, docs, T):
    # get the parameters for the group
    mu_c = theta_c[0:T]
    sigma_c = theta_c[T:2*T]
    beta_c = theta_c[2*T:]
    
    # get the corresponding clicks and non clicks
    ind_clicked = [i for i, v in enumerate(click_list) if v[1] == 1]
    ind_not_clicked = [i for i, v in enumerate(click_list) if v[1] == 0]
    
    #print ind_clicked, ind_not_clicked 
    
    # create all the combinations
    comb = list(itertools.product(*[ind_clicked, ind_not_clicked]))
    pair_prefs = []
    
    # for each element find the pairwise ranking preferences
    for elem in comb:
        # probably we need log probability in order to avoid underflows
        #pair_prefs.append(pair_pref(beta_c, docs[click_list[elem[0]][0]], docs[click_list[elem[1]][0]]))
        pair_prefs.append(np.log(pair_pref(beta_c, docs[click_list[elem[0]][0]], docs[click_list[elem[1]][0]])))
    
    if len(pair_prefs) > 1:
        rank_prob = np.sum(pair_prefs)
        #rank_prob = np.prod(pair_prefs)
    else:
        rank_prob = np.log(10**-150)
        #rank_prob = 10 ** -100
        
    # estimate the probability of the query given the mu and sigma
    # since the covariance matrix is diagonal simply multiply the 
    # univariate estimates
    probs = []
    for i in range(sigma_c.shape[0]):
        prob = norm(mu_c[i], sigma_c[i]).pdf(query[i])
        if prob > 0:
            probs.append(np.log(prob))
        else:
            probs.append(np.log(10 ** -150))
        #probs.append(norm(mu_c[i], sigma_c[i]).pdf(query[i]))
    

    p_q = np.sum(probs)
    #p_q = np.prod(probs)
    
    return p_q + rank_prob
    #return p_q * rank_prob

# eq.1 of the paper
def pair_pref(beta_c, doc_i, doc_j):
    return 1/(1 + np.exp(- np.dot(beta_c.T, doc_i - doc_j )))
    

#def uns_stirling(n,k):
#    return stirling1(n,k)
    

