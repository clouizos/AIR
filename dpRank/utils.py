from __future__ import division
import scipy
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import math, itertools, random
from scipy.optimize import check_grad
import multiprocessing as mpc
import shelve

# we will take log(0) = -Inf so turn off this warning
np.seterr(divide='ignore')


class random_normal:
    mean = 0
    variance = 1
    size = None

    def __init__(self, mu0, sigma0, T):
        self.mean = mu0
        self.variance = sigma0
        self.size = T

    def random(self):
        return np.random.normal(self.mean, self.variance, size=self.size)


class random_gamma:
    alpha = 0
    beta = 1
    size = None

    def __init__(self, alpha0, beta0, T):
        self.alpha = alpha0
        self.beta = beta0
        self.size = T

    def random(self):
        return np.random.gamma(self.alpha, self.beta, size=self.size)


# get the categorial results (each experiment directly and not the resulting sum)
def generate_indices(x, k):
    b = np.zeros(k, dtype=int)
    upper = np.cumsum(x)
    lower = upper - x
    for value in range(len(x)):
        b[lower[value]:upper[value]] = value
    # mix up the order, in-place, if you care about them not being sorted
    np.random.shuffle(b)
    return b


# p is the probabilities from the stick-breaking construction
# k is the number of samples (experiments)
def DP(k, p, theta):
    x = np.random.multinomial(k, p)
    # below is testing, ignore
    #indices = []
    #for i in range(len(x)):
    #    if x[i] > 0:
    #        indices.append(i)
    indices = generate_indices(x, k)
    return theta[indices, :]


def stick(alpha, k):
    # beta distribution for constructing the stick
    betas = np.random.beta(1.0, alpha, k)
    remaining_pieces = np.append(1, np.cumprod(1 - betas[:-1]))
    p = betas * remaining_pieces
    # do not normalize in order to have values
    # on the auxiliary gamma_e variable later
    return p


def proof_check(theta, dp, fig_n):
    bins = np.zeros(np.shape(theta)[0])
    for i in range(np.shape(theta)[0]):
        row = theta[i, :]
        indices = [j for j in range(np.shape(dp)[0]) if np.array_equal(dp[j, :], row)]
        bins[i] = len(indices)

    # create a histogram of the draws of the parameters
    plt.figure(fig_n)
    plt.bar(np.arange(len(bins)), bins, width=0.01)
    plt.show()


# eq.2 of the paper
#def joint_q_d(theta_c, query, click_list, docs, T, q_doc_features, str_query):
def joint_q_d(funcin):
    q_doc_features = shelve.open('/virdir/Scratch/saveFR2DP_330/docs.db')
    theta_c, query, click_list, docs, T, str_query = funcin
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
        pair_prefs.append(np.log(pair_pref(beta_c, np.squeeze(q_doc_features[str_query+':'+click_list[elem[0]][0].split(':')[1]]),
                                                   np.squeeze(q_doc_features[str_query+':'+click_list[elem[1]][0].split(':')[1]]))))
    #print pair_prefs
    if len(pair_prefs) >= 1:
        rank_prob = np.sum(pair_prefs)
        #rank_prob = np.prod(pair_prefs)
    else:
        #print "no pair preferences"
        rank_prob = np.log(10**-200)
        #rank_prob = 10 ** -100

    # estimate the probability of the query given the mu and sigma
    # since the covariance matrix is diagonal simply multiply the
    # univariate estimates
    probs = []
    for i in range(sigma_c.shape[0]):
        prob = scipy.stats.norm(mu_c[i], sigma_c[i]).pdf(query[i])
        #print prob
        if prob > 0:
            probs.append(np.log(prob))
        else:
            probs.append(np.log(10 ** -200))
        #probs.append(norm(mu_c[i], sigma_c[i]).pdf(query[i]))
    #print probs
    p_q = np.sum(probs)
    #p_q = np.prod(probs)

    return p_q + rank_prob

    #return p_q * rank_prob


# eq.1 of the paper
def pair_pref(beta_c, doc_i, doc_j):
    return 1/(1 + np.exp(- np.dot(beta_c.T, doc_i - doc_j)))


# gradient of eq.1 of the paper (needed for HMC)
def grad_pair_pref(funcin):
    beta_c, doc_i, doc_j = funcin
    grad = np.asfarray([(1/(1 + np.exp(- beta_c[i] * (doc_i[i] - doc_j[i]))))
            * (1 - (1/(1 + np.exp(- beta_c[i] * (doc_i[i] - doc_j[i])))))
            * (doc_i[i] - doc_j[i])
            for i in xrange(beta_c.shape[0])])

    return grad


# queries are the queries of a given group k
# beta_k, mu_k, sigma_k are the parameters of the group k
# users are the users of having queries in group k
def eval_loglike_theta(beta_k, theta_k, queries, component, users, user_q, click_list, docs, prior_params, q_doc_features, pool):
    alpha0 = prior_params[0]
    mu0 = prior_params[1]
    sigma0 = prior_params[2]
    T = prior_params[3]

    #beta_k = theta_k[2*T:]
    mu_k = theta_k[0:T]
    sigma_k = theta_k[T:2*T]

    # estimate each one independently since we have i.i.d. samples
    # likelihood according to the prior distributions
    # just adding a very small number to avoid the log of 0
    first_term = np.sum([np.log(scipy.stats.norm(0, alpha0).pdf(beta_k[i]) + 10 ** -200) for i in range(beta_k.shape[0])])
    first_term += np.sum([np.log(scipy.stats.norm(mu0, sigma0).pdf(mu_k[i]) + 10 ** -200) for i in range(mu_k.shape[0])])
    first_term += np.sum([np.log(scipy.stats.norm(mu0, sigma0).pdf(sigma_k[i])+ 10 ** -200) for i in range(sigma_k.shape[0])])

    # likelihoods given query and click list
    sec_term = 0
    inputs = []
    for query in component:
        for user in users:
            # if user has that query
            if query in user_q[user]:
                inputs.append((theta_k, queries[query], click_list[user+':'+query], docs, T, query))
                #sec_term += joint_q_d(theta_k, queries[query], click_list[user+':'+query], docs, T, q_doc_features, query)

    #pool = mpc.Pool(4)
    for data in pool.map(joint_q_d, inputs):
        #print data
        sec_term += data
    return first_term + sec_term

def eval_grad_loglike_theta(beta_k, theta_k, queries, component, users, user_q, click_list, docs, prior_params, q_doc_features, pool):
    #T = prior_params[3]
    #beta_k = theta_k[2*T:]
    grad = np.zeros(beta_k.shape[0])
    alpha0_squared = prior_params[0] ** 2
    grad = -beta_k/alpha0_squared
    #for i in range(beta_k.shape[0]):
    #    grad[i] = -beta_k[i]/alpha0_squared
    inputs = []
    for query in component:
        for user in users:
            # if user has that query
            if query in user_q[user]:
            #try:
                user_specific_click_list = click_list[user+':'+query]

                # get the corresponding clicks and non clicks
                ind_clicked = [i for i, v in enumerate(user_specific_click_list) if v[1] == 1]
                ind_not_clicked = [i for i, v in enumerate(user_specific_click_list) if v[1] == 0]

                # create all the combinations
                comb = list(itertools.product(*[ind_clicked, ind_not_clicked]))
                #pair_prefs = []
                # for each element find the pairwise ranking preferences

                for elem in comb:
                    # args = [docs[user_specific_click_list[elem[0]][0]], docs[user_specific_click_list[elem[1]][0]]]
                    # print 'check: ' + str(check_grad(pair_pref, grad_pair_pref, beta_k, *args))

                    #pair_prefs.append(grad_pair_pref(beta_k,
                    #    q_doc_features[query+':'+user_specific_click_list[elem[0]][0].split(':')[1]],
                    #    q_doc_features[query+':'+user_specific_click_list[elem[1]][0].split(':')[1]]))
                    inputs.append((beta_k,
                       q_doc_features[query+':'+user_specific_click_list[elem[0]][0].split(':')[1]],
                        q_doc_features[query+':'+user_specific_click_list[elem[1]][0].split(':')[1]]))
                # if there was such a combination
                #if len(pair_prefs) >= 1:
                #    grad += np.sum(pair_prefs, axis=0)
            #except:
            #    pass
    #pool = mpc.Pool(4)
    for data in pool.map(grad_pair_pref, inputs):
        if len(data) >= 1:
            grad += np.sum(data, axis=0)
    return grad

def HMC(iterations, theta_k, queries, component, users, user_q, user_clicks, docs, prior_params, q_doc_features, pool):
    # define stepsize of MCMC.
    stepsize = 0.000047
    accepted = 0.0
    #beta_k = theta_k[2*T:]
    chain = [theta_k]
    T = prior_params[3]
    V = prior_params[4]
    #beta_k = theta_k[2*T:]
    #args = [theta_k, queries, component, users, user_q, user_clicks, docs, prior_params, q_doc_features]

    # TODO : check the gradients again, the check_grad measurements seem to be off
    # print 'grad correction: ' + str(check_grad(eval_loglike_theta,
    #                                       eval_grad_loglike_theta,
    #                                       beta_k, *args))

    for i in range(iterations):
        print 'iteration HMC %i' % i
        old_theta_k = chain[len(chain) - 1]
        old_energy = -eval_loglike_theta(old_theta_k[2*T:], old_theta_k, queries,
                        component, users, user_q, user_clicks, docs, prior_params, q_doc_features, pool)
        #print 'old_energy: ' + str(old_energy)
        old_grad = -eval_grad_loglike_theta(old_theta_k[2*T:], old_theta_k, queries,
                                            component, users, user_q, user_clicks,
                                            docs, prior_params, q_doc_features, pool)
        #print 'old grad: ' + str(old_grad)

        new_theta_k = np.copy(old_theta_k)  # deep copy of array
        new_grad = np.copy(old_grad)   # deep copy of array

        # Suggest new candidate using gradient + Hamiltonian dynamics.
        # draw random momentum vector from unit Gaussian.
        p = np.random.normal(0, 1, V)
        H = np.dot(p, p)/2.0 + old_energy    # compute Hamiltonian

        new_beta_k = new_theta_k[2*T:]
        # Do 5 Leapfrog steps.
        for tau in range(5):
            #print 'leap %i' % tau
            # make half step in p
            p = p - stepsize*new_grad/2.0
            # make full step in alpha,
            # but only the beta parameters, mu_k, simga_k
            # are fixed from previously
            new_beta_k = new_beta_k + stepsize*p
            new_theta_k[2*T:] = new_beta_k
            # compute new gradient
            new_grad = -eval_grad_loglike_theta(new_beta_k, new_theta_k, queries,
                                                component, users, user_q, user_clicks,
                                                docs, prior_params, q_doc_features, pool)

            # make half step in p
            p = p - stepsize*new_grad/2.0

        # Compute new Hamiltonian. Remember, energy = -loglik.
        new_energy = -eval_loglike_theta(new_theta_k[2*T:], new_theta_k, queries,
                                         component, users, user_q, user_clicks,
                                         docs, prior_params, q_doc_features, pool)
        #print 'new energy: ' + str(new_energy)
        newH = np.dot(p, p)/2.0 + new_energy
        dH = newH - H

        # Accept new candidate in Monte-Carlo fashion.
        if (dH < 0.0):
            chain.append(new_theta_k)
            accepted = accepted + 1.0
        else:
            u = random.uniform(0.0, 1.0)
            if (u < math.exp(-dH)):
                chain.append(new_theta_k)
                accepted = accepted + 1.0
            else:
                chain.append(old_theta_k)

    return chain


# caching, for faster computation
def memoize(func):
    S = {}

    def wrappingfunction(*args):
        if args not in S:
            S[args] = func(*args)
        return S[args]
    return wrappingfunction


@memoize
def stirling1(n, k):
    """Returns the stirling number of the first kind using recursion.."""
    if n == 0 and k == 0:
        return 1
    if k == 0 and n >= 1:
        return 0
    if k > n:
        return 0
    return stirling1(n-1, k-1) + (n - 1) * stirling1(n-1, k)
