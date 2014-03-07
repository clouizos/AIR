# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from genDummyData import gen_dummy
from natsort import natsorted
import utils as u
import pymc
from copy import deepcopy
import multiprocessing as mpc
import cPickle as pickle

# we will take log(0) = -Inf so turn off this warning
# np.seterr(divide='ignore')

class user_q_assign:
    # matrix for query probabilities for each group
    #q_assign = None
    # name of the user
    user_name = ''
    # holding all the queries of the user
    queries = []
    # holding the clicks for each query of the user
    click_list = {}
    # vector of assignments of the queries
    assign = None

    def __init__(self, user, queries, user_clicks, K):
        #self.q_assign = np.zeros((len(queries), K))
        self.user_name = user
        self.queries = list(queries)
        self.assign = np.zeros(len(queries), dtype=int)
        for i in xrange(len(natsorted(self.queries))):
            self.click_list[self.queries[i]] = user_clicks[user+':'+self.queries[i]]
            # random assignment of the user queries to groups
            self.assign[i] = int(np.random.choice(K))

    # indices should be sorted in a descending order
    # shift the assignments to reflect the new parameters
    def adjust_deletion(self, indices):
        for ind in indices:
            for i in xrange(self.assign.shape[0]):
                if ind < self.assign[i]:
                    self.assign[i] += -1




def sample_c(users_objects, queries, alpha, eta, theta_c, theta_e, gamma_k, gamma_e, docs, K, prior_params):
    T = prior_params[3]
    # for each user
    for user_check in natsorted(user_q):
        user = users_objects[user_check]
        # for each query of the user sample the assingments
        for i in xrange(len(natsorted(user.queries))):
            # for each group
            assign_group = np.zeros(K)
            for k in xrange(K):
                mu_uk_minus_current = 0
                for j in xrange(len(natsorted(user.queries))):
                    if j != i:
                        if user.assign[j] == k:
                            mu_uk_minus_current += 1
                assign_group[k] = np.log((eta*gamma_k[k] + mu_uk_minus_current))
                assign_group[k] += u.joint_q_d(theta_c[k,:],
                                           queries[user.queries[i]],
                                           user.click_list[user.queries[i]],
                                           docs, T)
            aux_assign = np.log(eta*gamma_e)
            aux_assign += u.joint_q_d(theta_e, queries[user.queries[i]],
                                      user.click_list[user.queries[i]],
                                      docs, T)
            # get all the probabilities
            sample_prob = np.hstack((assign_group, aux_assign))
            # normalize
            sample_prob /= sample_prob.sum()
            user.assign[i] = np.random.multinomial(1, sample_prob).argmax()

        # assign the new object back
        users_objects[user_check] = user

    #for user in natsorted(users_objects):
    #    print users_objects[user].assign

    # get the observations
    occ_k = np.zeros(K, dtype=int)
    occ_aux = 0
    for user_check in natsorted(user_q):
        for elem in users_objects[user_check].assign:
            #auxiliary
            if elem == K:
                occ_aux += 1
            else:
                occ_k[elem] += 1

    # find possible empty components and
    # remove accordingly
    ind = []
    for i in xrange(occ_k.shape[0]):
        if occ_k[i] == 0:
            ind.append(i)

    # sort in descending order in place
    ind.sort(reverse=True)

    # adjust the indices of the assignments for the user queries
    if len(ind) >= 1:
        print 'Found empty components: ' + str(ind)
        for user_n in users_objects:
            users_objects[user_n].adjust_deletion(ind)
        gamma_k = np.delete(gamma_k, ind)
        theta_c = np.delete(theta_c, ind, axis=0)
        gamma_e = 1 - np.sum(gamma_k)
        K -= len(ind)

    # if we have assignments to the auxiliary component
    if occ_aux > 0:
        print 'Found %i assignments to the auxiliary component.' % occ_aux
        theta_c = np.vstack((theta_c, theta_e))
        theta_e = np.hstack((mu_kt.random(), tau_kt.random(), beta_ku.random()))
        # new gamma for the new component and new auxiliary gamma_e
        b_n = np.random.beta(1.0, alpha)
        gamma_k = np.append(gamma_k, b_n*gamma_e)
        gamma_e = (1 - b_n) * gamma_e
        K += 1

    return users_objects, gamma_k, gamma_e, theta_c, theta_e


def sample_g(users_objects, gamma_k, gamma_e, alpha, eta, user_q, out_q = None):
    #name = mpc.current_process().name
    #print name, 'Starting'

    # get the active components
    K = gamma_k.shape[0]
    # init the random variable h
    m_uik = np.zeros((len(user_q), K), dtype = int)

    # find the m_uik for every user
    cnt = 0
    for user_check in natsorted(users_objects):
        m_uik[cnt,:] = np.bincount(users_objects[user_check].assign, minlength=K)
        cnt += 1


    # it seems that we need to sample h first
    # and based on that we sample gamma
    h_uik = np.zeros((len(user_q), K))
    for i in xrange(len(user_q)):
        for j in xrange(K):
            # do not keep the first value (0) since
            # the stirling1 will produce 0 so it doesn't
            # affect the probabilities
            h_candidates = np.asfarray([u.stirling1(m_uik[i,j], h) *
                            (eta * gamma_k[j])** h for h in range(m_uik[i,j])[1:]])
            # normalize and pick the maximum
            # again maybe we need to pick a random one according to those probabilities?

            if h_candidates.shape[0] > 0 and not np.array_equal(h_candidates, np.zeros(h_candidates.shape[0])):
                h_candidates /= h_candidates.sum()
                # max one
                #h_uik[i,j] = h_candidates.argmax() + 1
                # random one
                #h_uik[i,j] = np.random.choice(range(m_uik[i,j])[1:], 1, p=h_candidates)[0]
                # again this is faster
                h_uik[i,j] = np.random.multinomial(1, h_candidates).argmax() + 1

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

    if out_q is not None:
        key = ['gamma_k', 'gamma_e']
        value = [gamma_k, gamma_e]
        out_q.put(dict(zip(key,value)))
        #print name, 'Exiting'
    else:
        return gamma_k, gamma_e


def sample_theta(users_objects, theta_c, queries, user_q, docs, user_clicks, prior_params, iter_HMC, out_q = None):

    #name = mpc.current_process().name
    #print name, 'Starting'

    mu0 = prior_params[1]
    sigma0 = prior_params[2]
    alpha0 = prior_params[0]
    beta0 = prior_params[5]
    K = theta_c.shape[0]
    T = prior_params[3]

    # find the total number of queries
    # assosiated to group k
    nk_tot = np.zeros(K, dtype=int)
    for user_check in users_objects:
        nk_tot += np.bincount(users_objects[user_check].assign, minlength=K)

    # I think this should be K (the number of active_components, but K != len(active_components))
    for k in xrange(K):

        mu_k = theta_c[k, 0:T]
        sigma_k = theta_c[k, T: 2*T]
        nk = nk_tot[k]

        queries_of_k = []
        for user_check in users_objects:
            q_ind = np.where(users_objects[user_check].assign == k)[0]
            for elem in q_ind:
                queries_of_k.append(users_objects[user_check].queries[elem])

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
            mu_n_kt = sigma_n_kt * ((mu0 / sigma0) + (sum_of_user_queries * lambda_kt))

            # UPDATE mu here to use to compute sigma_kt
            current_mu_kt = np.random.normal(mu_n_kt, sigma_n_kt)

            sum_of_user_queries_minus_mu = 0
            for q in queries_of_k:
                sum_of_user_queries_minus_mu += (queries[q][t] - current_mu_kt)**2

            alpha_n_kt = alpha0 + nk/2
            beta_nk_t = beta0 + 1/2 * sum_of_user_queries_minus_mu

            # what we draw here is the precision
            # so we need to get 1/x for the variance
            current_sigma_kt = 1./np.random.gamma(alpha_n_kt, beta_nk_t, 1)

            # write result to subvector
            mu_k[t] = current_mu_kt
            sigma_k[t] = current_sigma_kt

        # write vector update to theta
        theta_c[k, 0:T] = mu_k
        theta_c[k, T: 2*T] = sigma_k

        # begin the update for the weights beta_c
        users = []
        queries_check = set(queries_of_k)
        for user in natsorted(user_q):
            if len(set(user_q[user]).intersection(queries_check)) >= 1:
                users.append(user)

        chain = u.HMC(iter_HMC, theta_c[k,:], queries, queries_of_k, users, user_q, user_clicks, docs, prior_params)
        # Discard first 20% of MCMC chain
        clean = []
        for n in range(int(0.2 * iter_HMC),len(chain)):
            # thin the samples
            if (n % 2 == 0):
                clean.append(chain[n])

        theta_c[k,:] = np.mean(clean, axis = 0)

    if out_q is not None:
        out_q.put(dict(theta_c = theta_c))
        #print name, 'Exiting'
    else:
        return theta_c


if __name__ == '__main__':
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
    K = 2

    # nr of query features
    T = 5
    # nr of document features
    V = 65
    # nr of users
    N = 20

    prior_params = [alpha0, mu0, sigma0, T, V, beta0]
    # args for the dummy data generator
    nr_docs = 500
    nr_queries = 50

    # dummy groups
    groups = 5

    # active components (components that have observations)
    active_components = [[] for i in range(K)]

    # get some dummy data
    docs, queries, user_q, q_doc, user_clicks = gen_dummy(nr_docs, nr_queries, N, groups)

    # gibbs sampler params
    gibbs_iter = 1000
    thinning = 5
    burnin = int(0.2 * gibbs_iter)
    iter_HMC = 50

    ''' priors of the G0 '''
    # prior means
    mu_kt = u.random_normal(mu0, sigma0, T)
    # prior precision
    tau_kt = u.random_gamma(alpha0, beta0, T)
    # prior beta_ku
    beta_ku = u.random_normal(0, alpha0, V)

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

    users_objects = {}

    # delete users that have less than one query
    for user in natsorted(user_q):
        if len(user_q[user]) <= 1:
            del user_q[user]

    for user in natsorted(user_q):
        users_objects[user] = user_q_assign(user, user_q[user], user_clicks, K)


    tot_chain = []
    out_q = mpc.Queue()
    resultdict = {}

    for i in range(gibbs_iter):
        print 'Doing iteration %i ... ' % (i + 1)

        params = {}
        users_objects, gamma_k, gamma_e, theta_c, theta_e = sample_c(users_objects,
                                                                queries, alpha,
                                                                eta, theta_c,
                                                                theta_e, gamma_k,
                                                                gamma_e, docs, K,
                                                                prior_params)



        params['users_objects'] = deepcopy(users_objects)
        params['theta_e'] = np.copy(theta_e)
        print 'Sampled %i components.' % theta_c.shape[0]

        # since K changes locally inside the sample_c, get the new one
        K = theta_c.shape[0]

        # experimental
        args_g = (users_objects, gamma_k, gamma_e, alpha, eta, user_q, out_q)
        args_t = (users_objects, theta_c, queries, user_q, docs, user_clicks, prior_params, iter_HMC, out_q)

        # define the parallel processes
        d = mpc.Process(name='gamma_sampling', target=sample_g, args=args_g)
        n = mpc.Process(name='theta_sampling', target=sample_theta, args=args_t)

        # start them
        d.start()
        n.start()

        # wait for them to finish
        d.join()
        n.join()

        # parse their results
        for i in range(2):
            resultdict.update(out_q.get())

        gamma_k = resultdict['gamma_k']
        gamma_e = resultdict['gamma_e']
        theta_c = resultdict['theta_c']
        resultdict.clear()


        #gamma_k, gamma_e = sample_g(users_objects, gamma_k, gamma_e, alpha, eta, user_q)
        #theta_c = sample_theta(users_objects, theta_c, queries, user_q, docs, user_clicks, prior_params, iter_HMC)

        params['gamma_k'] = np.copy(gamma_k)
        params['gamma_e'] = np.copy(gamma_e)
        params['theta_c'] = np.copy(theta_c)

        for user in natsorted(users_objects):
            print users_objects[user].assign
        print gamma_k, gamma_e
        print theta_c.shape
        print theta_c[:,T:2*T]

        tot_chain.append(params.copy())

    print
    print 'total chain:'
    for elem in tot_chain:
        users_objects = elem['users_objects']
        for user in natsorted(users_objects):
            print users_objects[user].assign
        print
    print

    # Discard first 20% of MCMC chain (burnin)
    clean = []
    for n in range(burnin,len(tot_chain)):
        # thin the samples
        if (n % thinning == 0):
            clean.append(tot_chain[n])

    print 'clean chain:'
    for elem in clean:
        users_objects = elem['users_objects']
        for user in natsorted(users_objects):
            print users_objects[user].assign
        print

    with open('clean_total.pickle', 'wb') as handle:
        pickle.dump(clean, handle)


