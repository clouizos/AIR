from __future__ import division
import numpy as np
from natsort import natsorted


# dummy generation of documents and their features
def gen_documents(nr_docs, V, groups):
    doc_dic = {}
    if nr_docs < groups:
        print "Less documents than groups! Cannot continue"
        return
    else:
        split = int(nr_docs/groups)
    cnt = 1
    for i in range(nr_docs):
        if i == cnt * split:
            cnt += 1
        doc_dic['doc_%i' %i] = np.random.normal(cnt, 1., V)

    return doc_dic

# dummy generation of query features
def gen_queries(nr_q, T, groups):
    q_dic = {}
    if nr_q < groups:
        print "Less queries than groups! Cannot continue"
        return
    else:
        split = int(nr_q/groups)
    cnt = 1
    for i in range(nr_q):
        if i == cnt * split:
            cnt += 1
        q_dic['q_%i' %i] = np.random.normal(cnt, 1., T)
    return q_dic

# dummy assignment of queries to users
def gen_user_query(N, queries):
    user_q = {}
    for i in range(N):
        size = np.random.random_integers(1, len(queries), 1)
        q_set = []
        for j in range(size):
            q_set.append('q_%i' %np.random.random_integers(1, len(queries)-1,1)[0])
        user_q['user_%i' %i] = set(q_set)
    return user_q

# dummy assignment of queries to list of retrieved documents
def ass_query_doc(queries, docs, groups):
    q_doc = {}
    if len(queries) < groups:
        print "Less queries than groups! Cannot continue"
        return
    else:
        split = int(len(queries)/groups)
    cnt = 0
    cnt2 = 1
    for q in natsorted(queries):
        if cnt == cnt2 * split:
            cnt2 += 1
        # size of the random returned list, at least 10 docs
        size_rel = np.random.random_integers(10, 15, 1)[0]
        # get randomly the documents
        docs_assign = np.random.choice(natsorted(docs.keys())[(cnt2 - 1) * split:cnt2 * split], size_rel)
        # remove duplicates
        q_doc[q] = set(docs_assign)
        cnt += 1
    return q_doc

# dummy assignmet of clicks to docs per query and per user
def gen_clicks_per_user_query(user_q, q_doc):
    user_clicks = {}
    # probability of clicking, 0.8 for not and 0.2 for click
    p = [0.8, 0.2]
    # foreach user
    for i in user_q:
        # foreach query of the user
        for j in user_q[i]:
            clicks = np.random.choice([0,1], len(q_doc[j]), p=p)
            # ensure we have at least one click
            if 1 not in clicks:
                clicks[np.random.randint(0, clicks.shape[0],1)[0]] = 1
            user_clicks[i+':'+j] = zip(q_doc[j], clicks)

    return user_clicks


def gen_dummy(nr_docs, nr_queries, N, groups):

    # nr of query features
    T = 5
    # nr of document features
    V = 65

    # generate documents with random features
    docs = gen_documents(nr_docs, V, groups)
    # generate queries with random features
    queries = gen_queries(nr_queries, T, groups)
    # generate random assignment of users to queries
    user_q = gen_user_query(N, queries)
    # assign queries to list of documents
    q_doc = ass_query_doc(queries, docs, groups)
    # randomly generate assignments of clicks per query
    user_clicks = gen_clicks_per_user_query(user_q, q_doc)

    return docs, queries, user_q, q_doc, user_clicks
