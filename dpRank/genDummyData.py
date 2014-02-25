from __future__ import division
import numpy as np


# dummy generation of documents and their features
def gen_documents(nr_docs, V):
    doc_dic = {}
    for i in range(nr_docs):
        doc_dic['doc_%i' %i] = np.random.normal(0, 1., V)
    
    return doc_dic
    
# dummy generation of documents and their features
def gen_queries(nr_q, T):
    q_dic = {}
    for i in range(nr_q):
        q_dic['q_%i' %i] = np.random.random_integers(2, 5, T) 
    return q_dic

# dummy generation of query features
def gen_user_query(N, queries):
    user_q = {}
    for i in range(N):
        size = np.random.random_integers(1, len(queries), 1)
        q_set = []
        print len(queries)
        for j in range(size):
            q_set.append('q_%i' %np.random.random_integers(1, len(queries)-1,1)[0])
        user_q['user_%i' %i] = set(q_set)
    return user_q

# dummy assignment of queries to list of retrieved documents
def ass_query_doc(queries, docs):
    q_doc = {}
    for q in queries:
        # size of the random returned list, at least 10 docs
        size_rel = np.random.random_integers(10, 15, 1)[0]
        # get randomly the documents
        docs_assign = np.random.choice(docs.keys(), size_rel)
        # remove duplicates
        q_doc[q] = set(docs_assign)
    return q_doc

# dummy assignmet of clicks to docs per query and per user    
def gen_clicks_per_user_query(user_q, q_doc):
    user_clicks = {}
    # probability of clicking, 0.9 for not and 0.1 for click
    p = [0.90, 0.10]
    # foreach user
    for i in user_q:
        # foreach query of the user
        for j in user_q[i]:
            clicks = np.random.choice([0,1], len(q_doc[j]), p=p)
            user_clicks[i+':'+j] = zip(q_doc[j], clicks)
            
    return user_clicks
             
    
def gen_dummy(nr_docs, nr_queries, N):    
    
    # nr of query features
    T = 5
    # nr of document features
    V = 65

    # generate documents with random features
    docs = gen_documents(nr_docs, V)
    # generate queries with random features
    queries = gen_queries(nr_queries, T)
    # generate random assignment of users to queries
    user_q = gen_user_query(N, queries)  
    # assign queries to list of documents
    q_doc = ass_query_doc(queries, docs)
    # randomly generate assignments of clicks per query
    user_clicks = gen_clicks_per_user_query(user_q, q_doc)
    
    return docs, queries, user_q, q_doc, user_clicks
