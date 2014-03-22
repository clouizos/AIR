# -*- coding: utf-8 -*-
"""
@author: Chuan
"""
import MySQLdb.cursors
from elasticsearch import Elasticsearch
import json
import sys, logging
from gensim import corpora, models, similarities
#import corpera.Dictionary as dictionary

es_index = 'expressie'
#es_index = 'selectie'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Documents are 'created' on the fly and streamed as an iterator
# from the queries provided and the hits returned by elasticsearch
# 1 hit = 1 document
# next_doc provides what's currently extracted
class Docs:
    def __init__(self, cursor, sql_query):
        self.cur = cursor
        self.sql_q = sql_query
        self.cur.execute(self.sql_q)
        self.results = self.next_q(self.cur)
        self.result = ''
        
    def __iter__(self):
        return self
        
    def next(self):
        try:
            # return hit as doc
            # lowercase
            self.result = self.next_doc(self.results)
        except:
            # get new query
            self.results = self.next_q(cur)
            return next(self)
            #self.result = self.next_doc(self.results)
        return self.result
            
    # streaming queries: 1 at a time
    def next_q(self, cur):
        try:
            row = cur.fetchone()
            # extract the jsonQuery
            prefix = """jsonQuery":"""
            shcQ = row['txt_SearchHistoryCookedQuery']
            jsonQ = shcQ.split(prefix).pop()
            #print jsonQ[1:-2]
            jsonQ = jsonQ[1:-2]
            
            # Search and return json as python dictionary:
            
            # search_selectie, which contains items (for example, descriptions of stories within a news broadcast)
            #print es.search(index="search_selectie", body=jsonQ)
            # search_expressie, which contains full broadcasts
            #print es.search(index="search_expressie", body=jsonQ)
        
            result = es.search(index="search_"+es_index, body=jsonQ) # returns a dictionary
            results = result['hits']['hits'] # returns a list of hits(dict)
        except:
            raise StopIteration
        return iter(results)
        
    # streaming doc from electricsearch: 1 hit a time
    def next_doc(self, results):       
        result = next(results)
        
        # result['sort', '_type', '_source', '_score', '_index', '_id']
        result = result['_source'] # dictionary again
        # result['reeks', 'posities', 'expressie', 'aanmaakdatum', 'beschikbaar', 'realisatie', 'werk', 'laatst_aangepastdatum']
        # result['reeks', 'posities', 'selectie', 'aanmaakdatum', 'beschikbaar', 'realisatie', 'werk', 'laatst_aangepastdatum']           
        # print_keys(result['realisatie'])
        # [producent, titel, distributiekanaal, niveau, aanmaakdatum, overige_titels, parent_id, ext, recht, 
        #       verwijderd, id, laatst_aangepastdatum]
        try:
            t1 = result['realisatie']['titel']['tekst'] + '\n'
        except:
            t1 = ''
            
        result = result[es_index] # dict
        # result['recht', 'titel', 'publicaties', 'niveau', 'aanmaakdatum', 'kijkwijzer_classificaties', 'parent_id',
        #    'ext', 'taalgebruiken', 'verwijderd', 'context', 'id', 'laatst_aangepastdatum', 'tijdsduur']
        try:
            t2 = result['titel']['tekst'] + '\n'
        except:
            t2 = ''
            
        result = result['niveau']
        # result['geografische_namen', 'trefwoorden', 'namen', 'taalID', 'intentieID', 'samenvatting', 'taakID', 'beschrijving']
        #print_keys(result)
        s = result.get('samenvatting', '') + '\n'
        b = result.get('beschrijving', '')
    
#        keywords = result.get('trefwoorden') # returns a list of dictionaries again
#        try:
#            for keys in keywords:
#                for key, value in keys.iteritems() :
#                    print key, value
#        except:
#            pass
        return ''.join([t1, t2, s, b])
        
def print_keys(result):
    for key, value in result.iteritems() :
            print key            

# corpus iterator - for corpus streaming - 1 doc at a time
# converts the corpus to a vector space   
class Corpus(object):
    def __init__(self, _dict, docs):
        self._dict = _dict
        self.docs = docs
        
    def __iter__(self):
        for doc in self.docs:
            yield dictionary.doc2bow(doc.lower().split())

# ElasticSearch
# Connect to localhost:
#es = Elasticsearch(host='localhost', port=9876, url_prefix='/search_expressie,search_selectie/_search?pretty=1')
es = Elasticsearch(host='localhost', port=9876)

# MySQL
# connect database: (host, user, password, database)
try:
    # SSDictCursor for memory-efficient, streaming query: rows as dictionaries
    print "a"
    con = MySQLdb.connect('localhost', 'group2', 'group2air2k14', 'air2k14', cursorclass = MySQLdb.cursors.SSDictCursor);   
    print "b"
    # queries from ResultClickOverview    
    rco = "SELECT txt_SearchHistoryCookedQuery FROM \
        SearchHistory s JOIN ResultClickOverview r on r.int_SearchHistoryID = s.int_SearchHistoryID LIMIT 100"
    # queries from OrderOverview    
    oo = "SELECT txt_SearchHistoryCookedQuery FROM \
        SearchHistory s JOIN OrderOverview o on o.int_SearchHistoryID = s.int_SearchHistoryID LIMIT 100"
    cur = con.cursor()
    
    # 1st pass: create dictionary (corpus index)
    # docs to lowercase
    # doc iterator
    docs = Docs(cur, rco)
    dictionary = corpora.Dictionary(doc.lower().split() for doc in docs)
    # remove stop words and words that appear only once
    #stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
    #         if stopword in dictionary.token2id]
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
    dictionary.filter_tokens(once_ids) # remove stop words and words that appear only once
    dictionary.compactify() # remove gaps in id sequence after words that were removed    
    print len(dictionary)
    cur.close()
    
    # 2nd pass: vectorizing
    # reset iterators
    cur = con.cursor()
    docs = Docs(cur, rco)
    corpus = Corpus(dictionary, docs)
    # save corpus in Market Matrix format
    corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus)
    # load corpus iterator from MM file
    mm = corpora.MmCorpus('/tmp/corpus.mm')  
    
    # distributed
    # online lda
    #lda = models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=10, update_every=1, chunksize=10000, passes=1, distributed=True)
    # batch lda
    #lda = models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=10, update_every=0, passes=20, distributed=True)
    
    # non-distributed
    # online lda
    #lda = models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=10, update_every=1, chunksize=10000, passes=1)
    # batch lda
    lda = models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=10, update_every=0, passes=20)    
    lda.show_topics(10, formatted=False)    
    
    cur.close()
        
except MySQLdb.Error, e:  
    if con:
        con.rollback()
        
    print "Error %d: %s" % (e.args[0],e.args[1])
    sys.exit(1)
    
finally:    
    if con:        
        con.close()
#        
#con2 = MySQLdb.connect('localhost', 'group2', 'group2air2k14', 'air2k14');
#cur = con2.cursor()
#cur.execute(rco)
#rows = cur.fetchall()
#print len(rows)
#con2.close()
