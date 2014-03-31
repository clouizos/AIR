# author: Nicolo
# few edits by Agnes

# -*- coding: utf-8 -*-
from __future__ import division
import redis

from math import log
import string
import re
from nltk.stem.snowball import SnowballStemmer
from math import log

stemmer = SnowballStemmer("dutch")

def removePunct(txt):
  s = string.punctuation
  s2 = re.escape(s)
  s2 = '['+s2+']'
  txt = re.sub(s2, " ", txt)
  return txt

def stem(x):
        try:
                return stemmer.stem(x)
        except:
                return x

def get_db_connections():
        index_mono = redis.StrictRedis(host='localhost', port=6379, db=0)
        index_bi = redis.StrictRedis(host='localhost', port=6379, db=5)
        df_mono = redis.StrictRedis(host='localhost', port=6379, db=2)
        df_bi = redis.StrictRedis(host='localhost', port=6379, db=3)
        db_coll = redis.StrictRedis(host='localhost', port=6379, db=5)
        return (index_mono, index_bi, df_mono, df_bi, db_coll)


DB_CONNS = get_db_connections()


NUMBER_OF_DOCUMENTS = len(DB_CONNS[0].keys())



AVERAGE_DOCUMENT_LENGTH = dict({
        'title':3.6178696042268967,
        'description' : 82.0403598843585,
        'summary':45.95869305154022,
        'total':131.6169225401256})


COLLECTION_LENGTH = dict({
    'title':AVERAGE_DOCUMENT_LENGTH['title'] * NUMBER_OF_DOCUMENTS ,
        'description' : AVERAGE_DOCUMENT_LENGTH['description'] * NUMBER_OF_DOCUMENTS ,
        'summary':AVERAGE_DOCUMENT_LENGTH['summary'] * NUMBER_OF_DOCUMENTS ,
        'total':AVERAGE_DOCUMENT_LENGTH['total'] * NUMBER_OF_DOCUMENTS })

#print COLLECTION_LENGTH['total']



def linearInterpolation(query, documentID, lambdaVal):

  query = map(lambda x : stem(x), removePunct(query.lower()).split())

  connsResult = DB_CONNS[2].get(documentID)
  if connsResult != None:
    descriptor = eval(connsResult)
  else:
    descriptor = {}


  score = 0

  try:
    docLength = sum(descriptor['total'].values())
    for term in query:
      if term in descriptor['total'].keys():
        termCount= descriptor['total'][term]
      else:
        termCount = 0

      cf = DB_CONNS[4].get(term)
      if cf is not None and docLength > 0:
        score += log((termCount/docLength)*lambdaVal + (int(cf)/(COLLECTION_LENGTH['total']))*(1-lambdaVal))

      else:
        score = float("-inf")
  except:
    pass
  return score


def dirichletSmoothing(query, documentID, alpha):

  query = map(lambda x : stem(x), removePunct(query.lower()).split())

  #descriptor = eval(DB_CONNS[2].get(documentID))
  connsResult = DB_CONNS[2].get(documentID)
  if connsResult != None:
    descriptor = eval(connsResult)
  else:
    descriptor = {}


  score = 0

  #print descriptor['total'].keys()

  try:
    docLength = sum(descriptor['total'].values())
    for term in query:
      if term in descriptor['total'].keys():
        termCount= descriptor['total'][term]
      else:
        termCount = 0
      cf = DB_CONNS[4].get(term)
      #print "cf:%s\n" % cf
      if cf is not None:
        score += log ((termCount + alpha * ( int(cf)/COLLECTION_LENGTH['total'])) / (docLength + alpha))
      else:
        score = float("-inf")
  except:
    pass

  return score


def KLdivergence(query, documentID):

  query = map(lambda x : stem(x), removePunct(query.lower()).split())

  ptq = {} # probability of a term in the query

  for term in query:
    if term in ptq.keys():
      ptq[term] += 1
    else:
      ptq[term] = 1

  for k,v in ptq.iteritems() :
    ptq[k] = v / len(query)



  #descriptor = eval(DB_CONNS[2].get(documentID))
  connsResult = DB_CONNS[2].get(documentID)
  if connsResult != None:
    descriptor = eval(connsResult)
  else:
    descriptor = {}

  score=0
  try:
    docLength = sum(descriptor['total'].values())


    #calculating alpha
    sumD = 0
    sumC = 0

    for term in descriptor['total'].keys():
      sumD += descriptor['total'][term] / docLength
      cf = DB_CONNS[4].get(term)
      if (cf == None):
        cf = 0
      sumC += int(cf) / COLLECTION_LENGTH['total']

    alpha = (1 - sumD) / (1 - sumC)


    for term, probQ in ptq.iteritems():
      if term in descriptor['total'].keys():
        ptd = descriptor['total'][term] / docLength
        cf = DB_CONNS[4].get(term)
        if (cf == None):
          cf = 0
        ptc = int(cf) / COLLECTION_LENGTH['total']
        if (ptc > 0 and alpha > 0) :
          score += probQ *log( ptd / (alpha * ptc)) + log (alpha)
  except:
    pass

  return score



