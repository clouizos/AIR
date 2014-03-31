# author: Remi
# few edits by Agnes

# -*- coding: utf-8 -*-

print "Importing module tfidf_bm25_comparison"
import redis
import re
import string
from math import log
from nltk.stem.snowball import SnowballStemmer
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
  index_bi = redis.StrictRedis(host='localhost', port=6379, db=1)
  df_mono = redis.StrictRedis(host='localhost', port=6379, db=2)
  df_bi = redis.StrictRedis(host='localhost', port=6379, db=3)
  return (index_mono, index_bi, df_mono, df_bi)


DB_CONNS = get_db_connections()

#Some constants used by BM25
NUMBER_OF_DOCUMENTS = len(DB_CONNS[0].keys())
# This one was determined in a separate calculation.
AVERAGE_DOCUMENT_LENGTH = dict({
  'title':3.6178696042268967,
  'description' : 82.0403598843585,
  'summary':45.95869305154022,
  'total':131.6169225401256})

## in BM25 we should sum over all terms in the query, however tf(t, d) for a term that is not in the document
## will be zero. The increment in this case is zero, so we can skip over the terms of the query that are not in the document
def bm25_comparison(query, documentID, k1=1.2, b=0.75):
  query = map(lambda x : stem(x), removePunct(query.lower()).split())

  categories = ('title', 'description', 'summary', 'total')
  results = []

  # First get the monograms:

  connsResult = DB_CONNS[2].get(documentID)
  if (connsResult != None):
    descriptor = eval(connsResult)
  else:
    descriptor = {}
  #descriptor = eval(DB_CONNS[2].get(documentID))
  for cat in categories:
    score = 0
    try:
      documentLength = sum(descriptor[cat].values())
      #dKeys =  descriptor[cat].keys()
      #for dkey in dKeys:
      #  if not type(dkey) is unicode:
      #    print "not unicode, is %s: %s\n" % (type(dkey), dkey)

      for q in query:
        if q in descriptor[cat].keys():
          tf = descriptor[cat][q]
          idf = NUMBER_OF_DOCUMENTS / len(eval(DB_CONNS[0].get(q)))
          score += idf * ( ( (k1 + 1) * tf ) / ( k1 * ( (1 - b) + b * (documentLength/AVERAGE_DOCUMENT_LENGTH[cat]) ) + tf ) )
    except:
      pass
    results.append(score)

  # Now do the bigrams:
  push = lambda x, y : (x[1], y)
  #descriptor = eval(DB_CONNS[3].get(documentID))

  connsResult = DB_CONNS[3].get(documentID)
  if (connsResult != None):
    descriptor = eval(connsResult)
  else:
    descriptor = {}

  for cat in categories:
    score=0
    try:
      documentLength = sum(descriptor[cat].values())
      bigram = ("<s>","<s>")
      for q in query:
        bigram = push(bigram, q)
        if bigram in descriptor[cat].keys():
          tf = descriptor[cat][bigram]
          idf = NUMBER_OF_DOCUMENTS / len(eval(DB_CONNS[1].get(bigram)))
          score += idf * ( ( (k1 + 1) * tf ) / ( k1 * ( (1 - b) + b * (documentLength/AVERAGE_DOCUMENT_LENGTH[cat]) ) + tf ) )
    except:
      pass
    results.append(score)
  return results

def tfidf_comparison(query, documentID):
  query = map(lambda x : stem(x), removePunct(query.lower()).split())

  categories = (u'title', u'description', u'summary', u'total')
  results = []

  # First get the monograms:
  connsResult = DB_CONNS[2].get(documentID)
  if (connsResult != None):
    descriptor = eval(connsResult)
  else:
    descriptor = {}
  for cat in categories:
    score = 0
    for q in query:
      try:
        if q in descriptor[cat].keys():
          tf = descriptor[cat][q]
          idf = NUMBER_OF_DOCUMENTS / len(eval(DB_CONNS[0].get(q)))
          score += (1 + log(tf)) * idf
      except:
        pass
    results.append(score)

  # Now do the bigrams:
  push = lambda x, y : (x[1], y)
  #descriptor = eval(DB_CONNS[3].get(documentID))
  connsResult = DB_CONNS[3].get(documentID)
  if (connsResult != None):
    descriptor = eval(connsResult)
  else:
    descriptor = {}
  for cat in categories:
    score = 0
    bigram = ("<s>","<s>")
    for q in query:
      bigram = push(bigram, q)
      try:
        if bigram in descriptor[cat].keys():
          tf = descriptor[cat][bigram]
          idf = NUMBER_OF_DOCUMENTS / len(eval(DB_CONNS[1].get(bigram)))
          score += (1 + log(tf)) * idf
      except:
        pass
    results.append(score)
  return results

def tfidf_bm25_score(query, document):
  tfidf = tfidf_comparison(query, document)
  bm25 = bm25_comparison(query, document)
  return tuple(tfidf + bm25)

if __name__ == "__main__":
  print "Imports done."
  print tfidf_comparison('president johnson', "287148")
  print bm25_comparison('president johnson', "287148")
  print tfidf_bm25_score(u'belgië', "287148")
  print tfidf_bm25_score(u'president johnson', "287148")



