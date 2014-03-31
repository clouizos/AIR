# author: Agnes
# 
# maps the tf-idf-bm25 features, languagemodels features, and es features to 
# a vector space
#

# -*- coding: utf-8 -*-

import itertools
import dill as pickle
import pprint
import sys
from collections import defaultdict
import gc
import string
import re
import math
import multiprocessing as mp

featureDictES = {}
globalFacetDictES = {}

globalFieldToIndexDict={}

vectorDict = {}

ubDict={}

lm = 0
tfidf_bm25 = 0

dirichletSmoothingAlpha = 0.01
liSmoothingLambda = 0.01

saveDir="vectorDicts3/"

def importStuff() :

  print "Importing stuff."

  global ubDict
  global lm
  global tfidf_bm25

  sys.path.insert(0, "/virdir/Scratch/")
  from userBehaviorDict import dictionary as ubDict

  editUbDict()

  import LanguageModels as lm
  import tfidf_bm25_features as tfidf_bm25


# remove sessionid from userbehaviordict
# and translate clicked/order/viewed into positives/negatives
def editUbDict():

  global ubDict

  ubDictNew = {}

  for userId in ubDict:
    ubDictNew[userId] = {}
    sessionIds  = ubDict[userId].keys()

    for sessionId in sessionIds:
      for shId in ubDict[userId][sessionId]:

        ubDictNew[userId][shId] = {}

        docDictsClicked = ubDict[userId][sessionId][shId]["click"]
        docDictsPurch = ubDict[userId][sessionId][shId]["order"]
        docsViewed = ubDict[userId][sessionId][shId]["view"]

        clicked=[]
        viewed=[]
        purch=[]

        for doc in ubDict[userId][sessionId][shId]["click"]:
          clicked.append(doc["id"])
        for doc in ubDict[userId][sessionId][shId]["order"]:
          purch.append(doc["id"])

        viewed = ubDict[userId][sessionId][shId]["view"]
        viewed = list(set(viewed) - set(clicked))
        clicked = list(set(clicked) - set(purch))

        positivesPurch = purch
        positivesClicked = clicked
        negatives = viewed

        ubDictNew[userId][shId]["positivesPurch"] = positivesPurch
        ubDictNew[userId][shId]["positivesClicked"] = positivesClicked
        ubDictNew[userId][shId]["negatives"] = viewed


  ubDict = ubDictNew

# init global dicts that hold possible facet/field values
# for the ES features part
def initGlobalIndexDicts():

  global globalFacetDictES
  global globalFieldToIndexDict

  for k in globalFacetDictES.keys():
    if globalFacetDictES[k]["terms"] == []:
      del globalFacetDictES[k]
    else:
      indexToTermDict = dict(enumerate(globalFacetDictES[k]["terms"]))
      termToIndexDict = dict (zip(indexToTermDict.values(),indexToTermDict.keys()))
      globalFacetDictES[k]["tti"] = termToIndexDict


  possibleFields = [u"samenvatting", u"titel", u"beschrijving", u"catchall"]
  indexToFieldDict = dict(enumerate(possibleFields))
  globalFieldToIndexDict = dict (zip(indexToFieldDict.values(),indexToFieldDict.keys()))

#process a bunch of users' features
# make vector features of them
def process(batchNr):

  global vectorDict
  global saveDir

  initGlobalIndexDicts()

  nrUsers = len(featureDictES)
  userIDs = sorted(featureDictES.keys())

  vectorDict = {}
  for usersProcessed in range(0, nrUsers):
    userID = userIDs[usersProcessed]
    vectorDict[userID] =  {}

    for shID in featureDictES[userID] :
      vectorDict[userID][shID] = {}

      featureDictQuery = featureDictES[userID][shID]

      # query-only features
      queryVector = []

      queryVector.append(len(featureDictQuery["query_string"]))
      queryVector.append(len(featureDictQuery["query_string"].split()))
      queryVector.append(1*containsPunct(featureDictQuery["query_string"]))
      queryVector.append(featureDictQuery["queryNorm"])
      queryVector.append(1*featureDictQuery["containsDate"])

      (queryFilters,keywordsInFilter) = makeFilterDictVector(featureDictQuery)
      print "keywords check: %s\n" % keywordsInFilter
      queryVector.extend(queryFilters)

      #facetStatsVector = makeFacetStatsVector(featureDictQuery)
      #queryVector.extend(facetStatsVector)

      vectorDict[userID][shID]["q"] = queryVector

      #query-document features
      vectorDict[userID][shID]["qd"] = {}

      positivesPurch = ubDict[userID][shID]["positivesPurch"]
      positivesClicked = ubDict[userID][shID]["positivesClicked"]
      negatives = ubDict[userID][shID]["negatives"]

      # convert defaultdicts to dicts. defaultdict does not work with multiprocessing...
      featureDictQuery = convertDD(featureDictQuery)

      queryString = featureDictQuery["query_string"]
      print "queryString:%s, type:%s\n" % (queryString, type(queryString))

     # print type(featureDictQuery)
     # for k,v in featureDictQuery.iteritems():
     #   print type(v)

      featureDictDoc= featureDictQuery["doc"]

      # new data structures used for multiprocessing
      all = []
      for docID in negatives:
        all.append(("negatives", docID, featureDictQuery, featureDictDoc))
      for docID in positivesPurch:
        all.append(("positivesPurch", docID, featureDictQuery, featureDictDoc))
      for docID in positivesClicked:
        all.append(("positivesClicked", docID, featureDictQuery, featureDictDoc))

      #for test in all:
      #  result = getQDscoreStar( test)

      # calculate query-document features for all returned docs parallel
      pool = mp.Pool()
      results = pool.map(getQDscoreStar, all)

      # process results gotten from multiprocessing
      qdNegatives={}
      qdPositivesPurch={}
      qdPositivesClicked={}
      for (qtype, docID, vector) in results:
        if qtype == "negatives":
          qdNegatives[docID]=vector
        elif qtype == "positivesPurch":
          qdPositivesPurch[docID]=vector
        else:
          qdPositivesClicked[docID]=vector


      #print "len all: %d, len result:%s\n" % (len(all), len(results))
      #print "positivesPurch:%d\npositivesClicked:%d\nnegatives:%d\n" % (len(positivesPurch), len(positivesClicked), len(negatives))
      #print "qdPositivesPurch:%d\nqdPositivesClicked:%d\nqdNegatives:%d\n" % (len(qdPositivesPurch), len(qdPositivesClicked), len(qdNegatives))

      vectorDict[userID][shID]["qd"]["negatives"] = qdNegatives
      vectorDict[userID][shID]["qd"]["positivesClicked"] = qdPositivesClicked
      vectorDict[userID][shID]["qd"]["positivesPurch"] = qdPositivesPurch

    print "processed %d users" % usersProcessed

    if (usersProcessed % 3) == 0:
      print "saving to %s" % saveDir
      saveRawDict(vectorDict, saveDir+"vectorDict"+str(batchNr)+".py")

  saveRawDict(vectorDict, saveDir+"vectorDict"+str(batchNr)+".py")



# save dictionary as .py file (does not work with defaultdict)
def saveRawDict(dic, fn):
  f = open(fn, "w")
  f.write( "dictionary = " + str( dic ))
  f.close()


# convert all defaultdicts in a (nested) (default)dict structure to dict's
def convertDD(d) :
  if type(d) is defaultdict:
    d = dict(d)

  if (type(d) is dict):

    for k,v in d.iteritems():
      if type(v) is defaultdict or type(v) is dict:
        d[k]=convertDD(v)

    return d
  else:
    return d


# calls getQDscore, helper function for multiprocessing
def getQDscoreStar(typeDocIDtuple) :
  return getQDscore(*typeDocIDtuple)

# calculate some join query-document features
# as well as document-only features
def getQDscore(qtype, docID, featureDictQuery, featureDictDoc) :


  vector = []

  # containsdate
  vector.append(1*featureDictDoc[docID]["containsDate"])
  # date matches with date in query
  vector.append(1*featureDictDoc[docID]["dateMatches"])

  # which of two possible indexes does this doc belong to
  possibleIndexes = [0]*2
  if (featureDictDoc[docID]["index"][7:].startswith("expressie")):
    possibleIndexes[0]=1
  else:
    possibleIndexes[1] = 1
  vector.extend(possibleIndexes)

  # weight of fields (summary/description/title/etc.) in that document (ES feature)
  #fieldNormsVector = [0]*len(globalFieldToIndexDict)
  #fieldNormsDict = featureDictDoc[docID]["fieldNormsDict"]
  #for field in fieldNormsDict:
  #  try:
  #    index = globalFieldToIndexDict[field]
  #    fieldNormsVector[index] = fieldNormsDict[field]
  #  except:
  #    pass
  ##print "fieldNormsVector:%s\n" % fieldNormsVector
  #vector.extend(fieldNormsVector)

  # how rare each value of each facet of this doc is compared
  # to all returned docs (ES feature)
  #facetRarityList=[]
  #for facet in globalFacetDictES:

  #  nrFacetTerms = globalFacetDictES[facet]["nr_terms"]
  #  facetRarityThisFacet = [0]*nrFacetTerms

  #  try:
  #    for term in featureDictDoc[docID]["facetRarity"][facet]:
  #      index = globalFacetDictES[facet]["tti"][term]
  #      rarityValue = featureDictDoc[docID]["facetRarity"][facet][term]
  #      facetRarityThisFacet[index]=rarityValue
  #    facetRarityList.extend(facetRarityThisFacet)
  #  except:
  #    pass

  #vector.extend(facetRarityList)

  # simply the values per facet this doc has (ES feature)
  #facetValueList=[]
  #for facet in globalFacetDictES:
  #  nrFacetTerms = globalFacetDictES[facet]["nr_terms"]
  #  facetValueThisFacet = [0]*nrFacetTerms

  #  try:
  #    for term in featureDictDoc[docID]["facetValues"][facet]:
  #      index = globalFacetDictES[facet]["tti"][term]
  #      facetValueThisFacet[index]= 1
  #    facetValueList.extend(facetValueThisFacet)
  #  except:
  #    pass

  #vector.extend(facetValueList)

  queryString = featureDictQuery["query_string"]

  #print "type(queryString):%s\n" % type(queryString)

  #tf-idf and bm25 score
  tfidf_bm25List = tfidf_bm25.tfidf_bm25_score(queryString, docID)
  #print tfidf_bm25List
  vector.extend(tfidf_bm25List)

  # language models scores
  lmList=[]
  lmList.append( lm.linearInterpolation(queryString, docID, liSmoothingLambda))
  lmList.append( lm.dirichletSmoothing(queryString, docID, dirichletSmoothingAlpha))
  lmList.append( lm.KLdivergence(queryString, docID))

  #print lmList
  for i in range(0, len(lmList)):
    if lmList[i] ==  -float('Inf'):
      lmList[i] = 0
    elif lmList[i] == 0:
      lmList[i] = 0
    else:
      lmList[i] = math.exp(lmList[i])

  #print lmList
  vector.extend(lmList)

  return (qtype, docID, vector)




def makeFacetStatsVector(featureDictQuery):

  facetStatsList = []
  for facet in globalFacetDictES:

    entropy = featureDictQuery["facetStatsDict"][facet]["entropy_norm"]
    termProbDict = featureDictQuery["facetStatsDict"][facet]["term_prob"]

    nrFacetTerms =  globalFacetDictES[facet]["nr_terms"]
    facetListTermProb = [0]*nrFacetTerms

    for term in termProbDict:
      index = globalFacetDictES[facet]["tti"][term]
      facetListTermProb[index] = termProbDict[term]
      #print "term:%s, prob:%2.4f\n" % (term, termProbDict[term])

    facetStatsList.append(entropy)
    facetStatsList.extend(facetListTermProb)

  return facetStatsList

def makeFilterDictVector(featureDictQuery) :
  filterDictList = []

  #  only check catalogi.facet...
  #for facet in globalFacetDictES:
  nrFacetTerms =  globalFacetDictES["catalogi.facet"]["nr_terms"]
  facetList = [0]*nrFacetTerms
  try:
    for facetvalue in featureDictQuery["filterDict"][facet]:
      index = globalFacetDictES[facet]["tti"][facetvalue]
      facetList[index] = 1
  except:
    pass

  filterDictList.extend(facetList)

  keywordsUsed = 0
  keywords = []
  try:
    keywords = globalFacetDictES["trefwoorden.veld_query"]
    keywordsUsed = 1
  except:
    pass
  filterDictList.append(keywordsUsed)

  return (filterDictList, keywords)

def containsPunct(txt) :
  s = string.punctuation
  s2 = re.escape(s)
  s2 = '['+s2+']'
  result = re.findall(s2, txt)
  return len(result) > 0


def pickleDict(featureDict, filename="featuredict_es.dat"):
  f = open(filename,'w')
  pickle.dump(featureDict, f, 2)
  f.close()


def loadDict(filename="featuredict_es.dat"):
  file = open(filename, 'rb')
  s = file.read()
  file.close()

  fd = pickle.loads(s)
  return fd

def loadFeatureDictES(fdName):
  global featureDictES
  global globalFacetDictES

  print "loading featureDictES"
  featureDictES = loadDict(fdName)
  print "loading globalFacetDictES"
  from globalFacetDictPy import dictionary as globalFacetDictES

def splitSaveDict(dictionary, name, nrSplits=10) :

  nrKeys = len(dictionary)

  if (nrSplits > nrKeys):
    nrSplits = nrKeys

  interval = nrKeys / nrSplits
  intervals = map(lambda x: x*interval, range(0, nrSplits))
  intervals.append(nrKeys)

  items = dictionary.items()
  items.sort()

  for i in range(0, len(intervals)-1) :
    start = intervals[i]
    end = intervals[i+1]
    dictSubset = dict(items[start:end])
    pickleDict(dictSubset, name + str(i) + ".dat")




if __name__ == "__main__":

  print "\nhi\n===\n"

  gc.disable()

  importStuff()


  for i in range(0, 30):
    loadFeatureDictES("featureDicts/featuredict_es" + str(i) + ".dat")

    print "\n===\nbatchNr:%d\n===\n" % i

    print len(featureDictES.keys())
    #splitSaveDict(featureDictES, "featuredict_es0", 10)

    process(i)
