# AIR project group 2 code -- a feature extraction part
#
# author: Agnes van Belle
#
# get some info for possible features from ES
# supposed to run on server
#
# saves (pickled with dill) dictionary of form:

# userId ->
#  searchHistory ID (i.e. query id) ->
#   "query_string" : raw query string
#   "queryNorm" : queryNorm (float)
#   "filterDict" -> facet or keyword -> values (list) // what values filtered on
#   "containsDate" : bool
#   "facetStatsDict" -> facet -> "entropy_norm" : relative entropy over facet values
#                                                   // (relative to max. entropy given nr. distinct values in result list)
#                                "term_prob" -> value -> probability of that value in that facet
#                                                         // (e.g. if genre is often sports, sports has high prob.)
#   "doc" ->  docId ->
#                   "containsDate" : bool // date is fetched from title/description/summary fields)
#                   "dateMatches": bool (if any found date matches with in query)
#                   "index" : index (this can be search_expressie* or  search_selectie*)
#                   "fieldNormsDict" -> fieldName : fieldNorm // fields are: titel/beschrijving/samenvatting)
#                   "facetRarity" -> facet -> value : 1 / <how many docs in result list share that value for that facet>
#                                                      // e.g. is 1 if it's the only doc with that value for that facet
#                   "facetValues" -> facet : values  (set)
#
# the "facetStatsDict"'s "term_prob" and  "facetValues" and "facetRarity" do not contain values for all facets
# only for those facets whose nr. of possible values is below 1000
# otherwise no features can be made out of it anyway

import pprint
import json
import subprocess
import fileinput
import sys
import math
from collections import defaultdict
import re
import os.path
import dill as pickle

userPNDict = {}

globalFacetDict = {}

def getCurl(query):
  FNULL = open(os.devnull, 'w')
  add = 'http://localhost:9876/search_expressie,search_selectie/_search?pretty=1'
  result = subprocess.Popen(['curl','-XGET',add, '-d',query],stderr=FNULL,stdout=subprocess.PIPE, close_fds=True)
  #result = result.stdout.read()
  realResult = result.communicate()[0]
  result.stdout.close()
  #print realResult
  resultDic = json.loads(realResult)
  return (resultDic, result)


def getEmptyQuery(sampleSize=1000) :

  queryString = '''{"facets":
  {      "dragertype.facet":
          {"terms": {"field": "dragertype.facet", "size": 1000}},
        "genres.facet":
          {"terms": {"field": "genres.facet", "size": 1000}},
        "publicatie_begindatum.facet":
          {"terms": {"field": "publicatie_begindatum.facet", "size": 1000}},
        "zendgemachtigden.facet":
          {"terms": {"field": "zendgemachtigden.facet", "size": 1000}},
        "catalogi.facet":
          {"terms": {"field": "catalogi.facet", "size": 1000}},
        "productie.facet":
          {"terms": {"field": "productie.facet", "size": 1000}},
        "trefwoorden.facet":
          {"terms": {"field": "trefwoorden.facet", "size": 1000}},
        "distributiekanaal.facet":
          {"terms": {"field": "distributiekanaal.facet", "size": 1000}},
        "publicatie_einddatum.facet":
          {"terms": {"field": "publicatie_einddatum.facet", "size": 1000}},
        "dragersoort.facet":
          {"terms": {"field": "dragersoort.facet", "size": 1000}},
        "opnamedatum.facet":
          {"terms": {"field": "opnamedatum.facet", "size": 1000}},
        "kleur.facet":
          {"terms": {"field": "kleur.facet", "size": 1000}},
        "specifieke_programma.facet":
          {"terms": {"field": "specifieke_programma.facet", "size": 1000}}},
    "explain": "true",
    "size":''' + str(sampleSize) + '}'


  return  getCurl(queryString)

# used to get for each facet the max. nr. of possible values
# by issueing empty query
# sampleSize = how many docs allowed to return, to base estimation on
def getGlobalFacetValues(sampleSize=1240, saveToFile=False):

  global globalFacetDict

  if saveToFile or (not os.path.isfile("globalFacetDictPy.py")):

    print "creating and saving globalFacetDictPy.py...\n"
    (resDict, rawResult) = getEmptyQuery(sampleSize)
    facetDict = resDict["facets"]

    for k in facetDict:
      globalFacetDict[k] = {}
      globalFacetDict[k]["nr_terms"] = len(facetDict[k]["terms"])
      globalFacetDict[k]["terms"]=[]
      # >= 1000 possible values means useless for global features
      if not globalFacetDict[k]["nr_terms"] >= 1000:
        for term in facetDict[k]["terms"]:
          globalFacetDict[k]["terms"].append(term["term"])

    save(globalFacetDict, "globalFacetDictPy.py")

  else:
    from globalFacetDictPy import dictionary as globalFacetDict

# save dictionary as .py file (does not work with defaultdict)
def save(dic, fn):
  f = open(fn, "w+")
  f.write( "dictionary = " + str( dic ))
  f.close()

# chekc if userBehaviorDict and searchHistoryDict in /virdir/Scratch exists there
# as py files
def checkForPyFiles():

  pyFileUb = "/virdir/Scratch/userBehaviorDict.py"
  pyFileSh =  "/virdir/Scratch/searchHistoryDict.py"

  if not os.path.isfile(pyFileUb):
    ubDict = json.loads(open("/virdir/Scratch/userBehaviorDict.txt").read(), encoding="utf8")
    print "saving %s" % pyFileUb
    save(ubDict, pyFileUb)

  if not os.path.isfile(pyFileSh):
    shDict = json.loads(open("/virdir/Scratch/searchHistoryDict.txt").read(), encoding="utf8")
    print "saving %s" % pyFileSh
    save(shDict, pyFileSh)

# read userBehaviorDict and searchHistoryDict and create
# new dict (global userPNDict) fom them of form
# userId -> searchHistoryId -> "docs" : document id's
#                           -> "query": raw json query
def readDict():

  global userPNDict

  checkForPyFiles()

  sys.path.insert(0,"/virdir/Scratch/")

  from userBehaviorDict import dictionary as ubDict
  from searchHistoryDict import dictionary as shDict

  print ubDict.keys()[:10]
  print shDict.keys()[:10]

  ubDictKeys = ubDict.keys()

  userPNDict = {}

  for userId in ubDictKeys:
    userPNDict[userId] = {}
    sessionIds  = ubDict[userId].keys()
    for sessionId in sessionIds:
      for shId in ubDict[userId][sessionId]:
        docDictsClicked = ubDict[userId][sessionId][shId]["click"]
        docDictsPurch = ubDict[userId][sessionId][shId]["order"]
        docsViewed = ubDict[userId][sessionId][shId]["view"]
        # no need to save  info for this searchHistory-id  if query had no results
        # AND if no doc was clicked OR purchased... --> then no learning
        if len(docsViewed) > 0 and (len(docDictsClicked) > 0 or len(docDictsPurch) > 0):
          userPNDict[userId][shId] = {}
          userPNDict[userId][shId]["docs"] = docsViewed
          userPNDict[userId][shId]["query"] = shDict[shId]
          # apparently some docs where clicked/purch without being viewed (?) just store these too...
          docDictsClicked.extend(docDictsPurch)
          for dc in docDictsClicked:
            if not dc["id"] in userPNDict[userId][shId]["docs"]:
              userPNDict[userId][shId]["docs"].append(dc["id"])


  #for (k,v) in  userPNDict.iteritems():
  #  print "userId: %s" % k
  #  for (k2, v2) in v.iteritems():
  #    print "shId: %s\n" % k2
  #    print "docs: %s\n" % v2["docs"]


# edit query so to get explanation of docs and
# enough posisble values per facet
def editQuery(query):
  #query["track_scores"] = u"true"
  query["explain"] = u"true"
  #query["sort"] = u"_score"
  for (k,v) in query["facets"].iteritems():
    #print "v: %s" % v
    v["terms"]["size"] = 300
  #query["size"] = 50

  return query

# check which facets and "trefwoorden" was filtered upon
def checkFilter(query):

  filterDict = {}

  #TODO --> dit kan beter

  if "query" in query and "filtered" in query["query"] and "filter" in query["query"]["filtered"] and "bool" in query["query"]["filtered"]["filter"] and "must" in query["query"]["filtered"]["filter"]["bool"]:

    for l in query["query"]["filtered"]["filter"]["bool"]["must"]:
      for (k,v) in l.iteritems():
        for (k2, v2) in v.iteritems():
          if k2[-6:] == ".facet":
            filterDict[k2] = v2

  tws = findKeyWithValue(query, ".veld_query","",keyEndsWith=True, multiple=True, onlyKey=True)

  for tw in tws:
    for k in tw:
      if k.endswith(".veld_query"):
        if "query" in tw[k]:
          filterDict[k] = tw[k]["query"]

  #print filterDict
  return filterDict

# find in a json or dictionary the key(s) encompassing the key and/or value
def findKeyWithValue(myjson, key, value,keyEndsWith=False, startsWith=False, multiple=False, onlyKey=False):

  found=[]

  if type(myjson) == str or type(myjson) == unicode:
    myjson = json.loads(myjson)
  if type(myjson) is dict:
    for jsonkey in myjson:
      if onlyKey and (jsonkey == key or (keyEndsWith and jsonkey.endswith(key))):
        found.append(myjson)
        return found
      elif type(myjson[jsonkey]) in (list, dict):
        found.extend(findKeyWithValue(myjson[jsonkey], key, value,keyEndsWith, startsWith, multiple, onlyKey))
        if not multiple and len(found) >0:
          return found
      elif jsonkey == key or (keyEndsWith and jsonkey.endswith(key)):
        if (myjson[jsonkey] == value) or (startsWith and myjson[jsonkey].startswith(value)):
          found.append(myjson)
          return found
  elif type(myjson) is list:
    for item in myjson:
      if type(item) in (list, dict):
        found.extend(findKeyWithValue(item, key, value,keyEndsWith, startsWith, multiple, onlyKey))
        if not multiple and len(found) > 0:
          return found

  return found


# Shannon entropy
def entropy(l, sumL=None):
  if sumL == None:
    sumL = float(sum(l))
  l = [float(x)/float(sumL) for x in l]

  entrSum = 0.0
  for prob in l:
    if prob != 0:
      entrSum += (prob * math.log(prob,2))

  return (-1) * entrSum

def findDates(text):
  match=re.findall(r'(\d{2}-\d{2}-\d{2})',text)
  return match


def getFieldNorms(expDict):
  fieldNorms = defaultdict(lambda: 0.5)

  fieldDicts = findKeyWithValue(expDict, "description", "weight(", startsWith=True, multiple=True)
  #print "fieldDicts: (len: %d)" % len(fieldDicts)

  for f in fieldDicts:
    fieldName = re.match("weight\((.*?):",f["description"]).group(1)
    if fieldName in fieldNorms:
      break
    #print "fieldName: %s" % fieldName
    fNdict = findKeyWithValue(f, "description", "fieldNorm(", startsWith=True)
    fieldNorms[fieldName] = fNdict[0]["value"]

  return fieldNorms


def getQueryNorm(expDict):

  queryNormValue = 0.5
  qNdicts =  findKeyWithValue(expDict, "description", "queryNorm")
  if qNdicts != [] and "value" in qNdicts[0]:
    queryNormValue = qNdicts[0]["value"]
  return queryNormValue


def tooManyDocsMissFacetValue(nrMissing, nrTotalReturned):
  return nrMissing >= (0.5 * nrTotalReturned)


def getFacetInfo(resDict, nrDocsReturned):

  facetDict = resDict["facets"]

  facetStatsDict = {}

  for (k,v) in facetDict.iteritems():
    #print "facet name: %s" % k
    #print "total: %d, missing:%d, other:%d" % (v["total"], v["missing"], v["other"])
    allTermsList = v["terms"]
    nrTotalTerms = len(allTermsList)
    #print "nrTotalTerms: %d" % nrTotalTerms

    facetStatsDict[k] = {}

    facetStatsDict[k]["term_prob"] = defaultdict(float)

    # if more than half of returned docs misses a value for that facet
    # then the estimation will be off, stick with default value
    sum = 0
    eList = []
    for l in facetDict[k]["terms"]:

      sum += int(l["count"])
      eList.append(int(l["count"]))

      if globalFacetDict[k]["terms"] != []: # if we can make feature from it (not too many values)
        if not tooManyDocsMissFacetValue( int(v["missing"]), nrDocsReturned):
          facetStatsDict[k]["term_prob"][l["term"]] = float(l["count"]) / (float(v["total"]) + float(v["other"]))
        else:
          for anyTerm in globalFacetDict[k]["terms"]:
            facetStatsDict[k]["term_prob"][anyTerm] = 1 / float(globalFacetDict[k]["nr_terms"])
          #facetStatsDict[k]["term_prob"][l["term"]] =  nrTotalTerms/(float(v["total"])+float(v["other"]))


    relEntropy = 0.5
    if not tooManyDocsMissFacetValue( int(v["missing"]), nrDocsReturned):
      entrop = entropy(eList, sum)
      maxEntr = entropy( [1]*nrTotalTerms, nrTotalTerms)
      #print "maxEntropy: %2.2f" % maxEntr
      if maxEntr > 0:
        relEntropy = (entrop / float(maxEntr))

    facetStatsDict[k]["entropy_norm"] = relEntropy


  #for k in facetStatsDict:
  #  for term in facetStatsDict[k]["term_prob"]:
  #    print "k: %s, t: %s, v: %s" % (k, term, facetStatsDict[k]["term_prob"][term])

  return facetStatsDict

def getFacetDocInfo(docId, docIndex, resDict, facetPerDoc, docPerFacet):

  facetPerDoc[docId] = {}
  for k in globalFacetDict:
    if globalFacetDict[k]["terms"] != []:
      foundKeys = findKeyWithValue(resDict['hits']['hits'][docIndex], k[:-6], "", startsWith=True, multiple=True, onlyKey=True)
      facetPerDoc[docId][k] = set([])
      for key in foundKeys:
        value =  key[k[:-6]]
        #print "facet value for %s in doc %s: value: %s" % (k[:-6], docIndex, value)
        if type(value) != list:
          if docId not in docPerFacet[k][value]:
            docPerFacet[k][value].add(docId)
            facetPerDoc[docId][k].add(value)
        else:
          for v in value:
            if docId not in docPerFacet[k][v]:
              docPerFacet[k][v].add(docId)
              facetPerDoc[docId][k].add(v)

  return (facetPerDoc, docPerFacet)


def calcRelativeRarity(queryDict, facetPerDoc, docPerFacet):

  for docId in queryDict["doc"]:

    queryDict["doc"][docId]["facetRarity"]  = {}
    for facet in facetPerDoc[docId]:
      queryDict["doc"][docId]["facetRarity"][facet]=defaultdict(float)
      for value in facetPerDoc[docId][facet]:
        nrOther = len(docPerFacet[facet][value])
        queryDict["doc"][docId]["facetRarity"][facet][value] = 1 / float(nrOther)
        #print "docId: %s, facet: %s, value: %s, score:%2.2f" % (docId, facet, value, 1/float(nrOther))

  return queryDict


def getQueryString(query):
  query_string = findKeyWithValue(query, "query_string", "", onlyKey=True)
  if (query_string != []):
    return query_string[0]["query_string"]["query"]
  return ""

def datesInQuery(query):
  dates = []
  #query_string = findKeyWithValue(query, "query_string", "", onlyKey=True)
  #if (query_string != []):
  #  query_string = query_string[0]["query_string"]["query"]
  query_string = getQueryString(query)
  dates= findDates(query_string)
  return dates

def datesInDoc(docDict):
  summaries = findKeyWithValue(docDict, "samenvatting", "",multiple=True, onlyKey=True)
  descriptions = findKeyWithValue(docDict, "beschrijving", "", multiple=True,onlyKey=True)
  titles  = findKeyWithValue(docDict, "titel", "", multiple=True,onlyKey=True)

  dates = []
  for s in summaries:
    #print s["samenvatting"]
    dates.extend(findDates(json.dumps(s["samenvatting"], ensure_ascii=False).encode("utf8")))

  for t in titles:
    #print t["titel"]
    dates.extend(findDates(json.dumps(t["titel"], ensure_ascii=False).encode("utf8")))

  for d in descriptions:
    #print d["beschrijving"]
    dates.extend(findDates(json.dumps(d["beschrijving"], ensure_ascii=False).encode("utf8")))

  return dates



# here happens the main extraction process, per query
def  processQuery(exampleQuery):
  query = json.loads(exampleQuery)


  queryDict = defaultdict(lambda: None)

  # do smth with query e.g. "track_scores": true
  query  = editQuery(query)

  # get filter-on-facets info from query
  filterDict = checkFilter(query)
  queryDict["filterDict"] = filterDict
  #print "filterDict: %s" % filterDict


  #convert to json again..
  newQuery = json.dumps(query)


  # issue new query
  (resDict, rawResult)  = getCurl(newQuery)

  #print "resDict keys:"
  #for k,v in resDict.iteritems():
  #  print "k:%s" % (k)


  nrReturned =  len(resDict['hits']['hits'])
  #print "nr docs: %d" % nrReturned

  # check if query contains date
  queryDates = datesInQuery(newQuery)
  queryDict["containsDate"] = len(queryDates) > 0

  queryDict["query_string"] = getQueryString(newQuery)

  if nrReturned > 0:

    # get general (only query dependent) facet info
    facetStatsDict = getFacetInfo(resDict, nrReturned)
    for (k,v) in facetStatsDict.iteritems():
      #print "facet %s: %s"% (k,v)
      queryDict["facetStatsDict"] = facetStatsDict


    queryDict["doc"] = defaultdict(lambda: defaultdict(lambda: None))

    facetPerDoc = {}
    docPerFacet = defaultdict(lambda: defaultdict(lambda: set([])))

    # per document
    for docIndex in range(0, nrReturned):
      docId = resDict["hits"]["hits"][docIndex]["_id"]
      #print "docId: %s" % docId

      # check for dates in document's titel/beschrijving/samenvatting fields, and check
      # if any matches with one in the query
      dates = datesInDoc(resDict["hits"]["hits"][docIndex])
      queryDict["doc"][docId]["containsDate"] = len(dates) > 0
      queryDict["doc"][docId]["dateMatches"] = len(set(queryDates) - set(dates)) < len(set(queryDates))

      # get index of doc
      indexName = resDict["hits"]["hits"][docIndex]["_index"] # index: expressie or selectie
      #print "index: %s" % indexName
      queryDict["doc"][docId]["index"] = indexName

      # explanation dict. per doc
      expDict =  resDict["hits"]["hits"][docIndex]["_explanation"]

      # get queryNorm of query
      if docIndex == 0: # do the following for only 1 doc
        queryNormValue =  getQueryNorm(expDict)
        #print "queryNorm: %2.2f" % queryNormValue
        queryDict["queryNorm"] = queryNormValue

      # get field norms dict. per doc
      fieldNormsDict = getFieldNorms(expDict)
      #print "fieldNormsDict: %s" % fieldNormsDict
      queryDict["doc"][docId]["fieldNormsDict"] = fieldNormsDict


      (facetPerDoc, docPerFacet) = getFacetDocInfo(docId, docIndex, resDict, facetPerDoc, docPerFacet)

      queryDict["doc"][docId]["facetValues"] = facetPerDoc[docId]

    # calculate per doc per facet per value the relative rarity of it
    # (score 1 = this doc is the only doc in the result list with that value for that facet)
    queryDict = calcRelativeRarity(queryDict, facetPerDoc, docPerFacet)

  return queryDict

# print a dictionary neatly (also a defaultdict)
def getDictString(d, depth=0) :
  s = ""
  if len(d) > 0:
    for k, v in d.iteritems():
      for i in range(0, depth):
          s += '\t'
      if isinstance(v, defaultdict) or isinstance(v, dict):
        s += "%s --> \n%s\n" % (k, getDictString(v, depth+1))
      else:
        s += "%s --> %s\n" % (k, v)
  else:
    s =  "{}"
  return s


def pickleDict(featureDict, filename="featuredict_es.dat"):
  f = open(filename,'w')
  pickle.dump(featureDict, f, 2)
  f.close()

def loadDict(filename="featuredict_es.dat"):
  s = open(filename, 'rb').read()
  fd = pickle.loads(s)
  return fd


# main loop
def runExample():

  maxQueries = -1


  featureDict = {}

  nrQueriesProcessed = 0

  nrUsers = len(userPNDict.keys() )
  userIds = sorted(userPNDict.keys())

  usersProcessed = 0
  for usersProcessed in range(0, nrUsers):

    user = userIds[usersProcessed]

    print "\n\t=====\n\tprocessed %d users of %d\n\t=====\n" % (usersProcessed, nrUsers)

    usersProcessed += 1

    featureDict[user] = {}

    for shId in userPNDict[user]:
      featureDict[user][shId] = {}

      sh = userPNDict[user][shId]
      query = sh["query"]
      docs = sh["docs"]
      #print "query: %s\n" % query

      #print "query string: %s" % findKeyWithValue(query, "query_string", "", onlyKey=True)[0]["query_string"]["query"]

      queryDict = processQuery(query)

      #print "docs: %s\n" % docs
      #print "docs found again: %s" % queryDict["doc"].keys()

      # this should not happen but check anyway
      if len(set(docs) - set(queryDict["doc"].keys()) ) > 0:
        print "\n\t===\n\twarning: not all docs re-found during re-issueing of query!\n\t===\n"
        queryDict["not_all_refound"] = True
      else:
        queryDict["not_all_refound"]=False

      featureDict[user][shId] = queryDict

      nrQueriesProcessed += 1
      if nrQueriesProcessed % 50 == 0:
        print "processed %d queries" % nrQueriesProcessed

      if maxQueries > 0 and nrQueriesProcessed >= maxQueries:
        break

    # pickle already to be sure  for if crash might happen
    if usersProcessed % 20 == 0:
      pickleDict(featureDict)
      print "pickled dic for first %d users" % usersProcessed

    if maxQueries > 0 and nrQueriesProcessed >= maxQueries:
      break


  #print "feature dict: %s" % getDictString(featureDict)
  pickleDict(featureDict)


if __name__ == "__main__":

  print "\nhi\n=======\n"

  #fD = loadDict()
  #print getDictString( fD[fD.keys()[0]])

  getGlobalFacetValues(sampleSize=5000)
  readDict()
  runExample()

