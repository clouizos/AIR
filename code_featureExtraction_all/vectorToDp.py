# author: Agnes

# reconstructs the dictionaries that result
# from ``vectors.py'' to a few
# shelved dictonaries so that the structure is
# compatible with the dpRank algorithm

import gc
import sys
import os
import os.path
import numpy as np
import shelve

fromm=0
to=11

vectorDictDir = "vectorDicts3/"
vectorDict={}
saveDir="/virdir/Scratch/saveFR2DP/"

queries ={}
docs={}
user_q={}
user_clicks={}
q_doc={}

def conv() :

  global fromm
  global to

  global queries
  global docs
  global user_q
  global user_clicks
  global q_doc
  global saveDir



  nrUsers=0
  for userId in vectorDict:
    user_q[userId]=set([])
    for shId in vectorDict[userId]:
      user_q[userId].add(shId)

      queryFeatures = vectorDict[userId][shId]["q"]
      queries[shId] = np.array([queryFeatures])

      user_clicks[userId+":"+shId]=[]
      #print "queryFeatures:%s" % queries[shId]


      q_doc[shId]=set([])

      positives = dict(vectorDict[userId][shId]["qd"]["positivesClicked"].items() + vectorDict[userId][shId]["qd"]["positivesPurch"].items())

      for docId in positives:
        newDocId = str(shId) + ":" + str(docId)
        docs[newDocId] = np.array([positives[docId]])
        user_clicks[userId+":"+shId].append((newDocId, 1,))
        q_doc[shId].add(docId)

      negatives = vectorDict[userId][shId]["qd"]["negatives"]

      for docId in negatives:
        newDocId = str(shId)+":"+str(docId)
        docs[newDocId] = np.array([negatives[docId]])
        user_clicks[userId+":"+shId].append((newDocId, 0))
        q_doc[shId].add(docId)

      #print user_clicks[userId+":"+shId]
    nrUsers += 1

    if (nrUsers %10) == 0:
    #  saveShelveStuff(dir=saveDir)
      print "processed %d users" % nrUsers

  print "saving..."
  saveShelveStuff(dir=saveDir)

  print "processed %d users" % nrUsers

#def saveStuff(dir="saveDP/"):
#  saveRawDict(queries, dir+"queries.py")
#  saveRawDict(docs, dir+"docs.py")
#  saveRawDict(user_q, dir+"user_q.py")
#  saveRawDict(user_clicks, dir+"user_clicks.py")
#  saveRawDict(q_doc, dir+"q_doc.py")


def saveShelveStuff(dir="saveDP/") :
  if not os.path.exists(dir):
    os.makedirs(dir)

  saveShelve(queries, dir+"queries.db")
  print "saved queries"
  saveShelve(docs, dir+"docs.db")
  print "saved docs"
  saveShelve(user_q, dir+"user_q.db")
  print "saved user_q"
  saveShelve(user_clicks, dir+"user_clicks.db")
  print "saved user_clicks"
  saveShelve(q_doc, dir+"q_doc.db")
  print "saved q_doc"

def saveShelve(dic, sn):
  #print sn
  s = shelve.open(sn)

  newDic={}
  for key in dic.keys():
    newDic[key.encode('utf-8')] = dic[key]
    del dic[key]
    #s[key.encode("utf-8")]=dic[key]
  s.update(newDic)
  s.close()


#def saveRawDict(dic, fn):
#  f = open(fn, "w")
#  f.write( "dictionary = " + str( dic ))
#  f.close()


def run() :
  global vectorDict
  global vectorDictDir

  global fromm
  global to

  print vectorDictDir

  vectorDict={}

  gc.disable()

  sys.path.insert(0, vectorDictDir)

  print os.listdir(vectorDictDir)

  files = [name for name in os.listdir(vectorDictDir) if name.endswith('py')]
  nrDicts = len(files)

  files = sorted(files)

  print files
  print "nrDicts:%d"%nrDicts

  toAdd=""

  for i in range(max(0,fromm), min(nrDicts, to)) :

    print "loading dict %d" % i

    exec("from vectorDict" + str(i) + " import dictionary as dic" + str(i))

    print "loaded dict %d" % i

    exec("vectorDict.update(dic" + str(i) + ")")

    gc.collect()


  print len(vectorDict.keys())

  conv()

if __name__=="__main__":
  run()
