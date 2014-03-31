# author: Agnes

# reconstructs the dictionaries that result
# from ``vectors.py'' to a few
# shelved dictonaries so that the structure is
# compatible with the divide & conquer algorithm

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
saveDir="/virdir/Scratch/saveFR2_SVM/"

q_docs = {}
q_rels={}
q_docIds={}


def conv() :

  global fromm
  global to

  global q_docs
  global q_rels
  global q_docIds

  nrUsers=0

  for userId in vectorDict:
    for shId in vectorDict[userId]:

      positives = dict(vectorDict[userId][shId]["qd"]["positivesClicked"].items() + vectorDict[userId][shId]["qd"]["positivesPurch"].items())
      docNr=0
      for docId in positives:
        newDocId = str(shId) + ":" + str(docId)
        if docNr==0:
          q_docs[shId]=np.array([positives[docId]])
          q_rels[shId]=np.array([[1]])
          q_docIds[shId]=np.array([[newDocId]])
        else:
          q_docs[shId] = np.concatenate((q_docs[shId],np.array([positives[docId]])),axis=0)
          q_rels[shId] = np.concatenate((q_rels[shId],np.array([[1]])),axis=0)
          q_docIds[shId]=np.concatenate((q_docIds[shId],np.array([[newDocId]])),axis=0)
        docNr+=1

      negatives = vectorDict[userId][shId]["qd"]["negatives"]
      for docId in negatives:
        newDocId = str(shId) + ":" + str(docId)
        if docNr==0: # there was no positive doc
          q_docs[shId]=np.array([negatives[docId]])
          q_rels[shId]=np.array([[0]])
          q_docIds[shId]=np.array([[newDocId]])
        else:
          q_docs[shId] = np.concatenate((q_docs[shId],np.array([negatives[docId]])),axis=0)
          q_rels[shId] = np.concatenate((q_rels[shId],np.array([[0]])),axis=0)
          q_docIds[shId]=np.concatenate((q_docIds[shId],np.array([[newDocId]])),axis=0)

        docNr += 1

      #print user_clicks[userId+":"+shId]
    nrUsers += 1

    if (nrUsers %10) == 0:
    #  saveShelveStuff(dir=saveDir)
      print "processed %d users" % nrUsers

  print "saving..."
  saveShelveStuff(dir=saveDir)

  print "processed %d users" % nrUsers


def saveShelveStuff(dir="saveDP/") :
  if not os.path.exists(dir):
    os.makedirs(dir)

  saveShelve(q_docs, dir+"q_docs.db")
  print "saved q_docs"
  saveShelve(q_rels, dir+"q_rels.db")
  print "saved q_rels"
  saveShelve(q_docIds, dir+"q_docIds.db")
  print "saved q_docIds"



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
