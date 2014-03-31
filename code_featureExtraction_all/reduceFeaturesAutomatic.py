# author: Agnes


import heapq
import gc
import shelve
import numpy as np

dir = 'saveDP8-20small'

queries={}
#uq={}
docs={}
#uc={}
#q_doc={}

maxFeaturesQueries=10
maxFeaturesDocs=80

def checkDocs():

  print "checkDocs()"

  global docs
  #global uq
  #global docs
  #global uc
  #global q_doc

  global maxFeaturesDocs

  sampleSize=6000

  nrDocs=-1
  printInterval=500

  arr=0
  iteration=0
  for docId in docs:

    print docs[docId].shape

    if iteration == 0:
      arr = np.array([docs[docId]])
    else:
      arr = np.concatenate((arr,np.array([docs[docId]])),axis=0)

    if iteration%printInterval == 0:
      print "iteration:%d of %d" % (iteration, sampleSize)
    if iteration == sampleSize:
      break

    iteration+=1

  stdArr = np.std(arr,axis=0)
  meanArr=np.mean(arr,axis=0)

  meanZeros= np.where(meanArr==0)
  #newMeanArr=meanArr
  meanArr[meanZeros]=0.000001

  valueVector = stdArr / meanArr

  largestIndices = [t[0] for t in heapq.nlargest(maxFeaturesDocs, enumerate(valueVector), lambda t: t[1])]

  iteration=0
  for docId in docs:
    docs[docId] = np.array([docs[docId][largestIndices]])
    if iteration%printInterval == 0:
      print "iteration:%d of %d, dim. now %s" % (iteration, nrDocs,docs[docId].shape)
    iteration+=1


def checkQueries():

  print "checkQueries()"

  global queries
  #global uq
  #global docs
  #global uc
  #global q_doc

  global maxFeaturesQueries

  sampleSize=2000

  nrQueries=len(queries)
  printInterval=500

  arr=0
  iteration=0
  for qId in queries:

    #print type(queries[qId])
    if iteration == 0:
      arr = np.array([queries[qId]])
    else:
      arr = np.concatenate((arr,np.array([queries[qId]])),axis=0)

    if iteration%printInterval == 0:
      print "iteration:%d of %d" % (iteration, sampleSize)
    if iteration == sampleSize:
      break

    iteration+=1

  stdArr = np.std(arr,axis=0)
  meanArr=np.mean(arr,axis=0)

  meanZeros= np.where(meanArr==0)
  #newMeanArr=meanArr
  meanArr[meanZeros]=0.000001

  valueVector = stdArr / meanArr

  largestIndices = [t[0] for t in heapq.nlargest(maxFeaturesQueries, enumerate(valueVector), lambda t: t[1])]

  iteration=0
  for qId in queries:
    queries[qId] = np.array([queries[qId][largestIndices]])
    if iteration%printInterval == 0:
      print "iteration:%d of %d, dim. now %s" % (iteration, nrQueries,queries[qId].shape)
    iteration+=1




def load():

  global queries
  global docs

  global dir

  queries = shelve.open("/virdir/Scratch/"+dir+"/queries.db")#,writeback=True)
  docs  = shelve.open("/virdir/Scratch/"+dir+"/docs.db")#,writeback=True)

if __name__=="__main__":
  load()

  checkQueries()

  checkDocs()

