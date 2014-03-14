# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 11:15:20 2014

@author: root
"""
from __future__ import division
import json
import operator

ubDict = json.loads(open('userBehaviorDict.txt').read())

nUser = len(ubDict)
nQueryClickPerUsers = [] #will contain tuple (userId, nquery, nview, nclick, norder, avgClickPerQuery)
docViewed = {}
docClicked = {} #will contain tuple (docId, numberOfClick)
docOrdered = {}
docUniqueClicked = {}
count=0
queryClicked = {}
for uid,v in ubDict.iteritems():
    nquery = 0
    nview = 0
    nclick = 0
    norder = 0
    nclickUniq = set()
    for sid, v1 in v.iteritems():
        nquery+=len(v1)
        for qid , v2 in v1.iteritems():
            nview+=len(v2["view"])
            nclick+=len(v2["click"])
            queryClicked[qid]=len(v2["click"])
            #print v2["click"]
            #print len(v2["click"])
            norder+=len(v2["order"])
            for m in v2["view"]:
                try :
                    docViewed[m]=docViewed[m]+1
                except:
                    docViewed[m]=1
            for m in v2["click"]:
                nclickUniq.add(m["id"])
                try :
                    docClicked[m["id"]]=docClicked[m["id"]]+1
                except:
                    docClicked[m["id"]]=1
                try : 
                    docUniqueClicked[m["id"]].add(uid)
                except :
                    docUniqueClicked[m["id"]] = set([uid])
            for m in v2["order"]:
                try :
                    docOrdered[m["id"]]=docOrdered[m["id"]]+1
                except:
                    docOrdered[m["id"]]=1
    try : 
        avg = nclick/nquery
        avgUnique = len(nclickUniq)/nquery
    except :
        count+=1
        avg = 0
        avgUnique
    #print nclick
    #print len(nclickUniq)
    nQueryClickPerUsers.append((uid,nquery,nview,nclick,norder,avg, len(nclickUniq), avgUnique))
    
    
sorted_viewed = sorted(docViewed.iteritems(), key=operator.itemgetter(1), reverse = True)
sorted_clicked = sorted(docClicked.iteritems(), key=operator.itemgetter(1), reverse = True)
sorted_ordered = sorted(docOrdered.iteritems(), key=operator.itemgetter(1), reverse = True)
sorted_unique_clicked = sorted(docUniqueClicked, key=lambda k: len(docUniqueClicked[k]), reverse = True)
twoUserDoc = dict((k, v) for k, v in docUniqueClicked.items() if len(v)>1)
sorted_queryClicked = sorted(queryClicked.iteritems(), key=operator.itemgetter(1), reverse = True)


mostQuery = sorted(nQueryClickPerUsers,key=lambda x: x[1], reverse = True)
mostClick = sorted(nQueryClickPerUsers,key=lambda x: x[3], reverse = True)
mostOrder = sorted(nQueryClickPerUsers,key=lambda x: x[4], reverse = True)
mostAvg = sorted(nQueryClickPerUsers,key=lambda x: x[5], reverse = True)
mostUniqueClick = sorted(nQueryClickPerUsers,key=lambda x: x[6], reverse = True)

print("Total number of user : "+str(nUser))
print("Number of users with at least one query : "+str(len([i for i in mostQuery if i[1]>0])))
print("Number of users with more than one query : "+str(len([i for i in mostQuery if i[1]>0])))
print("Average number of query per user : "+str(sum([i[1] for i in mostQuery])/len(mostQuery)))

print("Number of users with at least one click : "+str(len([i for i in mostClick if i[3]>0])))
print("Number of users with more than one click : "+str(len([i for i in mostClick if i[3]>1])))
#print("Number of users with at least one unique click : "+str(len([i for i in mostUniqueClick if i[6]>0])))
print("Number of users with more than one unique click : "+str(len([i for i in mostUniqueClick if i[6]>1])))
print("Average click per user : "+str(sum([i[3] for i in mostClick])/len(mostClick)))
print("Average unique click per user : "+str(sum([i[6] for i in mostUniqueClick])/len(mostClick)))

print("Number of users with at least one purchase: "+str(len([i for i in mostOrder if i[4]>0])))
print("Number of users with more than one purchase: "+str(len([i for i in mostOrder if i[4]>1])))
print("Average purchase per user : "+str(sum([i[4] for i in mostOrder])/len(mostOrder)))

print("Number of query : "+str(len(queryClicked)))
print("Number of query with at least one click : "+str(len([i for i in sorted_queryClicked if i[1]>0])))
print("Average click per query : "+str(sum([i[1] for i in sorted_queryClicked])/len(queryClicked)))

print("Number of documents clicked at least once :"+str(len(docUniqueClicked)))
print("Number of documents was clicked at least by 2 different users :"+str(len(twoUserDoc)))


print("Top10 users with highest number of query:")
print("#\tUserID")
for i in mostQuery[:10]:
    print(str(i[1])+'\t'+str(i[0]))

print("Top10 users with highest number of click:")
print("#\tUserID")
for i in mostClick[:10]:
    print(str(i[3])+'\t'+str(i[0]))

print("Top10 users with highest number of unique click:")
print("#\tUserID")
for i in mostUniqueClick[:10]:
    print(str(i[6])+'\t'+str(i[0]))

print("Top10 users with highest number of order:")
print("#\tUserID")
for i in mostOrder[:10]:
    print(str(i[4])+'\t'+str(i[0]))

print("Top10 viewed documents :")
print("#\tDocID:")
for i in sorted_viewed[:10]:
    print(str(i[1])+'\t'+str(i[0]))

print("Top10 clicked documents :")
print("#\tDocID:")
for i in sorted_clicked[:10]:
    print(str(i[1])+'\t'+str(i[0]))

print("Top10 clicked documents by unique user:")
print("#\tDocID:")
for i in sorted_unique_clicked[:10]:
    print(str(len(docUniqueClicked[i]))+'\t'+i)
    
print("Top10 ordered documents :")
print("#\tDocID:")
for i in sorted_ordered[:10]:
    print(str(i[1])+'\t'+str(i[0]))