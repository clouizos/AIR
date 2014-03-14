# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 00:21:40 2014

@author: root
"""
import json
import subprocess

def getCurl(query):
    add = 'http://localhost:9876/search_expressie,search_selectie/_search?pretty=1'
    result = subprocess.Popen(['curl','-XGET',add, '-d',query],stdout=subprocess.PIPE)
    resultDic = json.loads(result.stdout.read())
    return resultDic

def saveToFile(text, ID):
    filename = 'result/'+ID+'.txt'
    f = open(filename, 'w')
    f.write(text.encode('utf8'))
    f.close()
    
def saveJson(text,filename):
    f = open(filename,'w')
    f2 = open("b"+filename,'w')
    f.write(json.dumps(text,encoding='latin-1'))
    f2.write(str(text))
    f.close()
    f2.close()
    
def grabDoc(r):
    total = []
    try : 
        total.append(r['_source']['expressie']['niveau']['beschrijving'])
    except :
        pass
    try :
        total.append(r['_source']['selectie']['niveau']['beschrijving'])
    except :
        pass
    try :
        total.append(r['_source']['expressie']['niveau']['samenvatting'])
    except :
        pass
    try :
        total.append(r['_source']['selectie']['niveau']['samenvatting'])
    except :
        pass
                        
    if len(total)>0:
        text = ' '.join(total)
    else:
        print 'total = 00000000'
        text = ''
    return text

def getDocumentList(searchHistoryID):
    try :
        query = str(shDic[searchHistoryID])
    except:
        print 'Search ID is not in searchHistory table'
        #notInSearchHistory.add(searchHistoryID)
        query = ''
            
    if query!='' :
        return getCurl(query)['hits']['hits']
    else :
        return []
getDocumentList
def getRanking(docId, hitList): #ranking starts from 1
    for index in range(len(hitList)):
        if hitList[index]['_id']==docId :
            return index+1
    return None
            
        

#ub = [l.strip('\n') for l in open('BZI_tbl_UserBehavior.txt')]
#sh = [l.strip('\n') for l in open('BZI_tbl_SearchHistory.txt')]

ubsplit = [filter(None, l.split('\t')) for l in [l.strip('\n') for l in open('BZI_tbl_UserBehavior.txt')]]
shsplit = [filter(None, l.split('\t')) for l in [l.strip('\n') for l in open('BZI_tbl_SearchHistory.txt')]]

shDic = {}
for l in shsplit:
    if len(l)<1:
        continue
    shDic[l[0]]=json.loads(l[-1],encoding='latin-1')['jsonQuery']

'''
f = open('shDic.txt','w')
f.write(json.dumps(shDic,encoding='latin-1'))
f.close()
'''

#shDic = json.loads(open('shDic.txt').read(),encoding='latin-1')
ubDic = {}
session_user = {}
withoutNewSearch = set()
notInSearchHistory = set()
noDescription = []
empty = 0
test = False
savedID = []
counter = 0
for l in ubsplit:
    counter+=1
    print str(counter)+' of '+str(len(ubsplit))
    if len(l)<1 :
        continue
    sesID = l[1]
    searchHistoryID = l[-4]
    action = l[3]
    docID = l[4][l[4].rfind('/')+1:l[4].rfind('@')]
    
    session = None
    if l[3] == '17': #create new User to dictionary, and Add current session ID to user ID
        UID = l[-5]                
        try : #already registered userID with new sessionID
            user = ubDic[UID]
            user[sesID]= {}
        except : #first time userID
            ubDic[UID] = {sesID : {}}
        session_user[sesID] = UID #so we can get userID for this sessionID later
        #session = ubDic[UID][sesID]
        #print '--'+UID+'--'+sesID+'--'
    elif l[3] in ['2','4','13','27'] :
        UID = None
        try : #get UserID
	    
            UID = session_user[sesID]
            #print 'sini'
        except : #not found, just consider sessionID as UserID
            UID = sesID            
            ubDic[UID] = {sesID : {}}
            session_user[sesID] = UID 
            #print 'sana'
        
        try :
            session = ubDic[UID][sesID]
        except :
            print UID
            print sesID
            f = open('userBehaviorDict.txt','w')
            f.write(json.dumps(ubDic))
            f.close()
            print l
            raise Exception("I know python!")
    
        search = None
    
        try :
            search = session[searchHistoryID]
        except :
            #print 'view ga tapi ga search?'
            session[searchHistoryID]={}
            search = session[searchHistoryID]
            search['view'] = []
            search['click'] = []
            search['order'] = []
    
    
        if l[3]=='27': #new search, add searchHistoryID to list
            #session[searchHistoryID]={}
            pass
        elif l[3]=='2': #view search result
            #print "action 2"
            searchedID = session.keys() #skip collecting views for the same searchID
            #if searchedID.count(searchHistoryID)==0:
            if len(session[searchHistoryID]['view'])==0:
                results = getDocumentList(searchHistoryID)
                #print 'number documents : '+str(len(results))
                for r in results:
                    #print "result----------"
                    search['view'].append(r['_id'])
                    # save documents to file with itemId as filename
                    #======================================================                        
                    '''                    
                    if savedID.count(r['_id'])==0:
                        savedID.append(r['_id'])
                        text = grabDoc(r)   
                        if text=='':
                            noDescription.append(str(searchHistoryID)+' '+str(r['_id']))
                        saveToFile(text, r['_id'])
                    '''
                    #======================================================
        elif l[3]=='4': #view one document
            #print 'action 4'
            hitList = getDocumentList(searchHistoryID)
            ranking = getRanking(docID, hitList)
            if ranking!=None : #if None it's very weird
                search['click'].append({"id":docID, "rank":ranking})
            #search['click']=list(set(search['click']).add(docID))
        elif l[3]=='13': #order one document
            #print 'action 13'
            hitList = getDocumentList(searchHistoryID)
            ranking = getRanking(docID, hitList)
            if ranking!=None : #if None it's very weird
                search['order'].append({"id":docID, "rank":ranking})
            #search['order']=list(set(so).add(docID))
    if counter%10000 ==0:
        f = open('userBehaviorbDic.txt','w')
        f.write(json.dumps(ubDic))
        f.close()

f = open('userBehaviorDict.txt','w')
f.write(json.dumps(ubDic))
f.close()

f = open('noDescription2.txt','w')
f.write(json.dumps(noDescription))
f.close()

f = open('notInSearchHistory2.txt','w')
f.write(json.dumps(list(notInSearchHistory)))
f.close()
