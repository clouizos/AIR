# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 00:21:40 2014

@author: root
"""
import json
import subprocess
import multiprocessing as mp

def getCurl(query):
    add = 'http://localhost:9876/search_expressie,search_selectie/_search?pretty=1'
    result = subprocess.Popen(['curl','-XGET',add, '-d',query],stdout=subprocess.PIPE)
    resultDic = json.loads(result.stdout.read())
    return resultDic

def saveToFile(text, ID):
    filename = '../../virdir/Scratch/extractedDocNew/'+ID+'.txt'
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
        total.append('--titel-- '+r['_source']['expressie']['titel']['tekst'])
    except :
        pass
    try : 
        total.append('--titel-- '+r['_source']['selectie']['titel']['tekst'])
    except :
        pass
    try : 
        total.append('--opnamedatum-- '+r['_source']['expressie']['ext']['opnamedatum'])
    except :
        pass
    try : 
        total.append('--opnamedatum-- '+r['_source']['selectie']['ext']['opnamedatum'])
    except :
        pass
    try : 
        total.append('--geografische_namen-- '+r['_source']['expressie']['niveau']['geografische_namen'])
    except :
        pass
    try : 
        total.append('--geografische_namen-- '+r['_source']['selectie']['niveau']['geografische_namen'])
    except :
        pass    
    try : 
        maker = r['_source']['expressie']['maker']
        mn = ''
        for m in maker :
            mn += (m['naam']+' - ')
        if len(mn)>2:
            total.append('--maker-- '+mn[:-2]) #remove the last "- "
    except :
        pass
    try : 
        maker = r['_source']['selectie']['maker']
        mn = ''
        for m in maker :
            mn += (m['naam']+' ')
        if len(mn)>2:
            total.append('--maker-- '+mn[:-2])
    except :
        pass    
    
    try : 
        total.append('--beschrijving-- '+r['_source']['expressie']['niveau']['beschrijving'])
    except :
        pass
    try :
        total.append('--beschrijving-- '+r['_source']['selectie']['niveau']['beschrijving'])
    except :
        pass
    try :
        total.append('--samenvatting-- '+r['_source']['expressie']['niveau']['samenvatting'])
    except :
        pass
    try :
        total.append('--samenvatting--'+r['_source']['selectie']['niveau']['samenvatting'])
    except :
        pass
                        
    if len(total)>0:
        text = ' '.join(total)
    else:
        text = ''
    return text
    
def process(li):
    if len(li)<1 :
        return
    searchID = li[-4]
    
    if li[3] in ['2','4','13','27']: #this action has searchID propery
    # save documents to file with itemId as filename
    #================================================
        
        if searchedID.count(searchID)==0:
            searchedID.append(searchID)
            try :
                query = str(json.loads(shDic[searchID][-1])['jsonQuery'])
            except:
                print 'Search ID is not in searchHistory table'
                query = ''
            
            if query!='' :
                results = getCurl(query)['hits']['hits']
                for r in results:
                    if savedID.count(r['_id'])==0:
                        savedID.append(r['_id'])
                        text = grabDoc(r)   
                        saveToFile(text, r['_id'])

#ub = [l.strip('\n') for l in open('BZI_tbl_UserBehavior.txt')]
#sh = [l.strip('\n') for l in open('BZI_tbl_SearchHistory.txt')]
if __name__ == '__main__':
    ubsplit = [filter(None, l.split('\t')) for l in [l.strip('\n') for l in open('BZI_tbl_UserBehavior.txt')]]
    shsplit = [filter(None, l.split('\t')) for l in [l.strip('\n') for l in open('BZI_tbl_SearchHistory.txt')]]

    shDic = {}
    for l in shsplit:
        if len(l)<1:
            continue
        shDic[l[0]]=l

    counter = 0
    savedID = []
    searchedID = []
    pool = mp.Pool()
    for l in ubsplit:
        counter+=1
        print str(counter)+' of '+str(len(ubsplit))
        pool.apply_async(process, args = (l, ))
    pool.close()
    pool.join()