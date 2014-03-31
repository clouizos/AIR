from semanticpy.vector_space import VectorSpace
import pickle

queries = pickle.load(open("QueryStrings.p", "rb"))
print "total queries"
print len(queries)
print "loaded queries"
documents = pickle.load(open("documentContentList2.p", "rb"))
print "loaded documents"
docIds = pickle.load(open("docIdList.p", "rb"))
print len(docIds)
print "loaded doc ids"
documents = list(documents)
print "loaded documents"
print len(documents)
#documents = documents[:100]
#docIds = docIds[:100]
#queries = queries[:10]
vector_space = VectorSpace(documents)
print "finished conversion"

print "load user click file"
userQueriesAndClicks = pickle.load(open("user_specific_positive_negative_examples_dic_test", "rb"))
print "finished loading user click file"
#queryResults = dict( [ (x[0], (x[1], x[2])) for x in userQueriesAndClicks_strict[userID] ])

# given a query and a ranking, this function provides the relevanceJudgements list as 
# required by averagePrecision
def turnIntoBinaryRelevanceThing(query, ranking, relevantDocuments):
        #rel = self.relevantDocuments[query]
        binarized = []
        for doc in ranking:
                if doc in relevantDocuments:
                        binarized.append(1)
                else:
                        binarized.append(0)
        return binarized

def precisionAt(relevanceJudgements):
    return sum(relevanceJudgements) / float(len(relevanceJudgements))

def averagePrecision(relevanceJudgements):
    ap = 0
    for k in range(len(relevanceJudgements)):
        if relevanceJudgements[k] != 0:
            ap += precisionAt(relevanceJudgements[:k+1])
    if(sum(relevanceJudgements) != 0):
        ap = ap / float(sum(relevanceJudgements))
    return ap

def getQueryResults(query, userQueriesAndClicks):
    #print "query fct: " + query
    for item in userQueriesAndClicks:
        #print "userID"
	#print item
	#print "values"
	#print userQueriesAndClicks[item]
        for i in range(len(userQueriesAndClicks[item])):
            #print "query"
	    #print userQueriesAndClicks[item][i][0]
            if(query == userQueriesAndClicks[item][i][0]):
                #print "pos examples"
                #print userQueriesAndClicks[item][i][1]
                #print "neg examples"
                #print userQueriesAndClicks[item][i][2]
                posExamples = userQueriesAndClicks[item][i][1]
                negExamples = userQueriesAndClicks[item][i][1]
                return (list(posExamples), list(negExamples))
    return ([],[])

print "loaded queries and clicks"

map = 0
counter = 0
for query in queries:
    #print "query is: " + query
    queryL = []
    queryL.append(query)
    try:
        results = vector_space.search(queryL)
    except:
        results = []
    
    if (len(results) != 0):
        print "query is: " + query
        #query terms found inside the matrix, sort the results
        retrievedDocs = zip(results, docIds)
        #sort them
        retrievedDocs = sorted(retrievedDocs)
        #print "documents retrieved for query " + query
        #print retrievedDocs
        (a,b) = getQueryResults(query, userQueriesAndClicks)
        if (a != [] and b != []):
            print "pos examples"
            print a
            print "neg examples"
            print b
            relevanceJudgements = turnIntoBinaryRelevanceThing(query, results, a)
            print "relevance judgements"
            print relevanceJudgements
            map += averagePrecision(relevanceJudgements)
            counter += 1

map = map /float(counter)
print "MAP score"
print map
