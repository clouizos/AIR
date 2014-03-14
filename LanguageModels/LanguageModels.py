import redis

from math import log

from nltk.stem.snowball import SnowballStemmer
from math import log

stemmer = SnowballStemmer("dutch")
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
		db_coll = redis.StrictRedis(host='localhost', port=6379, db=5)
        return (index_mono, index_bi, df_mono, df_bi, db_coll)


DB_CONNS = get_db_connections()


NUMBER_OF_DOCUMENTS = len(DB_CONNS[0].keys())



AVERAGE_DOCUMENT_LENGTH = dict({
        'title':3.6178696042268967,
        'description' : 82.0403598843585,
        'summary':45.95869305154022,
        'total':131.6169225401256})
		
		
COLLECTION_LENGTH = dict({
		'title':AVERAGE_DOCUMENT_LENGTH['title'] * NUMBER_OF_DOCUMENTS ,
        'description' : AVERAGE_DOCUMENT_LENGTH['description'] * NUMBER_OF_DOCUMENTS ,
        'summary':AVERAGE_DOCUMENT_LENGTH['summary'] * NUMBER_OF_DOCUMENTS ,
        'total':AVERAGE_DOCUMENT_LENGTH['total'] * NUMBER_OF_DOCUMENTS })		
		

				
				
def linearInterpolation(query, documentID, lambdaVal):

    query = map(lambda x : stem(x), query.lower().split(" "))
	
	descriptor = eval(DB_CONNS[2].get(documentID))
	
	docLength = sum(descriptor['total'].values())
	
	score = 0
	
	for term in query:
		if term in descriptor['total'].keys():
			termCount= descriptor['total'][term]
		else:
			termCount = 0
		score += math.Log((termCount/docLength)*lambdaVal + (DB_CONNS[4].get(term)/(COLLECTION_LENGTH['total']))*(1-lambdaVal))
				

	return score  	


def dirichletSmoothing(query, documentID, alpha):

    query = map(lambda x : stem(x), query.lower().split(" "))
	
	descriptor = eval(DB_CONNS[2].get(documentID))
	
	docLength = sum(descriptor['total'].values())
	
	score = 0
	
	for term in query:
		if term in descriptor['total'].keys():
			termCount= descriptor['total'][term]
		else:
			termCount = 0
		score += math.log ((termCount + alpha * ( DB_CONNS[4].get(term)/COLLECTION_LENGTH['total'])) / (docLength + alpha))

	return score  			
				
				
def KLdivergence(query, documentID):

    query = map(lambda x : stem(x), query.lower().split(" "))
	
	ptq = {} # probability of a term in the query
	
	for term in query:
		if term in ptq.keys():
			ptq[term] += 1
		else:
			ptq[term] = 1
	
	for k,v in ptq :
		ptq[k] = v / len(query)
			
	
	descriptor = eval(DB_CONNS[2].get(documentID))
	
	docLength = sum(descriptor['total'].values())

	
	#calculating alpha
	sumD = 0
	sumC = 0
	
	for term, tf in descriptor['total'].keys():
		sumD += descriptor['total'][term] / docLength
		sumC += DB_CONNS[4].get(term) / COLLECTION_LENGTH['total']
	
	alpha = (1 - sumD) / (1 - sumC)		

	score = 0
	
	for term, probQ in ptq.keys():
		if term in descriptor['total'].keys():
			ptd = descriptor['total'][term] / docLength
			ptc = DB_CONNS[4].get(term) / COLLECTION_LENGTH['total']
			score += probQ * math.log( ptd / (alpha * ptc)) + math.log (alpha)
		
		
	return score  

				
