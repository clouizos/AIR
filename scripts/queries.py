import pickle

strictness = 'strict'

uq = pickle.load(open('userQueries_'+strictness, 'rb'))

queries = dict()
queries['queries'] = set()
for user in uq:
	userQueries = uq[user]
	for query in userQueries:
		queries['queries'].add(query)

#print queries
#print len(queries['queries'])

pickle.dump(queries, open('queries_' + strictness, 'wb'))
