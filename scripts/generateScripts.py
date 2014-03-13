# This does everything you need!
import pickle 
import copy as copy

print "Opening the big data file!"
data = pickle.load(open('../../user_specific_positive_negative_examples_dic_strict', 'rb'))


def add(dic, key, value):
	if key in dic:
		theSet = dic[key]
		theSet.add(value)
		dic[key] = theSet
	else:
		print value
		dic[key] = set([value])
	
	return dic

#userQueries_strict
user_queries_train = dict()
user_queries_test = dict()

#queries_strict            
queries_train = dict()
queries_test = dict()

queries_train['queries'] = set()
queries_test['queries'] = set()

#clickedDocumentsAndQueries_strict 
dcq_train = dict()
dcq_test = dict()

#numberOfDocumentClicks_strict               
document_clicks_train = dict()
document_clicks_test = dict() 

#termFrequencies_strict
termFrequencies_train = copy.copy(user_queries_train)
termFrequencies_test = copy.copy(user_queries_test)   

# user_specific_positive_negative_examples_dic_strict
user_specific_positive_negative_examples_dic_train = dict()
user_specific_positive_negative_examples_dic_test = dict()

print "Starting!"
for user in data.keys():
	
	half = len(data[user]) / 2
	printTrain = False
	printTest = False
	
	print "This user has ", len(data[user]), " triplets, we'll switch at ", half
	for i, triplet in enumerate(data[user]):


		query = triplet[0]
		clicks = triplet[1]

		# train
		if i <= half:
			
			user_specific_positive_negative_examples_dic_train = add(user_specific_positive_negative_examples_dic_train, user, triplet)

			if printTrain == False:
				print "(", i, ") Train"
				printTrain = True

			if user in user_queries_train:
				queriesOfUser = user_queries_train[user]
				queriesOfUser.append(query)
				user_queries_train[user] = queriesOfUser
			else:
				user_queries_train[user] = [query]
			
			queries_train['queries'].add(query)
			
			for c in clicks:
				dcq_train = add(dcq_train, c, query)
				
				if c in document_clicks_train:
					document_clicks_train[c] += 1
				else:
					document_clicks_train[c] = 1
		# test
		else:
			user_specific_positive_negative_examples_dic_test = add(user_specific_positive_negative_examples_dic_test, user, triplet)

			if printTest == False:
				print "(", i, ") Test"
				printTest = True

			if user in user_queries_test:
				queriesOfUser = user_queries_test[user]
				queriesOfUser.append(triplet[0])
				user_queries_test[user] = queriesOfUser
			else:
				user_queries_test[user] = [triplet[0]]

			queries_test['queries'].add(triplet[0])

			for c in clicks:
				dcq_test = add(dcq_test, c, query)

				if c in document_clicks_test:
					document_clicks_test[c] += 1
				else:
					document_clicks_test[c] = 1

# tHis is the only thing that is not so nice...
for key in user_queries_train.keys():
	queryList = user_queries_train[key]
	x = [s.split(" ") for s in queryList]
	listOfTerms = sum(x, [])
	queryFrequency = dict( [ (term, listOfTerms.count(term)) for term in set(listOfTerms) ] )
	termFrequencies_train[key] = queryFrequency

for key in user_queries_test.keys():
	queryList = user_queries_test[key]
	x = [s.split(" ") for s in queryList]
	listOfTerms = sum(x, [])
	queryFrequency = dict( [ (term, listOfTerms.count(term)) for term in set(listOfTerms) ] )
	termFrequencies_test[key] = queryFrequency


print "userQueries: ", len(user_queries_train), len(user_queries_test)
print "termFrequencies: ", len(termFrequencies_train), len(termFrequencies_test)
print "Big thing: ", len(user_specific_positive_negative_examples_dic_test), len(user_specific_positive_negative_examples_dic_train)

if user_queries_train == user_queries_test or termFrequencies_test == termFrequencies_train or user_specific_positive_negative_examples_dic_test == user_specific_positive_negative_examples_dic_train:
	print "PANIC!!"

pickle.dump(user_queries_train, open('../../userQueries_train', 'wb'))
pickle.dump(user_queries_test, open('../../userQueries_test', 'wb'))

pickle.dump(queries_train, open('../../queries_train', 'wb'))
pickle.dump(queries_test, open('../../queries_test', 'wb'))

pickle.dump(dcq_train, open('../../clickedDocumentsAndQueries_train', 'wb'))
pickle.dump(dcq_test, open('../../clickedDocumentsAndQueries_test', 'wb'))

pickle.dump(document_clicks_train, open('../../numberOfDocumentClicks_train', 'wb'))
pickle.dump(document_clicks_test, open('../../numberOfDocumentClicks_test', 'wb'))

pickle.dump(termFrequencies_train, open('../../termFrequencies_train', 'wb'))
pickle.dump(termFrequencies_test, open('../../termFrequencies_test', 'wb'))


pickle.dump(user_specific_positive_negative_examples_dic_train, open('../../user_specific_positive_negative_examples_dic_train', 'wb'))
pickle.dump(user_specific_positive_negative_examples_dic_test, open('../../user_specific_positive_negative_examples_dic_test', 'wb'))







