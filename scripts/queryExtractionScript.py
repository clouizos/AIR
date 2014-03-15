from user_specific_positive_negative_examples_dic_strict import dictionary as data
import pickle 

queries_train = dict()
queries_test = dict()

for user in data.keys():
	
	half = len(data[user])
	
	for i, triplet in enumerate(data[user]):
		
		if i <= half:
			if user in queries:
				queriesOfUser = queries_train[user]
				queriesOfUser.append(triplet[0])
				queries_train[user] = queriesOfUser
			else:
				queries_train[user] = [triplet[0]]
		else:
			if user in queries:
				queriesOfUser = queries_test[user]
				queriesOfUser.append(triplet[0])
				queries_test[user] = queriesOfUser
			else:
				queries_test[user] = [triplet[0]]

pickle.dump(queries_train, open('userQueries_train', "wb"))
pickle.dump(queries_test, open('userQueries_test', "wb"))
