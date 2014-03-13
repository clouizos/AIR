from user_specific_positive_negative_examples_dic_strict import dictionary as data
import pickle 

queries = dict()

for user in data.keys():
	half = len(data[user])
	for i, triplet in enumerate(data[user]):
		
		if user in queries:
			queriesOfUser = queries[user]
			queriesOfUser.append(triplet[0])
			queries[user] = queriesOfUser
		else:
			queries[user] = [triplet[0]]

pickle.dump(queries_train, open('userQueries_train', "wb"))
pickle.dump(queries_test, open('userQueries_test', "wb"))
