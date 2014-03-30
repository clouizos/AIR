import pickle

usersMC = pickle.load(open('filename_for_user_mixture_stuff', 'rb'))
info = pickle.load(open('../../user_specific_positive_negative_examples_dic_strict', 'rb'))


def cosineSimilarity(vec1, vec2):
	sim = 0
	# normalize vectors
	lenVec1 = sqrt( sum([v**2 for v in vec1]) )
	lenVec2 = sqrt( sum([v**2 for v in vec2]) )
	vec1 = [v / lenVec1 for v in vec1]
	vec2 = [v / lenVec2 for v in vec2]
	if len(vec1) == len(vec2):
		for i in xrange(len(vec1)):
			sim += vec1[i] * vec2[i]
	else:
		sim = -1
	return sim

def getQueriesForUser(userID):
	queries = []
	for infoTriplet in info[userID]:
		queries.append(infoTriplet[0])
	return queries

def printMostSimilarUsers():
	for user in usersMC:
		bestSim = -1
		bestUser = -1
		found = False
		for userB in usersMC:
			if userB != user:
				sim = cosineSimilarity(userMC[user], userMC[userB])
				if sim > bestSim:
					found = True
					bestSim = sim
					bestUser = userB
		if found:
			print getQueriesForUser(user)
			print getQueriesForUser(bestUser)

printMostSimilarUsers()


