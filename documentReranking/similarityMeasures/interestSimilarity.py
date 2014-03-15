import userWordVector as vec
import pickle

users = pickle.load(open('../../../users', 'rb'))['users']
usersQueries = pickle.load(open('../../../userQueries_train', 'rb'))


for userA in users:
	print "Evaluating user ", userA
	bestD = 0
	bestUser = "bla"
	a = vec.UserVec(userA)
	relFreqA = a.relativeTermFrequencies
	termsA = relFreqA.keys()
	for userB in users:
		b = vec.UserVec(userB)
		relFreqB = b.relativeTermFrequencies
		dist = 0
		for term in relFreqB.keys():
			if term in termsA:
				dist += relFreqA[term] * relFreqB[term]
		if dist > 0:
			print "Found best!", dist, userB
			bestD = dist
			bestUser = userB
	print "Best match for user (", userA, ") is user (", bestUser, ")", bestD
	print usersQueries[userA]
	print usersQueries[bestUser]
	print





