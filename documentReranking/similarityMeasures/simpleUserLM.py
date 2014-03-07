# I'm feeling it's not so good to rely on the termfrequencies in the user class...
import user as user 
import pickle

termFrequencies = pickle.load(open('../../../termFrequencies', 'rb'))
userQueries = pickle.load(open('../../../userQueries', 'rb'))


def sim(userA, userB, minCommonTerms):
	# create two user objects
	A = user.User(userA)
	B = user.User(userB)
	similarityScore = 0

	# check if they have minCommonTerms terms in common
	# if not, return 0
	if A.termsInCommon(B) > minCommonTerms:
		# calculate the similarity
		for query in A.queries:
			similarityScore += B.p_q_u(query)
		similarityScore = similarityScore / A.numberOfQueries
	else:
		similarityScore = -9999

	return  similarityScore

def mutualSim(userA, userB, minCommonTerms):
	A = user.User(userA)
	B = user.User(userB)
	similarityScoreAB = 0
	similarityScoreBA = 0

	# check if they have minCommonTerms terms in common
	# if not, return 0
	if A.termsInCommon(B) > minCommonTerms:
		# calculate the similarity A -> B
		for query in A.queries:
			similarityScoreAB += B.p_q_u(query)
		similarityScoreAB = similarityScoreAB / A.numberOfQueries

		for query in B.queries:
			similarityScoreBA += A.p_q_u(query)
		similarityScoreBA = similarityScoreBA / B.numberOfQueries
	else:
		similarityScoreAB = -9999

	return similarityScoreAB + similarityScoreBA


def hoi():
	print "Found ", len(userQueries.keys()), " users"
	print len(termFrequencies.keys()), " users have queries"

	for userA in termFrequencies.keys():
		bestMatch = -9999
		bestUser = 0
		for userB in termFrequencies.keys():
			if userB != userA:
				similarity = mutualSim(userA, userB, 5)
				if similarity > bestMatch:
					bestMatch = similarity
					bestUser = userB
		if bestUser != 0:
			print
			print "Best matching score: ", bestMatch
			print "Queries user A: ", user.User(userA).queries
			print "Queries user B: ", user.User(bestUser).queries
			print "#queries A: ", len(user.User(userA).queries)
			print "#queries B: ", len(user.User(bestUser).queries)
		else:
			print 
			print "No best match!"




hoi()