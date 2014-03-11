# documentReranking
# Created by Anouk Visser (27-02-2014)

import user as user

numberOfSimilarUsers = 25
minTermsInCommon = 5
userID = 'UID48'
# so we have a user
userA = user.User(userID)
mostSimilarUsers = userA.getMostSimilarUsers(numberOfSimilarUsers, minTermsInCommon)


def getRankingScoreForDocument(similarUsers, document):
	rank = 0
	for uB in similarUsers:
		userID = uB[0]
		similarity = uB[1]
		userB = user.User(userID)
		if userB.didClickDocument(document):
			rank += similarity
	if rank == 0:
		rank = -9990
	return rank

for query in userA.queries:
	documentsToReRank = userA.queryResults[query]
	ranking = []
	for doc in documentsToReRank:
		score = getRankingScoreForDocument(mostSimilarUsers, doc)
		ranking.append((doc, score))
	print "Ranking for query: ", query
	print ranking
	print







