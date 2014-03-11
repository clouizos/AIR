# documentReranking
# Created by Anouk Visser (27-02-2014)

import user as user

numberOfSimilarUsers = 5
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
	return rank
	
ranking = []
for query in userA.queries:
	documentsToReRank = userA.queryResults[query]
	for doc in documentsToReRank:
		score = getRankingScoreForDocument(mostSimilarUsers, doc)
		ranking.append((doc, score))
print ranking







