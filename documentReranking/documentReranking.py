# documentReranking
# Created by Anouk Visser (27-02-2014)

import user as user
import pickle



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

def createReRankingDump():
	resultingRanks = dict()

	numberOfSimilarUsers = 25
	minTermsInCommon = 5
	userID = 'UID48'
	# so we have a user
	userA = user.User(userID)
	mostSimilarUsers = userA.getMostSimilarUsers(numberOfSimilarUsers, minTermsInCommon)

	queryRankingResults = dict()
	
	for query in userA.queries:
		documentsToReRank = userA.queryResults[query]
		ranking = []
		for doc in documentsToReRank:
			score = getRankingScoreForDocument(mostSimilarUsers, doc)
			ranking.append((doc, score))
		ranking.sort(key=lambda x : x[1], reverse=True)
		toWriteToFile = [elem[0] for elem in ranking]
		print toWriteToFile
		queryRankingResults[query] = toWriteToFile
	resultingRanks[userID] = queryRankingResults
	return resultingRanks

print createReRankingDump()








