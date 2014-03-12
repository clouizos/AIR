# documentReranking
# Created by Anouk Visser (27-02-2014)

import user as user
import document as document
import pickle
import sys
import similarityMeasures as sims

allUsers = pickle.load(open('../../users_strict', 'rb'))['users']
outputFile = sys.argv[1]

# gives the ranking score for a document based on its popularity (higher score = better)
def getRankingScoreForDocumentPopularityBased(documentID):
	rank = 0
	doc = document.Document(documentID)
	return doc.numberOfClicks

def getRankingScoreForDocument(query, documentID):
	doc = document.Document(documentID)
	queriesLeadingToDoc = doc.queries

	if len(queriesLeadingToDoc) == 0:
		avLevenshteinDistance = -9999
	else:
		avLevenshteinDistance = 0
		for q in queriesLeadingToDoc:
			avLevenshteinDistance += sims.levenshtein(q, query)
		avLevenshteinDistance = avLevenshteinDistance / float(len(queriesLeadingToDoc))
		avLevenshteinDistance = avLevenshteinDistance * -1


 	return avLevenshteinDistance


def createReRankingDump():
	resultingRanks = dict()

	counter = 0
	for userID in allUsers:
		counter += 1
		print "Working on user ", counter
		# take a user
		userA = user.User(userID)

		queryRankingResults = dict()
		
		# get ranking for every query
		for query in userA.queries:

			# get documents
			documentsToReRank = userA.queryResults[query]

			ranking = []
			for doc in documentsToReRank:
				
				# get ranking socre for the document given the mostSimilarUsers
				score = getRankingScoreForDocument(query, doc)
				ranking.append((doc, score))
			
			# sort it (high to low)
			ranking.sort(key=lambda x : x[1], reverse=True)

			# filter out the scores
			toWriteToFile = [elem[0] for elem in ranking]
			
			# create sub dictionary
			queryRankingResults[query] = toWriteToFile
		
		# write results for all queries for one user to the results dictionary
		resultingRanks[userID] = queryRankingResults

	return resultingRanks

results = createReRankingDump()
pickle.dump(results, open(outputFile, 'wb'))








