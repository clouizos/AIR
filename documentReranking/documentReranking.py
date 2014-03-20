# documentReranking
# Created by Anouk Visser (27-02-2014)

import user as user
import pickle
import sys
from math import sqrt, exp

outputFile = sys.argv[1]
model = sys.argv[2]
simMeausre = sys.argv[3]
numberOfSimilarUsers = int(sys.argv[4])

allUsers = pickle.load(open('../../users', 'rb'))['users']

# gives the ranking score for a document given similarUsers (everything is negative, but higher = better)

# THIS IS VERY INEFFICIENT, it will create new user objects for all the Similar users and that's not good.
def getRankingScoreForDocument(similarUsers, document):
	rank = 1
	# for all users, find the similarity score, add this to 
	# the total ranking score if the user actually clicked the documents
	# if no one clicked the document, set the score to -9999
	for uB in similarUsers:
		userB = uB[0]
		similarity = uB[1]
		if userB.didClickDocument(document):
			rank += similarity
	if rank == 0:
		rank = -9990
	return rank 

def createReRankingDump():
	resultingRanks = dict()

	minTermsInCommon = 5
	counter = 0
	for userID in allUsers:
		counter += 1
		print "Working on user ", counter
		
		# take a user
		userA = user.User(userID)

		# get mostSimilar users
		mostSimilarUsers = userA.getMostSimilarUsers(numberOfSimilarUsers, minTermsInCommon, model, simMeausre)
		mostSimilarUsers = [(user.User(tup[0]), tup[1]) for tup in mostSimilarUsers]

		queryRankingResults = dict()
		
		# get ranking for every query
		for query in userA.queries:

			# get documents
			documentsToReRank = userA.queryResults[query]

			ranking = []

			for doc in documentsToReRank:
				
				# get ranking socre for the document given the mostSimilarUsers
				score = getRankingScoreForDocument(mostSimilarUsers, doc)
				ranking.append((doc, score))
			
			# sort it
			ranking.sort(key=lambda x : x[1], reverse=True)

			# filter out the scores
			toWriteToFile = [elem[0] for elem in ranking]
			
			# create sub dictionary
			queryRankingResults[query] = toWriteToFile
		
		# write results for all queries for one user to the results dictionary
		resultingRanks[userID] = queryRankingResults

	return resultingRanks

print "Writing results to ", outputFile, "\Language Model:", model, "\nSimilarity Measure: ", simMeausre, "\nRe-ranking is done based on ", numberOfSimilarUsers, " most similar users." 
results = createReRankingDump()

pickle.dump(results, open(outputFile, 'wb'))








