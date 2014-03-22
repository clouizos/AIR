# evaluation script

# this is an evaluation script for testing 
# it will open a pickle dump with a dictionary with keys (user) with values dic(query, rankedResults)
# it uses the information in either user_specific_positive_negative_examples_dic_strict or the unstrict variant to 
# evaluate the MAP of the rankedResults

# it also provides a random ranking as a baseline!

import pickle
from random import shuffle
import sys


filename = sys.argv[1]
print "Testing results from ", filename

userQueriesAndClicks_strict = pickle.load(open('../../user_specific_positive_negative_examples_dic_test', 'rb'))
evaluationFile = pickle.load(open(filename, 'rb'))

class Result:
	queryResults = 0
	relevantDocuments = 0
	allDocuments = 0
	randomResults = 0

	def __init__(self, userID):
		self.queryResults = dict( [ (x[0], (x[1], x[2])) for x in userQueriesAndClicks_strict[userID] ])
		self.relevantDocuments = dict( [ (x[0], x[1]) for x in userQueriesAndClicks_strict[userID]])
		self.allDocuments = dict( [ (x[0], x[1].union(x[2])) for x in userQueriesAndClicks_strict[userID] ])
		self.randomResults = self.rankRandom()

	# assigns a random ranking for baseline testing purposes
	def rankRandom(self):
		randomRes = dict()
		for query in self.allDocuments:
			tmp = list(self.allDocuments[query])
			shuffle(tmp)
			randomRes[query] = tmp
		return randomRes

	# given a query and a ranking, this function provides the relevanceJudgements list as 
	# required by averagePrecision
	def turnIntoBinaryRelevanceThing(self, query, ranking):
		rel = self.relevantDocuments[query]
		binarized = []
		for doc in ranking:
			if doc in rel:
				binarized.append(1)
			else:
				binarized.append(0)
		return binarized

# test the test, haha
def test():
	overallMAPRandom = 0
	overallMap = 0
	overallMRRRandom = 0
	overallMRR = 0
	for user in userQueriesAndClicks_strict.keys():
		userInfo = userQueriesAndClicks_strict[user]
		res = Result(user)
		mapRandom = 0
		map = 0
		mrrRandom = 0
		mrr = 0
		for infoTriplet in userInfo:
			
			# get a random ranking for the query 
			ranking = res.randomResults[infoTriplet[0]]
			relevanceJudgementsRANDOM = res.turnIntoBinaryRelevanceThing(infoTriplet[0], ranking)
			
			# get the raking from the evaluation file
			ranking = evaluationFile[user][infoTriplet[0]]
			relevanceJudgements = res.turnIntoBinaryRelevanceThing(infoTriplet[0], ranking)

			mapRandom += averagePrecision(relevanceJudgementsRANDOM)
			map += averagePrecision(relevanceJudgements)

			mrrRandom += mrr(relevanceJudgementsRANDOM)
			mrr += mrr(relevanceJudgements)
	
		mapRandom = mapRandom / float(len(userInfo))
		map = map / float(len(userInfo))
		mrrRandom = mrrRandom / float(len(userInfo))
		mrr = mrr / float(len(userInfo))
		overallMAPRandom += mapRandom
		overallMap += map
		overallMRRRandom += mrrRandom
		overallMRR += mrr

	
		print "MAP: ", map, " Random: ", mapRandom, " Difference: ", map - mapRandom
	overallMAPRandom = overallMAPRandom / float(len(userQueriesAndClicks_strict.keys()))
	overallMap = overallMap / float(len(userQueriesAndClicks_strict.keys()))
	overallMRRRandom = overallMRRRandom / float(len(userQueriesAndClicks_strict.keys()))
	overallMRR = overallMRR / float(len(userQueriesAndClicks_strict.keys()))

	print "Overall map = ", overallMap, " (random = ", overallMAPRandom, ")"
	print "Overall mrr = ", overallMRR, " (random = ", overallMRRRandom, ")"


# Calculates precision at rank len(relevanceJudgements) (so the caller should provide the right k already)
# input ranked list of document with indicator 1 for relevant, 0 for irrelevant example: [1, 1, 0, 0, 1, 0]
def precisionAt(relevanceJudgements):
	return sum(relevanceJudgements) / len(relevanceJudgements)

# Calculates average precision of a ranked list
# input ranked list of document with indicator 1 for relevant, 0 for irrelevant example: [1, 1, 0, 0, 1, 0]
def averagePrecision(relevanceJudgements):
	ap = 0
	for k in range(len(relevanceJudgements)):
		if relevanceJudgements[k] != 0:
			ap += precisionAt(relevanceJudgements[:k+1])
	ap = ap / float(sum(relevanceJudgements))
	return ap

def mrr():
	mrr = 0
	for i in xrange(len(relevanceJudgements)):
		if relevanceJudgements[i] == 1:
			mrr = i
			break
	return mrr
test()
