# evaluation script

# this is an evaluation script for testing 
# it will open a pickle dump with a dictionary with keys (user) with values dic(query, rankedResults)
# it uses the information in either user_specific_positive_negative_examples_dic_strict or the unstrict variant to 
# evaluate the MAP of the rankedResults


# it also provides a random ranking as a baseline!
import pickle
from random import shuffle
userQueriesAndClicks_strict = pickle.load(open('../../user_specific_positive_negative_examples_dic_strict', 'rb'))

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

	def rankRandom(self):
		randomRes = dict()
		for query in self.allDocuments:
			tmp = list(self.allDocuments[query])
			shufle(tmp)
			randomRes[query] = tmp
		return randomRes

	def turnIntoBinaryRelevanceThing(self, query, ranking):
		rel = self.relevantDocuments[query]
		binarized = []
		for doc in ranking:
			if doc in rel:
				binarized.append(1)
			else:
				binarized.append(0)

def test():
	for user in userQueriesAndClicks_strict.keys():
		userInfo = userQueriesAndClicks_strict[user]
		res = Result(user)
		map = 0
		for infoTriplet in userInfo:
			# get a ranking for the query (now we're just taking a random I guess)
			ranking = res.randomResults[infoTriplet[0]]
			relevanceJudgements = res.turnIntoBinaryRelevanceThing(infoTriplet[0], ranking)
			map += averagePrecision(relevanceJudgements)
		map = map / float(len(userInfo))
		print "Map for user ", user, " is ", map

def precisionAt(relevanceJudgements):
	return sum(relevanceJudgements) / len(relevanceJudgements)

def averagePrecision(relevanceJudgements):
	ap = 0
	for k in range(len(relevanceJudgements)):
		if relevanceJudgements[k] != 0:
			ap += precisionAt(relevanceJudgements[:k+1])
	ap = ap / float(sum(relevanceJudgements))


test()
