# evaluation script

# this is an evaluation script for testing 
# it will open a pickle dump with a dictionary with keys (user) with values dic(query, rankedResults)
# it uses the information in either user_specific_positive_negative_examples_dic_strict or the unstrict variant to 
# evaluate the MAP of the rankedResults

# it also provides a random ranking as a baseline!
from random import shuffle
userQueriesAndClicks_strict = pickle.load(open('../../../user_specific_positive_negative_examples_dic_strict', 'rb'))


class Result:
	documents = 0
	resultRanking = 0
	actualRelevantDocuments = 0

	def __init__(self, userID, query):
		self.actualRelevantDocuments = userQueriesAndClicks_strict[userID][query][1]
		self.documents = userQueriesAndClicks_strict[userID][query][1].union(userQueriesAndClicks_strict[userID][query][2])
		self.resultRanking = rank()

		print self.actualRelevantDocuments
		print self.documents
		print self.resultRanking
		print

	def rank(self):
		docs = list(self.documents)
		shuffle(docs)
		return docs


def test():
	for user in userQueriesAndClicks_strict.keys():
		for query in userQueriesAndClicks_strict[user]:
			res = Result(user, query)


test()