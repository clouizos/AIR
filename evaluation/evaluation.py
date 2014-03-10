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

	def __init__(self, userID):

		self.queryResults = dict( [ (x[0], (x[1], x[2])) for x in userQueriesAndClicks_strict[userID] ])
		self.relevantDocuments = dict( [ (x[0], x[1]) for x in userQueriesAndClicks_strict[userID]])
		self.allDocuments = dict( [ (x[0], x[1].union(x[2])) for x in userQueriesAndClicks_strict[userID] ])

		print
		print
		print self.queryResults
		print self.relevantDocuments
		print self.allDocuments



	def rank(self):
		docs = list(self.documents)
		shuffle(docs)
		return docs


def test():

	for user in userQueriesAndClicks_strict.keys():
		res = Result(user)


test()
