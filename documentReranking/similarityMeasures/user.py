import numpy as np
import pickle

termFrequencies = pickle.load(open('../../../termFrequencies', 'rb'))
userQueries = pickle.load(open('../../../userQueries', 'rb'))


class User:
	vocabularySize = 0
	queries = []
	termFrequencies = []
	numberOfQueries = 0

	def __init__(self, userID):
		self.queries = userQueries[userID]
		self.termFrequencies = termFrequencies[userID]
		self.vocabularySize = sum(self.termFrequencies.values())
		self.numberOfQueries = len(self.queries)

	# this method has very simple smoothing by +1
	def getTermFrequency(self, term):
		if term in self.termFrequencies:
			termFrequency = self.termFrequencies[term] + 1
		else:
			termFrequency = 1
		return termFrequency

	# returns the number of terms that self has in common with otherUser
	def termsInCommon(self, otherUser):
		return len(set(self.termFrequencies.keys()).intersection(set(otherUser.termFrequencies.keys())))

	def p_q_u(self, query):
		prob = 0
		for term in query.split(" "):
			prob = prob + np.log(self.p_t_u(term))
		return prob

	def p_t_u(self, term):
		result = float(self.getTermFrequency(term)) / float(self.vocabularySize)
		return result