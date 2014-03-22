# user class contains basic information (i.e information I need) about the user

import numpy as np
import pickle

termFrequencies = pickle.load(open('../../termFrequencies_train', 'rb'))
userQueries = pickle.load(open('../../userQueries_train', 'rb'))
userQueriesAndClicks_strict = pickle.load(open('../../user_specific_positive_negative_examples_dic_train', 'rb'))
documents = pickle.load(open('../../documentContents', 'rb'))

class UserLMExtended:
	id = 0
	termFrequencies = 0
	clickedDocuments = 0
	vocabularySize = 0
	queries = 0
	numberOfQueries = 0

	def __init__(self, userID):
		print "Extended"
		self.id = userID

		# first extract termFrequnecies from the indexed file
		self.termFrequencies = termFrequencies[userID]
		
		# get clickedDocuments
		self.clickedDocuments = self.getClickedDocuments()

		# update term frequencies
		self.termFrequencies = self.updateTermFrequencies()

		# finally get vocabularySize
		self.vocabularySize = sum(self.termFrequencies.values())

		self.queries = userQueries[self.id]
		self.numberOfQueries = len(self.queries)
		
	def getClickedDocuments(self):
		clicks = set()
		userInfo = userQueriesAndClicks_strict[self.id]
		for infoTriplet in userInfo:
			c = infoTriplet[1]
			for d in c:
				clicks.add(d)
		return clicks

	def updateTermFrequencies(self):
		updatedTermFrequencies = self.termFrequencies
		for doc in self.clickedDocuments:
			if doc in documents:
				contents = documents[doc]
				contents = contents.split(' ')
				for term in contents:
					if term in updatedTermFrequencies:
						updatedTermFrequencies[term] += 1
					else:
						updatedTermFrequencies[term] = 1
		return updatedTermFrequencies

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