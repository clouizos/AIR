# user class contains basic information (i.e information I need) about the user

import numpy as np
import pickle

termFrequencies = pickle.load(open('../../../termFrequencies_train', 'rb'))
userQueriesAndClicks_strict = pickle.load(open('../../../user_specific_positive_negative_examples_dic_train', 'rb'))
documents = pickle.load(open('../../../documentContents', 'rb'))

class UserVec:
	id = 0
	termFrequencies = 0
	clickedDocuments = 0
	vocabularySize = 0
	relativeTermFrequencies = 0

	def __init__(self, userID):
		self.id = userID

		# first extract termFrequnecies from the indexed file
		self.termFrequencies = termFrequencies[userID]
		
		# get clickedDocuments
		self.clickedDocuments = self.getClickedDocuments()

		# update term frequencies
		self.termFrequencies = self.updateTermFrequencies()

		# finally get vocabularySize
		self.vocabularySize = sum(self.termFrequencies.values())

		self.relativeTF()

		print "user id: ", id
		print len(self.termFrequencies), " terms"
		print len(self.clickedDocuments), " clicked documents"
		print self.vocabularySize, " words"
		print self.relativeTermFrequencies


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

	def relativeTF(self):
		relativeTermFrequencies = dict()
		for term in self.termFrequencies:
			relFreq = self.termFrequencies[term] / float(self.vocabularySize)
			relativeTermFrequencies[term] = relFreq
		self.relativeTermFrequencies = relativeTermFrequencies
