# user
# Created by Anouk Visser
import similarityMeasures as sims
import pickle
import time

allUsers = pickle.load(open('../../users', 'rb'))['users']
userQueriesAndClicks_strict = pickle.load(open('../../user_specific_positive_negative_examples_dic_test', 'rb'))

class User:
	userID = 0
	queries = 0
	queryResults = 0
	clickedDocs = []
	
	def __init__(self, userID):
		self.userID = userID

		qs = set()
		for userInfo in userQueriesAndClicks_strict[self.userID]:
			qs.add(userInfo[0])
		self.queries = qs

		self.queryResults = dict( [ (x[0], x[1].union(x[2])) for x in userQueriesAndClicks_strict[userID] ])
		self.clickedDocs = self.getClickedDocuments();

	# returns 0/1 depending on whether the user clicked the document
	def didClickDocument(self, document):
		click = 0
		if document in self.clickedDocs:
			click = 1
		return click

	def getClickedDocuments(self):
		clicks = set()
		userInfo = userQueriesAndClicks_strict[self.userID]
		for infoTriplet in userInfo:
			c = infoTriplet[1]
			for d in c:
				clicks.add(d)
		return clicks
	
	# THIS SHOULD BE EXTENDED TO FIND MORE FORMS OF SIMILARITY
	def getMostSimilarUsers(self, numberOfMostSimilarUsers, minTermsInCommon, whichModel, whichSim):
		
		# default model is the simple model
		if whichModel == "extended":
		#	print "extended"
			me = sims.UserLMExtended(self.userID)
		else:
			me = sims.UserLM(self.userID)

		mostSimilar = [-9999 for i in range(numberOfMostSimilarUsers)]
		actualUserIDs = [-9999 for i in range(numberOfMostSimilarUsers)]
		for user in allUsers:
			if user != self.userID:
				if whichModel == "extended":
					b = sims.UserLMExtended(self.userID)
				else:
					b = sims.UserLM(self.userID)
				
				filled = False

				# default model is the regular similarity
				if whichSim == "mutual":
				#	print "mutual"
					similarityScore = sims.mutualSim(me, b, minTermsInCommon)
				else:
					similarityScore = sims.sim(me, b, minTermsInCommon)
				
				# find the numberOfMostSimilar users.
				for i in range(len(mostSimilar)):
					if similarityScore > mostSimilar[i] and filled == False:
						for j in range(len(mostSimilar)):
							index = len(mostSimilar) - 1 - j
							if index > i:
								mostSimilar[index] = mostSimilar[index-1]
								actualUserIDs[index] = actualUserIDs[index-1]
						mostSimilar[i] = similarityScore
						actualUserIDs[i] = user
						filled = True

		# We don't want to return things that weren't set, so if we don't meed the numberOfMostSimilarUsers requirement, never mind
		returnList = [(actualUserIDs[i], mostSimilar[i]) for i in range(len(mostSimilar)) if actualUserIDs[i] != -9999]

		return returnList

 



