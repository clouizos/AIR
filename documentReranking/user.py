# user
# Created by Anouk Visser
import similarityMeasures as sims
import pickle

allUsers = pickle.load(open('../../users_strict', 'rb'))['users']

class User:
	
	userID = 0
	
	def __init__(self, userID):
		print "initialize user with userID", userID
		self.userID = userID

	# returns 0/1 depending on whether the user clicked the document
	def didClickDocument(self, document, query):
		print "needs implementing"
		return 1
	
	def getMostSimilarUsers(self, numberOfMostSimilarUsers, minTermsInCommon):
		mostSimilar = [-9999 for i in range(numberOfMostSimilarUsers)]
		actualUserIDs = [-9999 for i in range(numberOfMostSimilarUsers)]
		for user in allUsers:
			if user != self.userID:
				filled = False
				similarityScore = sims.sim(self.userID, user, minTermsInCommon)
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
		return actualUserIDs

 



