# user
# Created by Anouk Visser
import similarityMeasures as sims
import pickle

allUsers = pickle.load(open('../../users_strict', 'rb'))['users']
userQueriesAndClicks_strict = pickle.load(open('../../user_specific_positive_negative_examples_dic_strict', 'rb'))

class User:
	userID = 0
	queries = 0
	queryResults = 0
	
	def __init__(self, userID):
		self.userID = userID

		qs = set()
		for userInfo in userQueriesAndClicks_strict[self.userID]:
			qs.add(userInfo[0])
		self.queries = qs

		self.queryResults = dict( [ (x[0], x[1].union(x[2])) for x in userQueriesAndClicks_strict[userID] ])

	# returns 0/1 depending on whether the user clicked the document
	def didClickDocument(self, document):
		click = 0
		userInfo = userQueriesAndClicks_strict[self.userID]
		for infoTriplet in userInfo:
			clickedDocs = infoTriplet[1]
			if document in clickedDocs:
				click = 1
				break
		return click
	
	# THIS SHOULD BE EXTENDED TO FIND MORE FORMS OF SIMILARITY
	def getMostSimilarUsers(self, numberOfMostSimilarUsers, minTermsInCommon):
		mostSimilar = [-9999 for i in range(numberOfMostSimilarUsers)]
		actualUserIDs = [-9999 for i in range(numberOfMostSimilarUsers)]
		for user in allUsers:
			if user != self.userID:
				filled = False
				# HERE WE CAN CALL VIRTUALLY ANY SIMILARITY| FUNCTION, how will we set this up?
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

		returnList = [(actualUserIDs[i], mostSimilar[i]) for i in range(len(mostSimilar))]
		return returnList

 



