# user
# Created by Anouk Visser
import similarityMeasures

class User:
	
	userID = 0
	
	def __init__(self, userID):
		print "initialize user with userID", userID
		self.userID = userID

	# returns 0/1 depending on whether the user clicked the document
	def didClickDocument(self, document, query):
		print "needs implementing"
		return 1

	def getMostSimilarUsers(self, numberOfMostSimilarUsers):
		print "needs implementing"
		return []

 



