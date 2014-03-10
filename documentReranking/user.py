# user
# Created by Anouk Visser

class User:
	
	userID = 0
	queries = []
	clickedDocuments = []
	similarUsersRankedList = [] 
	
	def __init__(self, userID, numberOfMostSimilarUsers):
		print "initialize user with userID", userID
		self.userID = userID

		# implement
		self.queries = []
		self.clickedDocuments = []

		self.similarUsersRankedList = self.mostSimilarUsers(numberOfMostSimilarUsers)

	# returns 0/1 depending on whether the user clicked the document
	def didClickDocument(self, document, query):
		return 1
 



