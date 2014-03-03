# documentReranking
# Created by Anouk Visser (27-02-2014)


# I'm not sure about the design of this file...
# This is just one big user class, which is kind of fun, but maybe not completely useful. 

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


	# needs to be implemented according to sim(a, b)
	# returns a list of size 'numberOfMostSimilarUsers' with users that are most similar
	def mostSimilarUsers(self, numberOfMostSimilarUsers):
		return ['a', 'b', 'c']

	# compares self with specified other user (userID or maybe userclass)
	# returns scalar
	# THERE WILL BE DIFFERENT SIM(...) IMPLEMENTATIONS, what will the strucutre be for this thing?
	def sim(self, otherUser):
		return 0

	# returns 0/1 depending on whether the user clicked the document
	def didClickDocument(self, document, query):
		return 1

	def getCollaborativeDocumentReranking(self):
		ranking = []

		# loops over all documents
			# compute r(query, document, numberOfMostSimilarUsers)

		return ranking
 
	# returns the collaborative ranking score for document d under query q in user u
	def r(query, document, numberOfMostSimilarUsers):
		# gets numberOfMostSimilarUsers similarUsers
		# rankingScore = 0
		# loop over the simmilarUsers
			# rankingScore += self.sim(similarUser) * similarUser.didClickDocument(document, query, similarUser)
		return 0



