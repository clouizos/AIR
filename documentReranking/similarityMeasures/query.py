# query class contains basic information about the query

class Query:
	query = ""
	length = 0
	clickedDocuments = []

	def __init__(self, q):
		self.query = q
		self.length = len(self.query)
