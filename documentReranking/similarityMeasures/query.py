# query class contains basic information about the query
print "Loading queries..."
queries = pickle.load(open('../../../queries', 'rb'))['queries']


class Query:
	query = ""
	length = 0
	clickedDocuments = []

	def __init__(self, q):
		self.query = q
		self.length = len(self.query)

def test():
	for 

