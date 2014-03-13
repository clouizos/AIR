# user
# Created by Anouk Visser
import pickle

documentClicks = pickle.load(open('../../numberOfDocumentClicks_train', 'rb'))
queriesLeadingToClick = pickle.load(open('../../clickedDocumentsAndQueries_train', 'rb'))


class Document:
	documentID = 0
	numberOfClicks = 0
	queries = []

	def __init__(self, id):
		if id in documentClicks:
			self.documentID = id
			self.numberOfClicks = documentClicks[self.documentID]
			self.queries = queriesLeadingToClick[self.documentID]


 



