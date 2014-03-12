# user
# Created by Anouk Visser
import pickle

documentClicks = pickle.load(open('../../numberOfDocumentClicks_strict', 'rb'))
queriesLeadingToClick = pickle.load(open('../../clickedDocumentsAndQueries_strict', 'rb'))


class Document:
	documentID = 0
	numberOFClicks = 0
	queries = 0

	def __init__(self, id):
		self.documentID = id
		self.numberOFClicks = documentClicks[self.documentID]
		self.queries = queriesLeadingToClick[self.documentID]


 



