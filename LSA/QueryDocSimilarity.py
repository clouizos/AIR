from __future__ import division
import numpy, pickle
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("dutch")

# documents collection - only sample data at the moment
documents = ['How to Bake Bread Without Recipes', 'The Classic Art of Viennese Pastry', 'Numerical Recipes: The Art of Scientific Computing',
                            'Breads, Pastries, Pies and Cakes : Quantity Baking Recipes', 'Pastry: A Book of Best French Recipe']

# each unique term in the vocabulary will represent one row entry for the term-document matrix
terms = ['bak', 'recipe', 'bread', 'cake', 'pastr', 'pie', 'how']

#query
query = "Baking bread"

def stem(x):
    try:
        return stemmer.stem(x)
    except:
        return x

class LatentSemanticIndexing:
    def __init__(self, terms, documents, query):
        self.terms = terms
        self.documents = documents
        self.query = query
        self.A = self.defineTermDocumentMatrix(self.terms, self.documents)
        self.queryVec = self.makeQueryVector(self.terms, self.query)

    def defineTermDocumentMatrix(self, terms, documents):
         #build a term-document matrix
         A = numpy.zeros((len(terms), len(documents)))
         #determine term frequencies
         for i,t in enumerate(self.terms):
             for j,d in enumerate(self.documents):
                 A[i,j] = d.lower().count(t)
         #normalize
         for i in range(len(documents)):
             A[:len(terms),i] = A[:len(terms),i] / numpy.linalg.norm(A[:len(terms),i])
         return A 

    def makeQueryVector(self, terms, query):
        query = map(lambda x : stem(x), self.query.lower().split(" "))
        queryVector = []
        for term in terms:
            if (term in query):
                queryVector.append(1)
            else:
                queryVector.append(0)
        print "query vector"
        print numpy.array(queryVector)
        return numpy.array(queryVector)

    def LSIScores(self, queryVec, documents):
        #normalize the query vector
        print self.queryVec
        if(numpy.linalg.norm(self.queryVec) != 0.0):
            self.queryVec = self.queryVec/numpy.linalg.norm(self.queryVec)
        LSIscores = []
        for i in range(len(documents)):
            docName = "doc" + str(i)
            docScore = numpy.dot(self.A[:len(terms), i].T, self.queryVec)
            ss = {'doc': docName, 'ss': docScore}
            LSIscores.append(ss)
        LSIscores = sorted(LSIscores, key=lambda k: k['ss'], reverse=True)
        return LSIscores

LSI = LatentSemanticIndexing(terms, documents, query)
#print LSI.LSIScores(query, documents)

pickle.dump(LSI.LSIScores(query, documents), open("LSIRanking.p", "wb"))
