import pickle

from userQueries import dic as userQueries

def saveCompilation(dic, filename="/virdir/Scratch/data/compiled.dict"):
	"""
		Save dictionary to disc.
	"""
	pickle.dump( dic, open( filename, "wb" ) )
	return

def loadCompilation(filename="/virdir/Scratch/data/compiled.dict"):
	"""
		Save dictionary to disc.
	"""
	return pickle.load( open( filename, "rb" ) )


for key in userQueries.keys():
	queryList = userQueries[key]
	x = [s.split(" ") for s in queryList]
	listOfTerms = sum(x, [])
	queryFrequency = dict( [ (i, listOfTerms.count(i)) for i in set(listOfTerms) ] )
	userQueries[key] = queryFrequency

saveCompilation(userQueries, filename="termFrequencies.py")


