from userQueries import dic as userQueries

# converts a dictionary with keys:userID value:listOfQueries
# to a dictionary with keys:userID value:[(term, termFrequency)]
def saveCompilation(dic, filename="compiled.py"):
	"""
		Save dictionary to disc.
	"""
	f = open(filename,'w')
	f.write("dic = " + str(dic))
	f.close()
	return


for key in userQueries.keys():
	queryList = userQueries[key]
	x = [s.split(" ") for s in queryList]
	listOfTerms = sum(x, [])
	queryFrequency = dict( [ (i, listOfTerms.count(i)) for i in set(listOfTerms) ] )
	userQueries[key] = queryFrequency

saveCompilation(userQueries, filename="termFrequencies.py")


