# here I will define some query similartiy measures
import pickle
print "Loading queries..."
queries = pickle.load(open('../../../queries', 'rb'))['queries']

# Got this from http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
# Should take a closer look at it, looks good at first sight
def levenshtein(s1, s2):
	if len(s1) < len(s2):
		return levenshtein(s2, s1)

	# len(s1) >= len(s2)
	if len(s2) == 0:
		return len(s1)

	previous_row = xrange(len(s2) + 1)
	for i, c1 in enumerate(s1):
		current_row = [i + 1]
		for j, c2 in enumerate(s2):
			insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
			deletions = current_row[j] + 1       # than s2
			substitutions = previous_row[j] + (c1 != c2)
			current_row.append(min(insertions, deletions, substitutions))
		previous_row = current_row

	return previous_row[-1]

# will return the list of clicked documents the queries have in common
# this can be seen as 'query intent', as the documents were relevant to the query
# and fullfill its intent
def clicksOnDocument():

	return 0

def test():
	for q1 in queries:
		bestMatch = 0
		bestScore  = 100000
		for q2 in queries:
			if q1 != q2:
				score = levenshtein(q1, q2)
				if score < bestScore:
					bestMatch = q2
					bestScore = score
		print "Evaluated q1: ", q1
		print "Best match: ", bestMatch, " with score: ", bestScore, "\n"


test()