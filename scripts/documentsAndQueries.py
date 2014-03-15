import pickle 

from user_specific_positive_negative_examples_dic_strict import dictionary as docs

def add(dic, key, value):
	if key in dic:
		theSet = dic[key]
		theSet.add(value)
		dic[key] = theSet
	else:
		dic[key] = set([value])
	
	return dic

dcq = dict()
for user in docs.keys():
	for trip in docs[user]:
		query = trip[0]
		clicks = trip[1]
		for c in clicks:
			dcq = add(dcq, c, query)


print len(dcq.keys()), "documents were clicked"		

avg = 0
for doc in dcq.keys():
	avg += len(dcq[doc])

avg = avg / float(len(dcq.keys()))	


print "Every document has on average ", avg, " query "

pickle.dump(dcq, open('clickedDocumentsAndQueries', 'wb'))

