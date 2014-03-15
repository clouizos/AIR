import pickle
strictness = 'unstrict'

info = pickle.load(open('../user_specific_positive_negative_examples_dic_'+strictness, 'rb'))

documentClicks = dict()
for user in info:
	userInfo = info[user]
	for infoTriplet in userInfo:
		clicks = infoTriplet[1]
		for doc in clicks:
			if doc in documentClicks:
				documentClicks[doc] += 1
			else:
				documentClicks[doc] = 1

pickle.dump(documentClicks, open('numberOfDocumentClicks_'+strictness, 'wb'))
