# documentReranking
# Created by Anouk Visser (27-02-2014)

import user as user

numberOfSimilarUsers = 5
minTermsInCommon = 5
userID = 'UID48'
# so we have a user
u = user.User(userID)

for query in u.queries:
	print query
print u.getMostSimilarUsers(numberOfSimilarUsers, minTermsInCommon)




