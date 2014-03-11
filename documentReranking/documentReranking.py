# documentReranking
# Created by Anouk Visser (27-02-2014)

import user as user

numberOfSimilarUsers = 5
minTermsInCommon = 5
userID = 'UID48'
u = user.User(userID)
print u.getMostSimilarUsers(numberOfSimilarUsers, minTermsInCommon)




