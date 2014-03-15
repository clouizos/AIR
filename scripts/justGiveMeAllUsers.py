import pickle
strictness = 'unstrict'
uq = pickle.load(open('userQueries_'+strictness, 'rb'))

users = dict()
users['users'] = set()
for user in uq:
	users['users'].add(user)

pickle.dump(users, open('users_'+strictness, 'wb'))

  
