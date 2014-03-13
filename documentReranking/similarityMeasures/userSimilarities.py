import userLM as user
import userLMExtended as userExtended

def sim(A, B, minCommonTerms):
	similarityScore = 0

	# check if they have minCommonTerms terms in common
	# if not, return 0
	if A.termsInCommon(B) > minCommonTerms:
		# calculate the similarity
		for query in A.queries:
			similarityScore += B.p_q_u(query)
		similarityScore = similarityScore / A.numberOfQueries
	else:
		similarityScore = -9999

	return  similarityScore

def mutualSim(A, B, minCommonTerms):
	similarityScoreAB = 0
	similarityScoreBA = 0

	# check if they have minCommonTerms terms in common
	# if not, return 0
	if A.termsInCommon(B) > minCommonTerms:
		# calculate the similarity A -> B
		for query in A.queries:
			# !! not very efficient since users are expected to have many duplicate queries... 
			similarityScoreAB += B.p_q_u(query)
		similarityScoreAB = similarityScoreAB / A.numberOfQueries

		for query in B.queries:
			similarityScoreBA += A.p_q_u(query)
		similarityScoreBA = similarityScoreBA / B.numberOfQueries
	
	else:
		similarityScoreAB = -9999
		similarityScoreBA = -9999

	return similarityScoreAB + similarityScoreBA
