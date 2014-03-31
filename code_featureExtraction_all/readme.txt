fetching basic features:
  esinfo.py pickles a dictionary with ES features (saves featuredict_es.dat & helper dict globalFacetDictPy.py)
  LanguageModels.py can be used to get scores for 3 different Statistical LM's, monogram, all doc. content
  tfidf_bm25_features.py can eb used to get scores for tf-idf&bm25, monogram&bigram, 4 doc. fields seperately
  
converting to vector space:
  vectors.py can make vectors of features using the (data from the) above three scripts, and save them to dicts vectorDictx.py;
    additionally, it can split featuredict_es.dat from esinfo.py first into multiple smaller dicts (to aid with loading time)
    
converting to dpRank/Divide&Conquer format:
  vectorToDp.py converts the dicts from vectors.py into a shelved data structure directly compatible with dpRank
  vectorToSVM.py converts the dicts from vectors.py into a shelved data structure directly compatible with the Divide&Conquer algorithm
  
reducing features: 
  reduceFeaturesAutomatic.py reduces the vector space features resulting from vectorToDp.py automatically.
