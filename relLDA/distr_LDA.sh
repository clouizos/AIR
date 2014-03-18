export PYRO_SERIALIZERS_ACCEPTED=pickle
export PYRO_SERIALIZER=pickle

# Pyro’s name server
python2 -m Pyro4.naming -n 0.0.0.0 &

# worker script per node
python2 -m gensim.models.lda_worker &
python2 -m gensim.models.lda_worker &

# job scheduler in charge of worker synchronization
python2 -m gensim.models.lda_dispatcher &

# run distributed LDA
python2 relLDA.py