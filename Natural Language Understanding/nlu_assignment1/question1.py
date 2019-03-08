# coding: utf-8

import gensim
import math
from copy import copy
import re

# Define helper functions.
def is_sparse(vector):
    return not vector or isinstance(vector[0], tuple)

def sparse2full(sparse, length=5000):
    sparse = list(sparse)
    if not is_sparse(sparse):
        return sparse
    if length is None:
        length = max([tup[0] for tup in sparse]) + 1
    full = [0] * length
    for tup in sparse:
        full[tup[0]] = tup[1]
    return full

def full2sparse(full):
    if is_sparse(full):
        return full
    sparse = [(c, n) for c, n in enumerate(full) if n > 0]
    return sparse

def make_test_word_list():
    w1 = (80, 'house.n')
    w2 = (143, 'home.n')
    w3 = (12, 'time.n')
    return [w1, w2, w3]

def test_words(vectors, words_to_test = make_test_word_list()):
    for i, tup1 in enumerate(words_to_test):
        for j, tup2 in enumerate(words_to_test):
            if j > i:
                id1, str1 = tup1
                id2, str2 = tup2
                sim = cosine_similarity(vectors[id1], vectors[id2])
                print 'similarity between {} and {}: {}'.format(str1, str2, sim)
    return None

'''
(f) helper class, do not modify.
provides an iterator over sentences in the provided BNC corpus
input: corpus path to the BNC corpus
input: n, number of sentences to retrieve (optional, standard -1: all)
'''
class BncSentences:
	def __init__(self, corpus, n=-1):
		self.corpus = corpus
		self.n = n
	
	def __iter__(self):
		n = self.n
		ret = []
		for line in open(self.corpus):
			line = line.strip().lower()
			if line.startswith("<s "):
				ret = []
			elif line.strip() == "</s>":
				if n > 0:
					n -= 1
				if n == 0:
					break
				yield copy(ret)
			else:
				parts = line.split("\t")
				if len(parts) == 3:
					word = parts[-1]
					idx = word.rfind("-")
					word, pos = word[:idx], word[idx+1:]
					if word in ['thus', 'late', 'often', 'only', 'usually', 'however', 'lately', 'absolutely', 'hardly', 'fairly', 'near', 'similarly', 'sooner', 'there', 'seriously', 'consequently', 'recently', 'across', 'softly', 'together', 'obviously', 'slightly', 'instantly', 'well', 'therefore', 'solely', 'intimately', 'correctly', 'roughly', 'truly', 'briefly', 'clearly', 'effectively', 'sometimes', 'everywhere', 'somewhat', 'behind', 'heavily', 'indeed', 'sufficiently', 'abruptly', 'narrowly', 'frequently', 'lightly', 'likewise', 'utterly', 'now', 'previously', 'barely', 'seemingly', 'along', 'equally', 'so', 'below', 'apart', 'rather', 'already', 'underneath', 'currently', 'here', 'quite', 'regularly', 'elsewhere', 'today', 'still', 'continuously', 'yet', 'virtually', 'of', 'exclusively', 'right', 'forward', 'properly', 'instead', 'this', 'immediately', 'nowadays', 'around', 'perfectly', 'reasonably', 'much', 'nevertheless', 'intently', 'forth', 'significantly', 'merely', 'repeatedly', 'soon', 'closely', 'shortly', 'accordingly', 'badly', 'formerly', 'alternatively', 'hard', 'hence', 'nearly', 'honestly', 'wholly', 'commonly', 'completely', 'perhaps', 'carefully', 'possibly', 'quietly', 'out', 'really', 'close', 'strongly', 'fiercely', 'strictly', 'jointly', 'earlier', 'round', 'as', 'definitely', 'purely', 'little', 'initially', 'ahead', 'occasionally', 'totally', 'severely', 'maybe', 'evidently', 'before', 'later', 'apparently', 'actually', 'onwards', 'almost', 'tightly', 'practically', 'extremely', 'just', 'accurately', 'entirely', 'faintly', 'away', 'since', 'genuinely', 'neatly', 'directly', 'potentially', 'presently', 'approximately', 'very', 'forwards', 'aside', 'that', 'hitherto', 'beforehand', 'fully', 'firmly', 'generally', 'altogether', 'gently', 'about', 'exceptionally', 'exactly', 'straight', 'on', 'off', 'ever', 'also', 'sharply', 'violently', 'undoubtedly', 'more', 'over', 'quickly', 'plainly', 'necessarily']:
						pos = "r"
					if pos == "j":
						pos = "a"
					ret.append(gensim.utils.any2unicode(word + "." + pos))

'''
(a) function load_corpus to read a corpus from disk
input: vocabFile containing vocabulary
input: contextFile containing word contexts
output: id2word mapping word IDs to words
output: word2id mapping words to word IDs
output: vectors for the corpus, as a list of sparse vectors
'''
def load_corpus(vocabFile, contextFile):
    id2word = {}
    word2id = {}
    vectors = []

    with open(vocabFile, 'r') as f:
        vocab = f.read().splitlines()
    
    with open(contextFile, 'r') as f:
        context = f.read().splitlines()
    
    for big_string in context:
        tups = []
        if big_string[0] != '0':  # words without context words have a string that starts with '0'
            cns = big_string.split(' ')[1:]
            for cn in cns:
                c, n = cn.split(':')
                tups.append((int(c), int(n)))
        vectors.append(tups)
    
    for ID, word in enumerate(vocab):
        id2word[ID] = word
        word2id[word] = ID

    return id2word, word2id, vectors

'''
(b) function cosine_similarity to calculate similarity between 2 vectors
input: vector1
input: vector2
output: cosine similarity between vector1 and vector2 as a real number
'''
def cosine_similarity(vector1, vector2):
    
    # Make sure inputs are full.
    vector1 = sparse2full(vector1)
    vector2 = sparse2full(vector2)
    
    def dot_prod(v1, v2):
        return sum([x * y for x, y in zip(v1, v2)])

    num = dot_prod(vector1, vector2)
    denom = dot_prod(vector1, vector1)**0.5 * dot_prod(vector2, vector2)**0.5
    
    return num / float(denom)

'''
(d) function tf_idf to turn existing frequency-based vector model into tf-idf-based vector model
input: freqVectors, a list of frequency-based vectors
output: tfIdfVectors, a list of tf-idf-based vectors
'''
def tf_idf(freqVectors):
    
    tfIdfVectors = []
    N = len(freqVectors)
    
    # Convert to sparse vectors.
    freqVectors = [full2sparse(fv) for fv in freqVectors]
    
    # Define function that calculates TF-IDF for a single term.
    def tf_idf_word(tf, N, df):
        return (1 + math.log(tf, 2)) * (1 + math.log(N / float(df), 2))

    # Generate dictionary df (document frequency of term). Do this by iterating over documents.
    df_dict = {}
    for fv in freqVectors:
        for term, _ in fv:  # don't need frequency
            df_dict[term] = df_dict.get(term, 0) + 1
        
    for fv in freqVectors:
        tfIdfVectors.append([(term, tf_idf_word(tf, N, df_dict[term])) for term, tf in fv])

    return tfIdfVectors

'''
(f) function word2vec to build a word2vec vector model with 100 dimensions and window size 5
'''
def word2vec(corpus, learningRate, downsampleRate, negSampling):
    
    model = gensim.models.word2vec.Word2Vec(sentences=corpus,
                                            alpha=learningRate,
                                            sample=downsampleRate,
                                            negative=negSampling,
                                            size=100,
                                            window=5)
    
    return model

'''
(h) function lda to build an LDA model with 100 topics from a frequency vector space
input: vectors
input: wordMapping mapping from word IDs to words
output: an LDA topic model with 100 topics, using the frequency vectors
'''
def lda(vectors, wordMapping):
    model = gensim.models.ldamodel.LdaModel(vectors, id2word=wordMapping, num_topics=100, passes=10, update_every=0)
    return model

'''
(j) function get_topic_words, to get words in a given LDA topic
input: ldaModel, pre-trained Gensim LDA model
input: topicID, ID of the topic for which to get topic words
input: wordMapping, mapping from words to IDs (optional)
'''
def get_topic_words(ldaModel, topicID, topn=10):
    return ldaModel.show_topic(topicID, topn)

if __name__ == '__main__':
	import sys
	
	part = sys.argv[1].lower()
	
	# these are indices for house, home and time in the data. Don't change.
	house_noun = 80
	home_noun = 143
	time_noun = 12
	
	# this can give you an indication whether part a (loading a corpus) works.
	# not guaranteed that everything works.
	if part == "a":
		print("(a): load corpus")
		try:
			id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
			if not id2word:
				print("\tError: id2word is None or empty")
				exit()
			if not word2id:
				print("\tError: id2word is None or empty")
				exit()
			if not vectors:
				print("\tError: id2word is None or empty")
				exit()
			print("\tPass: load corpus from file")
		except Exception as e:
			print("\tError: could not load corpus from disk")
			print(e)
		
		try:
			if not id2word[house_noun] == "house.n" or not id2word[home_noun] == "home.n" or not id2word[time_noun] == "time.n":
				print("\tError: id2word fails to retrive correct words for ids")
			else:
				print("\tPass: id2word")
		except Exception:
			print("\tError: Exception in id2word")
			print(e)
		
		try:
			if not word2id["house.n"] == house_noun or not word2id["home.n"] == home_noun or not word2id["time.n"] == time_noun:
				print("\tError: word2id fails to retrive correct ids for words")
			else:
				print("\tPass: word2id")
		except Exception:
			print("\tError: Exception in word2id")
			print(e)
	
	# this can give you an indication whether part b (cosine similarity) works.
	# these are very simple dummy vectors, no guarantee it works for our actual vectors.
	if part == "b":
		import numpy
		print("(b): cosine similarity")
		try:
			cos = cosine_similarity([(0,1), (2,1), (4,2)], [(0,1), (1,2), (4,1)])
			if not numpy.isclose(0.5, cos):
				print("\tError: sparse expected similarity is 0.5, was {0}".format(cos))
			else:
				print("\tPass: sparse vector similarity")
		except Exception:
			print("\tError: failed for sparse vector")
		try:
			cos = cosine_similarity([1, 0, 1, 0, 2], [1, 2, 0, 0, 1])
			if not numpy.isclose(0.5, cos):
				print("\tError: full expected similarity is 0.5, was {0}".format(cos))
			else:
				print("\tPass: full vector similarity")
		except Exception:
			print("\tError: failed for full vector")

	# you may complete this part to get answers for part c (similarity in frequency space)
	if part == "c":
		print("(c) similarity of house, home and time in frequency space")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		test_words(vectors)
	
	# this gives you an indication whether your conversion into tf-idf space works.
	# this does not test for vector values in tf-idf space, hence can't tell you whether tf-idf has been implemented correctly
	if part == "d":
		print("(d) converting to tf-idf space")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		try:
			tfIdfSpace = tf_idf(vectors)
			if not len(vectors) == len(tfIdfSpace):
				print("\tError: tf-idf space does not correspond to original vector space")
			else:
				print("\tPass: converted to tf-idf space")
		except Exception as e:
			print("\tError: could not convert to tf-idf space")
			print(e)
	
	# you may complete this part to get answers for part e (similarity in tf-idf space)
	if part == "e":
		print("(e) similarity of house, home and time in tf-idf space")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		tf_idf_vectors = tf_idf(vectors)
		test_words(tf_idf_vectors)
	
	# you may complete this part for the first part of f (estimating best learning rate, sample rate and negative samplings)
	if part == "f1":
		print("(f1) word2vec, estimating best learning rate, sample rate, negative sampling")
		print "Did this in Jupyter notebook."
	
	# you may complete this part for the second part of f (training and saving the actual word2vec model)
	if part == "f2":
		import logging
		# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		# print("(f2) word2vec, building full model with best parameters. May take a while.")
		# learningRate = 0.05
		# downsampleRate = 0.001
		# negSampling = 10
		# model = word2vec(BncSentences('/Users/sipola/Desktop/bnc.vert'),
		#                  learningRate,
		#                  downsampleRate,
		#                  negSampling)
		# model.save('run/word2vec_model_dummy')
	
	# you may complete this part to get answers for part g (similarity in your word2vec model)
	if part == "g":
		print("(g): word2vec based similarity")
		model = gensim.models.word2vec.Word2Vec.load('run/word2vec_model')
		words_to_test = ['house.n', 'home.n', 'time.n']
		for i, w1 in enumerate(words_to_test):
		    for j, w2 in enumerate(words_to_test):
		        if j > i:
		            sim = model.similarity(w1, w2)
		            print 'similarity between {} and {}: {}'.format(w1, w2, sim)
	
	# you may complete this for part h (training and saving the LDA model)
	if part == "h":
		import logging
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		print("(h) LDA model")
		model = lda(vectors, id2word)
		model.save('run/lda_model_dummy')
	
	# you may complete this part to get answers for part i (similarity in your LDA model)
	if part == "i":
		print("(i): lda-based similarity")
		id2word, word2id, vectors = load_corpus(sys.argv[2], sys.argv[3])
		model = gensim.models.ldamodel.LdaModel.load('run/lda_model')
		words_to_test = ['house.n', 'home.n', 'time.n']
		for i, w1 in enumerate(words_to_test):
		    for j, w2 in enumerate(words_to_test):
		        if j > i:
		            v1 = model[vectors[word2id[w1]]]
		            v2 = model[vectors[word2id[w2]]]
		            sim = cosine_similarity(v1, v2)
		            print 'similarity between {} and {}: {}'.format(w1, w2, sim)

	# you may complete this part to get answers for part j (topic words in your LDA model)
	if part == "j":
		print("(j) get topics from LDA model")
		model = gensim.models.ldamodel.LdaModel.load('run/lda_model')
		for topic_id in range(100):
		    wp = get_topic_words(model, topic_id)
		    word_str = ', '.join([re.sub('\..*', '', tup[0]) for tup in wp])
		    print 'Topic {}: {}'.format(topic_id, word_str)
