# coding: utf-8

from question1 import *
import json

def equalize_full_lens(v1, v2):  
    
    diff = len(v2) - len(v1)
    if diff > 0:
        v1.extend([0] * diff)
    else:
        v2.extend([0] * -diff)
        
    return v1, v2

def write_file(model, csType, filename, vocab_unicode):
    print '=' * 25
    print filename
    print '=' * 25
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'w') as f:
        for jsonSentence in sents:
            target_word = jsonSentence['target_word']
            ID = jsonSentence['id']
            best_sub = best_substitute(jsonSentence, thesaurus, word2id, model, frequencyVectors, csType, vocab_unicode=vocab_unicode)
            if best_sub is None:
                best_sub = ''
            output = '{} {} :: {}'.format(target_word, ID, re.sub('\..*', '', best_sub))
            print output  # can comment this out if don't want printing to console
            f.write(output + '\n')

'''
helper class to load a thesaurus from disk
input: thesaurusFile, file on disk containing a thesaurus of substitution words for targets
output: the thesaurus, as a mapping from target words to lists of substitution words
'''
def load_thesaurus(thesaurusFile):
	thesaurus = {}
	with open(thesaurusFile) as inFile:
		for line in inFile.readlines():
			word, subs = line.strip().split("\t")
			thesaurus[word] = subs.split(" ")
	return thesaurus

'''
(a) function addition for adding 2 vectors
input: vector1
input: vector2
output: addVector, the resulting vector when adding vector1 and vector2
'''
def addition(vector1, vector2):
    
    # Make sure inputs are full.
    vector1 = sparse2full(vector1, None)
    vector2 = sparse2full(vector2, None)
    vector1, vector2 = equalize_full_lens(vector1, vector2)
    
    added = [x + y for x, y in zip(vector1, vector2)]
    
    return full2sparse(added)

'''
(a) function multiplication for multiplying 2 vectors
input: vector1
input: vector2
output: mulVector, the resulting vector when multiplying vector1 and vector2
'''
def multiplication(vector1, vector2):
    
    # Make sure inputs are full.
    vector1 = sparse2full(vector1, None)
    vector2 = sparse2full(vector2, None)
    vector1, vector2 = equalize_full_lens(vector1, vector2)
    
    multiplied = [x * y for x, y in zip(vector1, vector2)]
    
    return full2sparse(multiplied)

'''
(d) function prob_z_given_w to get probability of LDA topic z, given target word w
input: ldaModel
input: topicID as an integer
input: wordVector in frequency space
output: probability of the topic with topicID in the ldaModel, given the wordVector
'''
def prob_z_given_w(ldaModel, topicID, wordVector):
    topic_probs = ldaModel.get_document_topics(wordVector, minimum_probability=0.)
    try:
        prob_topic = [tup[1] for tup in topic_probs if tup[0]==topicID][0]
    except IndexError:
        prob_topic = 0.
    return prob_topic

'''
(d) function prob_w_given_z to get probability of target word w, given LDA topic z
input: ldaModel
input: targetWord as a string
input: topicID as an integer
output: probability of the targetWord, given the topic with topicID in the ldaModel
'''
def prob_w_given_z(ldaModel, targetWord, topicID):
    words = ldaModel.show_topic(topicID, 20000)  # 20000 gives all
    try:
        word_prob = [tup[1] for tup in words if gensim.utils.any2unicode(tup[0])==targetWord][0]
    except IndexError:
        word_prob = 0.
    return word_prob

'''
(f) get the best substitution word in a given sentence, according to a given model (tf-idf, word2vec, LDA) and type (addition, multiplication, lda)
input: jsonSentence, a string in json format
input: thesaurus, mapping from target words to candidate substitution words
input: word2id, mapping from vocabulary words to word IDs
input: model, a vector space, Word2Vec or LDA model
input: frequency vectors, original frequency vectors (for querying LDA model)
input: csType, a string indicating the method of calculating context sensitive vectors: "addition", "multiplication", or "lda"
output: the best substitution word for the jsonSentence in the given model, using the given csType
'''
def best_substitute(jsonSentence, thesaurus, word2id, model, frequencyVectors, csType, vocab_unicode=None):
    
    window = 5
    
    target_word = jsonSentence['target_word'].lower()
    target_position = int(jsonSentence['target_position'])
    sentence = [w.lower() for w in jsonSentence['sentence'].split(' ')]
    words = thesaurus[target_word]
    
    if vocab_unicode is None:
        vocab_unicode = [gensim.utils.any2unicode(w) for w in word2id.keys()]

    # Dicts are necessary for a reasonable run time given how this had been coded.
    # Otherwise the prob_z_given_w and prob_w_given_z calculations each take
    # 0.02-0.15 seconds, and they must be performed *many* times (200 sentences
    # * ~5 thesaurus words * <=10 context words * 100 topics = ~1 million).
    prob_z_given_w_dict = {}
    prob_w_given_z_dict = {}
    
    # (b) use addition to get context sensitive vectors
    if csType == "addition":
        def context_sensitive(v1, v2, context=None, target_word=None):
            return addition(v1, v2)

    # (c) use multiplication to get context sensitive vectors
    elif csType == "multiplication":
        def context_sensitive(v1, v2, context=None, target_word=None):
            return multiplication(v1, v2)
    
    # (d) use LDA to get context sensitive vectors
    elif csType == "lda":
        topicIds = [lst[0] for lst in model.show_topics(-1)]  # -1 gives all topics
        def context_sensitive(t, c, context, target_word):
            cs_vector = []
            for topicId in topicIds:
                # Get prob_z_given_w.
                try:
                    prob_z_given_w_value = prob_z_given_w_dict[(topicId, target_word)]
                except KeyError:
                    prob_z_given_w_value = prob_z_given_w(model, topicId, frequencyVectors[word2id[target_word]])
                    prob_z_given_w_dict[(topicId, target_word)] = prob_z_given_w_value
                # Get prob_w_given_z.
                try:
                    prob_w_given_z_value = prob_w_given_z_dict[(topicId, context)]
                except KeyError:
                    prob_w_given_z_value = prob_w_given_z(model, context, topicId)
                    prob_w_given_z_dict[(topicId, context)] = prob_w_given_z_value
                # Add their product to vector.
                cs_vector.append(prob_z_given_w_value * prob_w_given_z_value)
            return cs_vector
    
    contexts = []
    for i in range(target_position - window, target_position + window + 1):
        if i != target_position and i >= 0 and i < len(sentence) and sentence[i] in vocab_unicode:
            contexts.append(sentence[i])
    if not contexts:
        return None  # fail to predict if no context words
    
    def get_vector(model, target_word):
        if csType == 'lda':
            vector = frequencyVectors[word2id[target_word]]  # no need for model
        else:
            if type(model) == gensim.models.word2vec.Word2Vec:
                vector = list(model[target_word])
            else:
                vector = model[word2id[target_word]]
        return vector
    
    best_word = None
    best_score = 0.
    t = get_vector(model, target_word)
    for word in words:
        w = get_vector(model, word)
        score = 0.
        for context in contexts:
            try:
                c = get_vector(model, context)
            except KeyError:  # e.g., u'continually.r': word not in vocab
                continue
            if not c:  # context word has no vector
                continue
            tc = context_sensitive(t, c, context, target_word)
            if not tc:  # sometimes multiplication returns []; then ignore context word
                continue
            score += cosine_similarity(w, tc)
        if score > best_score:
            best_score = score
            best_word = word
    
    return best_word

if __name__ == "__main__":
	import sys
	
	part = sys.argv[1]
	
	# this can give you an indication whether part a (vector addition and multiplication) works.
	if part == "a":
		print("(a): vector addition and multiplication")
		v1, v2, v3 , v4 = [(0,1), (2,1), (4,2)], [(0,1), (1,2), (4,1)], [1, 0, 1, 0, 2], [1, 2, 0, 0, 1]
		try:
			if not set(addition(v1, v2)) == set([(0, 2), (2, 1), (4, 3), (1, 2)]):
				print("\tError: sparse addition returned wrong result")
			else:
				print("\tPass: sparse addition")
		except Exception as e:
			print("\tError: exception raised in sparse addition")
			print(e)
		try:
			if not set(multiplication(v1, v2)) == set([(0,1), (4,2)]):
				print("\tError: sparse multiplication returned wrong result")
			else:
				print("\tPass: sparse multiplication")
		except Exception as e:
			print("\tError: exception raised in sparse multiplication")
			print(e)
		try:
			addition(v3,v4)
			print("\tPass: full addition")
		except Exception as e:
			print("\tError: exception raised in full addition")
			print(e)
		try:
			multiplication(v3,v4)
			print("\tPass: full multiplication")
		except Exception as e:
			print("\tError: exception raised in full addition")
			print(e)
	
	# you may complete this to get answers for part b (best substitution words with tf-idf and word2vec, using addition)
	if part == "b":
		print("(b) using addition to calculate best substitution words")
		print "Did this in Jupyter notebook. See write_file function for how this was implemented."
	
	# you may complete this to get answers for part c (best substitution words with tf-idf and word2vec, using multiplication)
	if part == "c":
		print("(c) using multiplication to calculate best substitution words")
		print "Did this in Jupyter notebook. See write_file function for how this was implemented."
	
	# this can give you an indication whether your part d1 (P(Z|w) and P(w|Z)) works
	if part == "d":
		print("(d): calculating P(Z|w) and P(w|Z)")
		print("\tloading corpus")
		id2word,word2id,vectors=load_corpus(sys.argv[2], sys.argv[3])
		print("\tloading LDA model")
		ldaModel = gensim.models.ldamodel.LdaModel.load("run/lda_model")
		houseTopic = ldaModel[vectors[word2id["house.n"]]][0][0]
		try:
			if prob_z_given_w(ldaModel, houseTopic, vectors[word2id["house.n"]]) > 0.0:
				print("\tPass: P(Z|w)")
			else:
				print("\tFail: P(Z|w)")
		except Exception as e:
			print("\tError: exception during P(Z|w)")
			print(e)
		try:
			if prob_w_given_z(ldaModel, "house.n", houseTopic) > 0.0:
				print("\tPass: P(w|Z)")
			else:
				print("\tFail: P(w|Z)")
		except Exception as e:
			print("\tError: exception during P(w|Z)")
			print(e)
	
	# you may complete this to get answers for part d2 (best substitution words with LDA)
	if part == "e":
		print("(e): using LDA to calculate best substitution words")
		print "Did this in Jupyter notebook. See write_file function for how this was implemented."
