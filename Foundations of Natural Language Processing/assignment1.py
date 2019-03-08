#!/usr/bin/python
# coding: utf-8

import nltk
import sys

# Import numpy as we will need it to calculate mean and standard deviation
import numpy as np

# Import the Presidential inaugural speeches, Brown and CONLL corpora
# conll2007 is not installed by default
nltk.data.path.append('/group/sgwater/data/nltk_data')
from nltk.corpus import inaugural, brown, conll2007

# directory with special twitter module
sys.path.extend(['/group/ltg/projects/fnlp', '/group/ltg/projects/fnlp/packages_2.6'])

# Import the Twitter corpus and LgramModel
from twitter import xtwc, LgramModel

# Stopword list
from nltk.corpus import stopwords

twitter_file_ids = xtwc.fileids()[11:13]


#################### SECTION A: COMPARING CORPORA ####################

##### Solution for question 1 #####

def get_corpus_tokens(corpus, list_of_files):
    '''Get the tokens from (part of) a corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :type list_of_files: list(file)
    :param list_of_files: files to read from
    :rtype: list(str)
    :return: the tokenised contents of the files'''

    # Construct "corpus_tokens" (a list of all tokens in the corpus)
    corpus_tokens = [w.lower() for w in corpus.words(list_of_files)]

    # Return the list of corpus tokens
    return corpus_tokens

def q1(corpus, list_of_files):
    '''Compute the average word type length from (part of) a corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :type list_of_files: list(str)
    :param list_of_files: names of files to read from
    :rtype: float
    :return: the average word type length over all the files'''

    # Get a list of all tokens in the corpus
    corpus_tokens = get_corpus_tokens(corpus, list_of_files)

    # Construct a list that contains the lengths for each word
    #  type in the document
    type_lengths = [len(w) for w in set(corpus_tokens)]  # already lowercase

    # Find the average word type length
    avg_type_length = np.mean(type_lengths)

    # Return the average word type length of the document
    return avg_type_length

##### Solution for question 2 #####

def q2():
    '''Question: Why might the average type length be greater for
       twitter data?

    :rtype: str
    :return: your answer'''

    return """
    The average word type length greater for the Twitter data. This is because the Twitter corpus contains words where a single letter is repeated for emphasis ("blahhhhhh..." or "jealoussssss..."), URLs, repeated words/patterns ("passeipasseipasseipassei..."), unicode ("\\u767d\\u3072\\u3052\\u3048..."), hashtags, and other word types that are not used in more formal writing. In contrast, the longest inaugural word is "antiphilosophists".
    """


#################### SECTION B: DATA IN THE REAL WORLD ####################

##### Solution for question 3 #####

def q3(corpus, list_of_files, x):
    '''Tabulate and plot the top x most frequently used word types
       and their counts from the specified files in the corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :type list_of_files: list(str)
    :param list_of_files: names of files to read from
    :rtype: list(tuple(string,int))
    :return: top x word types and their counts from the files'''

    # Get a list of all tokens in the corpus
    corpus_tokens = get_corpus_tokens(corpus, list_of_files)

    # Construct a frequency distribution over the lowercased tokens in the document
    fd_doc_types = nltk.FreqDist(corpus_tokens)  # already lowercase

    # Find the top x most frequently used types in the document
    top_types = fd_doc_types.most_common(x)

    # Produce a plot showing the top x types and their frequencies
    fd_doc_types.plot(x)

    return top_types

##### Solution for question 4 #####

def q4(corpus_tokens):
    '''Clean a list of corpus tokens

    :type corpus_tokens: list(str)
    :param corpus_tokens: (lowercased) corpus tokens
    :rtype: list(str)
    :return: cleaned list of corpus tokens'''

    stops = list(stopwords.words("english"))

    # If token is alpha-numeric and NOT in the list of stopwords,
    #  add it to cleaned_tokens
    cleaned_corpus_tokens = [w for w in corpus_tokens if w.isalnum() and w not in stops]

    return cleaned_corpus_tokens

##### Solution for question 5 #####

def q5(cleaned_corpus_tokens, x):
    '''Tabulate and plot the top x most frequently used word types
       and their counts from the corpus tokens

    :type corpus_tokens: list(str)
    :param corpus_tokens: (cleaned) corpus tokens
    :rtype: list(tuple(string,int))
    :return: top x word types and their counts from the files'''

    # Construct a frequency distribution over the lowercased tokens in the document
    fd_doc_types = nltk.FreqDist(cleaned_corpus_tokens)  # already lowercase

    # Find the top x most frequently used types in the document
    top_types = fd_doc_types.most_common(x)

    # Produce a plot showing the top x types and their frequencies
    fd_doc_types.plot(x)

    # Return the top x most frequently used types
    return top_types

##### Solution for question 6 #####

def q6():
    '''Problem: URLs in twitter data

    :rtype: str
    :return: your answer'''

    return """
    One major problem was the existence of common words in languages that are not English. For example, the second and third most common words are "de" and "que", which are Spanish for "of" and "what" (respectively). Other words, like "u", are abbreviations of words that are likely stopwords. There are also word types that may still be related to URLs, such as "eu" and "com". I also see numbers, like "1" and "2", and terms like "n\\xe3o".

    Some solutions to alleviate these problems are: (1) removing Spanish stopwords; (2) removing Unicode (words that start with "\\u" or "\\x"); (3) keeping terms that contain only letters (as opposed to a mixture of letters and numbers); and (4) removing one-letter words (since these are likely abbreviations of stopwords, or are otherwise uninformative or not very interesting). A more complex solution, implemented later in this assignment, is to create a Spanish bigram model, which can be used to discriminate between English and Spanish tweets through the calculation of cross-entropy.
    """


#################### SECTION C: LANGUAGE IDENTIFICATION ####################

##### Solution for question 7 #####

def q7(corpus):
    '''Build a bigram letter language model using LgramModel
       based on the all-alpha subset the entire corpus

    :type corpus: nltk.corpus.CorpusReader
    :param corpus: An NLTK corpus
    :rtype: LgramModel
    :return: A padded letter bigram model based on nltk.model.NgramModel'''

    corpus_tokens = [w.lower() for w in corpus.words() if w.isalpha()]

    bigram_model = LgramModel(2, corpus_tokens, pad_left=True, pad_right=True)

    # Return the letter bigram LM
    return bigram_model

##### Solution for question 8 #####

def q8(file_name,bigram_model):
    '''Using a character bigram model, compute sentence entropies
       for a subset of the tweet corpus, removing all non-alpha tokens and
       tweets with less than 5 all-alpha tokens

    :type file_name: str
    :param file_name: twitter file to process
    :rtype: list(tuple(float,list(str)))
    :return: ordered list of average entropies and tweets'''

    list_of_tweets = xtwc.sents(file_name)

    cleaned_list_of_tweets = []
    for tweet in list_of_tweets:
        cleaned_tweet = [w.lower() for w in tweet if w.isalpha()]
        if len(cleaned_tweet) >= 5:
            cleaned_list_of_tweets.append(cleaned_tweet)
    
    # For each tweet in the cleaned corpus, compute the average word
    #  entropy, and store in a list of tuples of the form: (entropy,tweet)
    list_of_tweets_and_entropies = []
    for tweet in cleaned_list_of_tweets:
        e = np.mean([bigram_model.entropy(w, pad_left=True, pad_right=True, perItem=True) for w in tweet])
        list_of_tweets_and_entropies.append((e, tweet))
            
    
    # Sort the list of (entropy,tweet) tuples by entropy
    list_of_tweets_and_entropies = sorted(list_of_tweets_and_entropies, key=lambda tup: tup[0])

    # Return the sorted list of tuples
    return list_of_tweets_and_entropies

##### Solution for question 9 #####

def q9():
    '''Question: What differentiates the beginning and end of the list
       of tweets and their entropies?

    :rtype: str
    :return: your answer'''

    return """
    The tweets with the best entropy, such as "{and, here, is, proof, the}" and "{and, bailed, he, here, is, man, on, that, the}", contain words with common bigrams. For example, the words "and" and "that" contain the bigrams "an" and "th", which I imagine are very common letter combinations. The tweet "{s, s, s, s, s, s, s, s, s, s}" has low entropy because we padded both sides of the word, and the model recognizes that "s" is a common letter for starting *and* ending words. This also applies to the two words mentioned before ("and" and "that"): many words start with "a" or "t" and end with "d" or "t".

    The worst entropies are simply tweets without English characters: many (if not all) of these tweets are made of Japanese characters (according to the autmatic language detection tool in Google Translate). The bigram model assigns low probabiilites to all of these bigrams since they were not observed in the (English language) data the model was trained on.
    """

##### Solution for question 10 #####

# Output: 
def q10(list_of_tweets_and_entropies):
    '''Compute entropy mean, standard deviation and using them,
       likely non-English tweets in the all-ascii subset of list of tweets
       and their biletter entropies

    :type list_of_tweets_and_entropies: list(tuple(float,list(str)))
    :param list_of_tweets_and_entropies: tweets and their
                                    internal average biletter entropy
    :rtype: tuple(float, float, list(tuple(float,list(str)))
    :return: mean, standard deviation, ascii tweets and entropies,
             not-English tweets and entropies'''

    # Find the "ascii" tweets - those in the lowest-entropy 90%
    #  of list_of_tweets_and_entropies
    threshold = int(len(list_of_tweets_and_entropies) * 0.9)
    list_of_ascii_tweets_and_entropies = list_of_tweets_and_entropies[:threshold]

    # Extract a list of just the entropy values
    list_of_entropies = [tup[0] for tup in list_of_ascii_tweets_and_entropies]

    # Compute the mean of entropy values for "ascii" tweets
    mean = np.mean(list_of_entropies)

    # Compute their standard deviation
    standard_deviation = np.std(list_of_entropies)

    # Get a list of "probably not English" tweets, that is, "ascii"
    # tweets with an entropy greater than (mean + (0.674 * std_dev))
    threshold = mean + (0.674 * standard_deviation)
    list_of_not_English_tweets_and_entropies = [tup for tup in list_of_ascii_tweets_and_entropies if tup[0] > threshold]
    
    # sort...
    list_of_not_English_tweets_and_entropies = sorted(list_of_not_English_tweets_and_entropies, key=lambda tup: tup[0])

    # Return the mean and standard_deviation values and the two lists
    return (mean, standard_deviation,
            list_of_ascii_tweets_and_entropies,
            list_of_not_English_tweets_and_entropies)

##### Solution for question 11 #####

def q11(list_of_files, list_of_not_English_tweets_and_entropies):
    '''Build a padded spanish bigram letter bigram model and use it
       to re-sort the probably-not-English data

    :type list_of_files: list(str)
    :param list_of_files: spanish corpus files
    :type list_of_tweets_and_entropies: list(tuple(float,list(str)))
    :param list_of_tweets_and_entropies: tweets and their
                                    internal average biletter entropy
    :rtype: list(tuple(float,list(str)))
    :return: probably-not-English tweets and _spanish_ entropies'''

    # Build a bigram letter language model using "LgramModel"
    corpus_tokens = [w.lower() for w in conll2007.words(list_of_files) if w.isalpha()]
    bigram_model = LgramModel(2, corpus_tokens, pad_left=True, pad_right=True)

    # Compute the entropy of each of the tweets in list (list_of_not_English_tweets_and_entropies) using the new bigram letter language model
    # list_of_not_English_tweets_and_entropies = [(bigram_model.entropy(tup[1], pad_left=True, pad_right=True, perItem=True), tup[1]) for tup in list_of_not_English_tweets_and_entropies]
    tweets = [tup[1] for tup in list_of_not_English_tweets_and_entropies]
    list_of_not_English_tweets_and_entropies = []
    for tweet in tweets:
        e = np.mean([bigram_model.entropy(w, pad_left=True, pad_right=True, perItem=True) for w in tweet])
        list_of_not_English_tweets_and_entropies.append((e, tweet))

    # Sort the new list of (entropy,tweet) tuples
    list_of_not_English_tweets_and_entropies = sorted(list_of_not_English_tweets_and_entropies, key=lambda tup: tup[0])

    # Return the list of tweets with _new_ entropies, re-sorted
    return list_of_not_English_tweets_and_entropies


##### Answers #####

def ppEandT(eAndTs):
    '''Pretty print a list of entropy+tweet pairs

    :type eAndTs: list(tuple(float,list(str)))
    :param eAndTs: entropies and tweets
    :return: None'''

    for entropy,tweet in eAndTs:
        print (u"%.3f {%s}"%(entropy,", ".join(tweet))).encode("utf-8")

# Uncomment the print statements as you fill in the corresponding functions
def answers():
    # So we can see these during development
    global answer1a, answer1b, answer2, answer3a, answer3b, answer4a, answer4b
    global answer5a, answer5b, answer6, brown_bigram_model, answer8, answer9
    global answer10, answer11
    ### Question 1
    print "*** Question 1 ***"
    answer1a = q1(inaugural,inaugural.fileids())
    print "Average token length for inaugural corpus: %.2f"%answer1a
    answer1b = q1(xtwc,twitter_file_ids)
    print "Average token length for twitter corpus: %.2f"%answer1b
    ### Question 2
    print "*** Question 2 ***"
    answer2 = q2()
    print answer2
    ### Question 3
    print "*** Question 3 ***"
    print "Most common 50 types for the inaugural corpus:"
    answer3a = q3(inaugural,inaugural.fileids(),50)
    print answer3a
    print "Most common 50 types for the twitter corpus:"
    answer3b = q3(xtwc,twitter_file_ids,50)
    print answer3b
    ### Question 4
    print "*** Question 4 ***"
    corpus_tokens = get_corpus_tokens(inaugural,inaugural.fileids())
    answer4a = q4(corpus_tokens)
    print "Inaugural Speeches:"
    print "Number of tokens in original corpus: %s"%len(corpus_tokens)
    print "Number of tokens in cleaned corpus: %s"%len(answer4a)
    print "First 100 tokens in cleaned corpus:"
    print answer4a[:100]
    print "-----"
    corpus_tokens = get_corpus_tokens(xtwc,twitter_file_ids)
    answer4b = q4(corpus_tokens)
    print "Twitter:"
    print "Number of tokens in original corpus: %s"%len(corpus_tokens)
    print "Number of tokens in cleaned corpus: %s"%len(answer4b)
    print "First 100 tokens in cleaned corpus:"
    print answer4b[:100]
    ### Question 5
    print "*** Question 5 ***"
    print "Most common 50 types for the cleaned inaugural corpus:"
    answer5a = q5(answer4a, 50)
    print answer5a
    print "Most common 50 types for the cleaned twitter corpus:"
    answer5b = q5(answer4b, 50)
    print answer5b
    ### Question 6
    print "*** Question 6 ***"
    answer6 = q6()
    print answer6
    ### Question 7
    print "*** Question 7: building brown bigram letter model ***"
    brown_bigram_model = q7(brown)
    ### Question 8
    print "*** Question 8 ***"
    answer8 = q8("20100128.txt",brown_bigram_model)
    print "Best 10 entropies:"
    ppEandT(answer8[:10])
    print "Worst 10 entropies:"
    ppEandT(answer8[-10:])
    ### Question 9
    print "*** Question 9 ***"
    answer9 = q9()
    print answer9
    ### Question 10
    print "*** Question 10 ***"
    answer10 = q10(answer8)
    print "Mean: %s"%answer10[0]
    print "Standard Deviation: %s"%answer10[1]
    print "'Ascii' tweets: Best 10 entropies:"
    ppEandT(answer10[2][:10])
    print "'Ascii' tweets: Worst 10 entropies:"
    ppEandT(answer10[2][-10:])
    print "Probably not English tweets: Best 10 entropies:"
    ppEandT(answer10[3][:10])
    print "Probably not English tweets: Worst 10 entropies:"
    ppEandT(answer10[3][-10:])
    ### Question 11
    print "*** Question 11 ***"
    list_of_not_English_tweets_and_entropies = answer10[3]
    answer11 = q11(["esp.test","esp.train"],list_of_not_English_tweets_and_entropies)
    print "Best 10 entropies:"
    ppEandT(answer11[:10])
    print "Worst 10 entropies:"
    ppEandT(answer11[-10:])

if __name__ == '__main__':
    answers()