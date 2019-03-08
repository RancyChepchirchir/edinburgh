import nltk, cky

# Import nltk.tree and nltk.CFG
from nltk import tree, CFG

from cky import CKY, tokenise

from pprint import pprint as pp

import lab4_fix
from lab4_fix import parse_grammar

from nltk.app import rdparser_app as rd

def app(grammar,sent="we sleep"):
    """
    Create a recursive descent parser demo, using a simple grammar and
    text.
    """
    rd.RecursiveDescentApp(grammar, sent.split()).mainloop()

#################### EXERCISE 1 ####################

# Solution for exercise 1
# Input: none
# Output: none
def test1():
    # Load the CFG grammar
    grammar = parse_grammar("""
        S -> NP VP
        PP -> P NP
        NP -> Det N | NP PP | 'I'
        VP -> V NP | VP PP
        Det -> 'an' | 'my'
        N -> 'elephant' | 'pyjamas'
        V -> 'shot'
        P -> 'in'
        """)

    # Print out the grammar rules, one line at a time
    # pp is a method in the "parse" module
    print "Grammar rules:"
    pp(grammar.productions())

    # Step through the parse (using a Recursive Descent / Top-down parser)
    print "Running interactive Recursive-Descent parser..."
    print "  Close the window to end execution"
    app(grammar)
    

### Uncomment to test exercise 1
### Look in the terminal window to see the list of grammar rules
### In the pop-up window, do the following:
###    Click on "Edit" in the menu bar and select "Edit Text"
###    Enter "I shot an elephant in my pyjamas" in the text box
###    Click on "OK"
###    Click the "Autostep" button and watch the parser step through
###    Do you notice a problem?
###    Click the "Autostep" button again to freeze the animation
#test1()




#################### EXERCISE 2 ####################

# A somewhat more detailed grammar for Exercise 2
grammar2=parse_grammar([
"S -> Sdecl '.' | Simp '.' | Sq '?' ", # Take advantage of the punctuation
"Sdecl -> NP VP",
"Simp -> VP",
"Sq -> Sqyn | Swhadv",
"Sqyn -> Mod Sdecl | Aux Sdecl", # Re-use where we can
"Swhadv -> WhAdv Sqyn", # likewise
"Sc -> Subconj Sdecl", # For "that S"
"NP -> PropN | Pro | NP0 ", # NP that allows no modification
"NP0 -> NP1 | NP0 PP", # NP with (multiple) PP attachment
"NP1 -> Det N2sc | Det N2mp | N2mp | Sc", # Common nouns and sentential complements
"N2sc -> Adj N2sc | Nsc | N3 Nsc", # Adjectival mod, head noun for singular
                                   #  count nouns
"N2mp -> Adj N2mp | Nmp | N3 Nmp", # Adjectival mod, head noun for plural
                                   #  count nouns or mass nouns
"N3 -> N | N3 N", # Noun-noun compounds
"N -> Nsc | Nmp",
"VP -> VPi | VPt | VPdt | Mod VP | VP Adv | VP PP", # a bit of subcat,
                                                    # allows mixed Adv and PP
                                                    # modifications :-(
"VPi -> Vi", # intransitive
"VPt -> Vt NP", # transitive
"VPdt -> VPo PP", # ditransitive, obligatory NP (obj.) & PP complements
"VPdt -> VPio NP", # ditransitive, obligatory NP (iobj.) & NP (obj)
"VPo -> Vdt NP", # direct object of ditransitive
"VPio -> Vdt NP", # indirect obj. part of dative-shifted ditransitive
"PP -> Prep NP",
"Det -> 'a' | 'the' | 'an' | 'my'",
"Nmp -> 'salad' | 'mushrooms' | 'pyjamas'",  # mass nouns, plural count nouns
"Nsc -> 'book' | 'fork' | 'flight' | 'salad' | 'drawing' | 'elephant'", # singular count
"Prep -> 'to' | 'with' | 'in'",
"Vi -> 'ate'",
"Vt -> 'ate' | 'book' | 'Book' | 'gave' | 'told' | 'shot'",
"Vdt -> 'gave' | 'told' ",
"Subconj -> 'that'",
"Mod -> 'Can' | 'will'",
"Aux -> 'did' ",
"WhAdv -> 'Why'",
"PropN -> 'John' | 'Mary' | 'NYC' | 'London'",
"Adj -> 'nice' | 'drawing'",
"Pro -> 'you' | 'he' | 'I'",
"Adv -> 'today'"
])

chart=CKY(grammar2)

def test2_1(verbose=False):
    global chart
    s="I ate salad."
    chart.parse(tokenise(s),verbose)
    print str(chart.firstTree())
    chart.pprint()

def test2_2(verbose=False):
    global chart
    s="I shot an elephant in my pyjamas."
    parsed=chart.parse(tokenise(s),verbose)
    if parsed:
        for tree in chart.allTrees():
            print str(tree)
            tree.draw()

#print "CKY parser..."
### Parse a simple sentence with the CKY parser and show the resulting matrix
#test2_1()

### show what happens in each cell of the matrix during the parse
#test2_1(True)

### Try the Groucho Marx sentence
#test2_2()

### Look at the matrix
# chart.pprint()

#################### EXERCISE 5 ####################
# In the first part we look at an existing grammar and identify the
# problems with number agreement

# To complete Exercise 5 you have to complete the definition of the
#  g1_agreement by copying rules from the g1 grammar and modifying
#  many of them

g1=parse_grammar("""
  # Grammatical productions.
  S -> NP VP
  NP -> Det N | Pro
  VP -> Vi | Vt NP
  # Lexical productions.
  Pro -> "i" | "we" | "you" | "he" | "she"
  Det -> "a" | "an" | "the"
  N ->  "dog" | "banana"
  Vi -> "sleep" | "eat"
  Vt -> "eat" | "ate"
  """)

# TODO modify the above grammar to enforce number agreement by expanding the
#  set of categories used
# Only the first rule is given. Add similar rules to complete, including
#  for example lexicon entries such as: Nsg ->  "dog" | "banana"
#                                       Vtsg -> "ate" | "eats"
g1_agreement=parse_grammar("""
  # Grammatical productions.
  S -> NPsg VPsg | NPpl VPpl
  # Lexical productions.
  """)

def test5():
  # code to test the parser with the first grammar
  app(g1, 'i sleep')
  app(g1, 'a dog eat banana')
  app(g1, 'the dogs sleep')
  
  # code to test the parser with the grammar with number constraints defined by you
  app(g1_agreement, 'the dogs sleep')
  app(g1_agreement, 'a dogs sleep')
  app(g1_agreement, 'a dog sleeps')
  app(g1_agreement, 'a dog sleep')

#Uncomment to test the code for Exercise 5
#test5()

################ EXERCISE 6a #######################
# For part a of Exercise 6 you have to inspect the output of the
#   parser when using the grammar_number_feature grammar
# Look at ParseWithFeatures to see how to parse with a grammar

from nltk import load_parser

# Function for loading a grammar and parsing a sentence with that grammar
# Input: the name of the grammar file and a sentence to parse
# Ouput: prints out the parser trace and the parse tree for the sentence if one exists
def ParseWithFeatures(feature_parser, sentence):
  global trees
  #to not see the rule aplications take out the trace argument
  
  #split sentence into tokens
  print "Parsing the sentence: %s" % sentence
  tokens = sentence.split()
  
  #parse the sentence and print the final parse tree
  trees = feature_parser.parse(tokens)
  empty=True
  for tree in trees:
    empty=False
    print(tree)
  if empty:
    print "No parse!"

def test6(trace=0):
  # File containing the grammar with number
  number_feature_file="/group/ltg/projects/fnlp/grammar_number_feature.fcfg"
  
  # Inspect the grammar
  print "The grammar with number feaure:"
  nltk.data.show_cfg(number_feature_file)
  print "\nCreating bottom-up parser with original grammar...\n"
  cp=load_parser(number_feature_file, trace=trace)
  
  # Test part a
  for s in ["the dogs sleep",
            "a dog sleeps",
            "i sleep",
            "he sleeps",
            "a dogs sleep",
            "a dog sleep"]:
    ParseWithFeatures(cp,s)
    print "---------------"

  print
  

################ EXERCISE 6b #######################
  # TODO modify a local copy of the grammar,
  #  provide the name of your grammar file, uncomment
  # Your grammar should enforce PERSON agreement. Hint when
  #  defining the lexicon you can leave a feature value underspecified
  #  such as PER=?p

  # Uncomment as necessary below here to test part b

  # number_person_feature_file="your_grammar_file.fcfg"
  
  # Inspect your augmented grammar
  #print "The grammar with number and person feaures:"
  #nltk.data.show_cfg(number_person_feature_file)
  #print "\nCreating bottom-up parser with your augmented grammar...\n"
  #cp=load_parser(number_person_feature_file, trace=trace)
  
  # for s in ["the dogs sleep",
            # "a dog sleeps",
            # "i sleep",
            # "he sleeps",
            # "a dogs sleep",
            # "a dog sleep"]:
#    ParseWithFeatures(cp,s)
#    print "---------------"

  return cp

# Uncomment to test your solution for Exercise 6 
#cp=test6()
