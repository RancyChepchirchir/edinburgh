# coding: iso-8859-1
'''Simple CKY parser'''
import sys,re
import nltk
from collections import defaultdict
import lab4_fix
from lab4_fix import parse_grammar, CFG
from nltk.grammar import Nonterminal
from nltk import Tree
from pprint import pprint
# The printing and tracing functionality is in a separate file in order
#  to make this file easier to read

class CKY:
    """An implementation of a Cocke-Kasami-Younger (bottom-up) CFG parser.

    Goes beyond strict CKY's insistance on Chomsky Normal Form.
    It allows arbitrary unary productions, not just NT->T
    ones, that is X -> Y with either Y -> A B or Y -> Z .
    It also allows mixed binary productions, that is NT -> NT T or -> T NT"""

    def __init__(self,grammar):
        '''Create an extended CKY parser for a particular grammar

        Grammar is an NLTK CFG
        consisting of unary and binary rules (no empty rules,
        no more than two symbols on the right-hand side
        (We use "symbol" throughout this code to refer to _either_ a string or
        an nltk.grammar.Nonterminal, that is, the two thinegs we find in
        nltk.grammar.Production)

        :type grammar: nltk.grammar.CFG, as fixed by cfg_fix
        :param grammar: A context-free grammar'''

        self.verbose=False
        assert(isinstance(grammar,CFG))
        self.grammar=grammar
        # split and index the grammar
        self.buildIndices(grammar.productions())

    def buildIndices(self,productions):
        '''Build indices to the productions passed in,
        splitting into unary (allowing symbol as RHS)
        and binary (allowing pairs of symbols as RHS)
        instance variables.
        In both cases the _value_ is a list of LHSes, so always Nonterminals

        :type productions: list(nltk.grammar.Production)
        :param productions: the rules for our language
        '''
        self.unary=defaultdict(list) # Indexed by RHS, a symbol
        self.binary=defaultdict(list) # Indexed by RHS, a pair of symbols
        for production in productions:
            # get the two parts of the production
            rhs=production.rhs()
            lhs=production.lhs()
            assert(len(rhs)>0 and len(rhs)<=2) # enforce the length restriction
            if len(rhs)==1:
                self.unary[rhs[0]].append(lhs) # index on the RHS symbol itself
            else:
                self.binary[rhs].append(lhs) # index on the RHS pair, which
                                             #  is a 2-tuple

    def parse(self,tokens,verbose=False):
        '''Initialise self.matrix from a candidate sentence,
        then fill it in using self.grammar.
        :type tokens: list(str)
        :param tokens: assumed to be drawn from the alphabet of the language
        :type verbose: boolean
        :param verbose: Print debugging info if True (default = False)
        :rtype: boolean
        :return: True iff the input is in the language defined by self.grammar
        '''
        self.verbose=verbose
        self.words = tokens
        self.n = len(self.words)+1 # because we number the 'spaces'
                                   #  _between_ the words
        self.matrix = []
        # We index by row, then column
        #  So Y below is 1,2 and Z is 0,3
        #    1   2   3  ...
        # 0  X   X   Z
        # 1      Y   X
        # 2          X
        # ...
        for r in range(self.n-1):
             # rows, indexed by the point _before_ a word
             row=[]
             for c in range(self.n):
                 # columns, indexed by the point _after_ a word
                 if c>r:
                     # This is one we care about, in or above the
                     #   main diagonal, so add a cell
                     row.append(Cell(r,c,self))
                 else:
                     # just a filler
                     row.append(None)
             self.matrix.append(row)
        self.unaryFill() # fill in the main diagonal
        self.binaryScan() # then scan the rest of the top-right half,
                           #  in length order
        # We win if the start symbol of the gramamr
        #  is in the top-right corner
        return self.matrix[0][self.n-1].hasCat(self.grammar.start())

    def unaryFill(self):
        '''Fill in the main diagonal of the CKY chart in self.matrix with
        tokens from the input, and look them up in the self.grammar
        '''
        for r in range(self.n-1):
            # for each cell on the main diagonal
            cell=self.matrix[r][r+1]
            # interpret it as the word _between_ input positions r and r+1
            word=self.words[r]
            label=TerminalLabel(word) # make a simple Label
            # add the word
            cell.addLabel(label)

    def binaryScan(self):
        '''The heart of the implementation:
        proceed across the upper-right diagonals
        left to right and
        in increasing order of constituent length
        '''
        for span in xrange(2, self.n):
            # We start with constituents of length 2,
            #  because unaryFill did length 1
            for start in xrange(self.n-span):
                # there are n-2 possible places for a length 2 constituent,
                #  n-3 for one of length 3, etc.
                end = start + span
                for mid in xrange(start+1, end):
                    # over all the possible intermediate points of
                    #  a span from start to end,
                    #  see if we can build something
                    self.maybeBuild(start, mid, end)

    def maybeBuild(self, start, mid, end):
        '''Build any constituents we can from start->end
        by combining a constituent that runs from start->mid
        with one that runs from mid->end, checking all the possible
        combinations in the binary rules of our grammar.
        :type start: int
        :param start: the left end of the substring we hope to parse
        :type mid: int
        :param mid: the point at which we will divide that substring
        :type end: int
        :param end: the right end of the substring we hope to parse
        '''
        self.log("%s--%s--%s:",start, mid, end)
        cell=self.matrix[start][end] # this is where our results will go
        left=self.matrix[start][mid] # this has possible first parts of a pair
        right=self.matrix[mid][end] # and this has the second parts
        leftLabs=left.labTab() # this has possible first parts of a pair
        rightLabs=right.labTab() # and this has the second parts
        for lsym in leftLabs.keys(): # labels we have in left cell
            for rsym in rightLabs.keys(): # symbols we have in right cell
                # lsym, rsym are all possible pairs of symbols
                if (lsym,rsym) in self.binary:
                    # OK, we have some new constituents
                    for nt in self.binary[(lsym,rsym)]:
                        self.log("%s -> %s %s", nt, lsym, rsym, indent=1)
                        # nt will be each possible symbol for a new spanning
                        #  constituent
                        label=BinaryNTLabel(nt,lsym,left,rsym,right)
                        cell.addLabel(label)

    def firstTree(self,symbol=None):
        """Put together a Tree rooted here for our 'first' parse
        with category specified by symbol, which defaults to
        the start symbol of our grammar
        :type symbol: nltk.grammar.Nonterminal (default = None)
        """
        if symbol is None:
            symbol=self.grammar.start()
        return self.matrix[0][self.n-1].trees(symbol,True)

    def allTrees(self,symbol=None):
        """Put together all Trees rooted here
        with category specified by symbol, which defaults to
        the start symbol of our grammar"""
        if symbol is None:
            symbol=self.grammar.start()
        # See if the top-right cell has a tree with the appropriate label
        return self.matrix[0][self.n-1].trees(symbol,False)

    def pprint(self,cell_width=8):
        '''Try to print matrix in a nicely lined-up way'''
        row_max_height=[0]*(self.n)
        col_max_width=[0]*(self.n)
        print_matrix=[]
        for r in range(self.n-1):
             # rows
             row=[]
             for c in range(r+1,self.n):
                 # columns
                 if c>r:
                     # This is one we care about, get a cell form
                     #  and tabulate width, height and update maxima
                     cf=self.matrix[r][c].str(cell_width)
                     nlines=len(cf)
                     if nlines>row_max_height[r]:
                         row_max_height[r]=nlines
                     if cf!=[]:
                         nchars=max(len(l) for l in cf)
                         if nchars>col_max_width[c]:
                             col_max_width[c]=nchars
                     row.append(cf)
             print_matrix.append(row)
        row_fmt='|'.join("%%%ss"%col_max_width[c] for c in range(1,self.n))
        row_index_len=len(str(self.n-2))
        row_index_fmt="%%%ss"%row_index_len
        row_div=(' '*(row_index_len+1))+(
            '+'.join(('-'*col_max_width[c]) for c in range(1,self.n)))
        print (' '*(row_index_len+1))+(' '.join(str(c).center(col_max_width[c])
                       for c in range(1,self.n)))
        for r in range(self.n-1):
            if r!=0:
                print row_div
            mrh=row_max_height[r]
            for l in range(mrh):
                print row_index_fmt%(str(r) if l==mrh/2 else ''),
                row_strs=['' for c in range(r)]
                row_strs+=[wtp(l,print_matrix[r][c],mrh) for c in range(self.n-(r+1))]
                print row_fmt%tuple(row_strs)

    def log(self,message,*args,**kwargs):
        if self.verbose:
            print ' '*kwargs.get('indent',0)+(message%args)

# A utility function
def wtp(l,subrows,maxrows):
    '''figure out what row or filler from within a cell
    to print so that the printed cell fills from
    the bottom.  l will be in range(mrh)'''
    offset=maxrows-len(subrows)
    if l>=offset:
        return subrows[l-offset]
    else:
        return ''

class Cell:
    '''A cell in a CKY matrix'''
    def __init__(self,row,column,matrix):
        '''Create a cell for a constituent from
         row to coll
         :type row: int
         :param row: the left endpoint of the corresponding constituent
         :type col: int
         :param col: the right endpoint of the corresponding constituent
         :type matrix: CKY
         :param matrix: Our parent matrix/parser, for logging/grammar/etc.
         '''
        self._row=row
        self._column=column
        self.matrix=matrix
        # We index all the labels by their symbol (aka category)
        # In other words, _labels.keys() will be a set (i.e. no
        #  duplicates) equivalent to what the _labels themselves are
        #  in the original simpler CKY recogniser.
        self._labels={}

    def __repr__(self):
        '''A simple string representation of the cell'''
        return "<Cell %s,%s>"%(self._row,self._column)

    def str(self,width=8):
        '''Try to format labels in a rectangule,
        aiming for max-width as given, but only
        breaking between labels'''
        labs=self.symbols()
        n=len(labs)
        res=[]
        if n==0:
            return res
        i=0
        line=[]
        ll=-1
        while i<n:
            s=str(labs[i])
            m=len(s)
            if ll+m>width and ll!=-1:
                res.append(' '.join(line))
                line=[]
                ll=-1
            line.append(s)
            ll+=m+1
            i=i+1
        res.insert(0,' '.join(line))
        return res
    
    def addLabel(self,label):
        '''Add a Label to this Cell
        :type label: Label
        :param label: the label to be added
        '''
        assert(isinstance(label,Label))
        symbol=label.symbol()
        assert(isinstance(label,Label))
        if symbol in self._labels:
            self.log("found another %s",symbol)
            self._labels[symbol].append(label)
        else:
            self._labels[symbol]=[label]
            # and propagate upward from its label by checking unary rules
            self.unaryUpdate(label,1)

    def symbols(self):
        '''symbols of all constituents here
        :rtype: list(str|Nonterminal)
        :return: one per Label'''
        return self._labels.keys()

    def labels(self):
        '''labels of all constituents here
        :rtype: list(Label)
        :return: the cell contents'''
        return self._labels.values()

    def labelsFor(self,symbol):
        '''The labels we have for a specified symbol.
        Will throw a KeyError if asked about an unknown symbol

        :type symbol: Nonterminal|str
        :param symbol: symbol to look for in the labels of this cell
        :rtype: list(Label)
        :return: all labels for symbol
        '''
        # Return the labels for a given symbol
        return self._labels[symbol]

    def labTab(self):
        return self._labels

    def hasCat(self,symbol):
        '''True iff a constituent of the given category is found here, i.e.
        there are one or more labels here for that category'''
        return symbol in self.labTab()

    def unaryUpdate(self,label,depth=0,recursive=False):
        """We've just added label here: apply any rules for its
        category found in unaries (a dictionary of LHS symbols
        indexed by RHS symbol)
        """
        symbol=label.symbol()
        if not recursive:
            self.log("%s",str(symbol),indent=depth)
        if symbol in self.matrix.unary:
            for parent in self.matrix.unary[symbol]:
                self.matrix.log("%s -> %s",
                                parent,symbol,indent=depth+1)
                self.addLabel(UnaryNTLabel(parent,symbol,self))

    def trees(self,symbol,justOne):
        """Pull together the Trees for a parse with category
        cat rooted in this cell (if justOne is False), or just
        the first one (if justOne is True)"""
        roots=self.labelsFor(symbol)
        if justOne:
            # Just build a tree using the first label in the list of labels
            #  for this category.
            #  It will be the first one found, which in turn
            #  will be the one with the simplest (fewest unary branches)
            #  and/or shortest (left child is shorter than all others)
            #  This results in a maximally right-branching tree
            return roots[0].trees(True)
        else:
            # Concatenate all the lists of all possible subtrees
            #  for all the labels we have here
            return sum((root.trees(False) for root in roots),[])

  # Debugging support
    def log(self,message,*args,**kwargs):
        self.matrix.log("%s,%s: "+message,self._row,
                        self._column,*args,**kwargs)

class Label:
    '''A label for a substring in a CKY chart Cell

    Includes a terminal or non-terminal symbol, possibly other
    information.  Add more to this docstring when you start using this
    class'''
    def __init__(self,symbol):
        '''Create a label from a symbol
        :type symbol: a string (for terminals) or an nltk.grammar.Nonterminal
        :param symbol: a terminal or non-terminal
        '''
        assert(isinstance(symbol,Nonterminal) or isinstance(symbol,str))
        self._symbol=symbol

    def __str__(self):
        '''A prettier short version of ourselves -- just our cat'''
        return str(self.symbol())

    def symbol(self):
        return self._symbol

class TerminalLabel(Label):
    def __eq__(self,other):
        '''How to test for equality -- other must be a Label,
        symbols match'''
        assert isinstance(other,Label)
        return self._symbol==other._symbol

    def __repr__(self):
        '''A simple string representation of a terminal Label'''
        return "<Label %s>"%self._symbol

    def trees(self,justOne):
        if justOne:
            # We just use our symbol
            return self.symbol()
        else:
            # Need a list, because higher up it will be iterated over
            return [self.symbol()]

class UnaryNTLabel(Label):
    def __init__(self,cat,lCat=None,lCell=None):
        Label.__init__(self,cat)
        assert(isinstance(lCat,Nonterminal) or
               isinstance(lCat,str))
        assert(isinstance(lCell,Cell))
        self._lCat=lCat
        self._lCell=lCell

    def __eq__(self,other):
        '''How to test for equality -- other must be a Label,
        and three local variables have to be equal'''
        assert isinstance(other,Label)
        return (self._symbol==other._symbol and
                self._lCat==other._lCat and
                self._lCell==other._lCell)

    def __repr__(self):
        '''A simple string representation of a unary NT Label'''
        return "<Label %s%s>"%(self._symbol,
                             (" %s@%s"%(self._lCat,repr(self._lCell))))

    def trees(self,justOne):
        if justOne:
            # We just need one Tree from our (left) sub-cell
            #  to make the single child of a new Tree with our category
            res=Tree(self.symbol(),[self._lCell.trees(self._lCat,True)])
        else:
            # We need to build one Tree with our category
            #  for every possible of subTree
            #  from our (left) sub-cell
            res=[Tree(self.symbol(),[tree])
                 for tree in self._lCell.trees(self._lCat,False)]
        return res

class BinaryNTLabel(Label):
    def __init__(self,cat,lCat=None,lCell=None,rCat=None,rCell=None):
        Label.__init__(self,cat)
        assert(isinstance(lCat,Nonterminal) or
               isinstance(lCat,str))
        assert(isinstance(lCell,Cell))
        self._lCat=lCat
        self._lCell=lCell
        assert(isinstance(rCat,Nonterminal)
               or isinstance(rCat,str))
        assert(isinstance(rCell,Cell))
        self._rCat=rCat
        self._rCell=rCell

    def __repr__(self):
        '''A simple string representation of a binary Label'''
        return "<Label %s%s%s>"%(self._symbol,
                             (" %s@%s"%(self._lCat,repr(self._lCell))),
                             (" %s@%s"%(self._rCat,repr(self._rCell))))

    def __eq__(self,other):
        '''How to test for equality -- other must be a Label,
        and all five local variables have to be equal'''
        assert isinstance(other,Label)
        return (self._symbol==other._symbol and
                self._lCat==other._lCat and
                self._rCat==other._rCat and
                self._lCell==other._lCell and
                self._rCell==other._rCell)

    def trees(self,justOne):
        """Pull together Trees for this label (if justOne is False), or just
        the first one (if justOne is True)"""
        if justOne:
            # We need one Tree each from our left and right sub-cells,
            #  allowing us to build a new Tree with our category
            #  as node label and those Trees as children
            res=Tree(self.symbol(),[self._lCell.trees(self._lCat,True),
                                   self._rCell.trees(self._rCat,True)])
        else:
            # We need to build one Tree with our category
            #  for every possible pair of subTrees
            #  from our left and right sub-cells
            res=[]
            for lTree in self._lCell.trees(self._lCat,False):
                for rTree in self._rCell.trees(self._rCat,False):
                    res.append(Tree(self.symbol(),[lTree,rTree]))
        return res

####################

def tokenise(tokenstring):
    '''Split a string into a list of tokens

    We treat punctuation as
    separate tokens, and split contractions into their parts.

    So for example "I'm leaving." --> ["I","'m","leaving","."]
      
    @type tokenstring: str
    @param tokenstring the string to be tokenised
    @rtype: list(str)
    @return: the tokens found in tokenstring'''

    # Note that we do _not_ split on word-internal hyphens, and do
    #  _not_ attempt to diagnose or repair end-of-line hyphens
    # Nor do we attempt to distinguish the use of full-stop to mark
    #  abbreviations from its end-of-sentence use, or the use of single-quote
    #  for possessives from its use for contractions and quotations (for which
    #  the following arguably does the wrong thing.
    return re.findall(
        # We use three sub-patterns:
        #   one for words and the first half of possessives
        #   one for the rest of possessives
        #   one for punctuation
        r"[-\w]+|'\w+|[^-\w\s]+",
        tokenstring,
        re.U # Use unicode classes, otherwise we would split
             # "são jaques" into ["s", "ão","jaques"]
        )


