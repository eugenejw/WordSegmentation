"""
English Word Segmentation in Python

Word segmentation is the process of dividing a phrase without spaces back
into its constituent parts. For example, consider a phrase like "thisisatest".
For humans, it's relatively easy to parse. 
This module makes it easy for
machines too. Use `segment` to parse a phrase into its parts:
>>> from wordsegment import segment
>>> segment('thisisatest')
['this', 'is', 'a', 'test']
In the code, 1024908267229 is the total number of words in the corpus. A
subset of this corpus is found in unigrams.txt and bigrams.txt which
should accompany this file. A copy of these files may be found at
http://norvig.com/ngrams/ under the names count_1w.txt and count_2w.txt
respectively.
Copyright (c) 2015 by Weihan Jiang

Scoring mechanism based on formula from the chapter "Natural Language Corpus Data"
from the book "Beautiful Data" (Segaran and Hammerbacher, 2009)
http://oreilly.com/catalog/9780596157111/
Original Copyright (c) 2008-2009 by Peter Norvig
"""

import sys
from os.path import join, dirname, realpath
import networkx as nx
from itertools import groupby, count
from math import log10
import copy

if sys.hexversion < 0x03000000:
    range = xrange

def parse_file(filename):
    '''
    Global function that parses file and form a dictionary.
    '''
    with open(filename) as fptr:
        lines = (line.split('\t') for line in fptr)
        return dict((word, float(number)) for word, number in lines)

#UNIGRAM_COUNTS = parse_file(join(dirname(realpath(__file__)), 'corpus', 'unigrams.txt'))
UNIGRAM_COUNTS = parse_file(join(dirname(realpath(__file__)), 'corpus', 'unigrams.txt.original'))

def as_range(group):
    '''
    Global function returns range
    '''
    tmp_lst = list(group)
    return tmp_lst[0], tmp_lst[-1]

class Data(object):
    '''
    Read corpus from path, and provide the following functionalities,
    1. data as "property", it is a dictionary where key is word,
       while the value is the frequency count of this word.
    2. generator that yield word and its frequency
    '''
    def __init__(self, use_google_corpus):
        self._unigram_counts = dict()
        self._use_google_corpus = use_google_corpus
        if self._use_google_corpus:
            #use pure google corpus
            self._unigram_counts = parse_file(
                join(dirname(realpath(__file__)), 'corpus', 'filtered_1_2_letter_only.txt')
            )
        else:
            #use dictionary-filtered google corpus
            self._unigram_counts = parse_file(
                join(dirname(realpath(__file__)), 'corpus', 'unigrams.txt')
            )

    @property
    def data(self):
        '''
        return the whole dictionary out to user as a property.
        '''
        return self._unigram_counts

    def __iter__(self):
        for each in self._unigram_counts.keys():
            yield each

class ConstructCorpus(object):
    '''
    according to the minimal character limit,
    construct a corpus at initial time.
    it provides the following two properties,
    1. ngram_distribution -- a dictionary where key is the ngram,
       value is an int of summation of frequency of each English
       word starts with that specific ngram.
    2. ngram_tree -- a dictionary where key is the ngram,
       value is a list containing all possile English word
       starts with that specific ngram.
    '''
    def __init__(self, min_length, use_google_corpus):
        self._minlen = min_length
        self._use_google_corpus = use_google_corpus

    @property
    def ngram_distribution(self):
        '''
        return a dictionary containing the following pairs,
        key: ngram string, for example, when minlen=5,
             the ngram string for word "university" is "unive".
        value: added-up frequency(from google corpus) of all
               words starting with "unive".
        '''
        ngram_distribution = dict()
        instance_d = Data(self._use_google_corpus)
        data = instance_d.data
        for entry in instance_d:
            if len(entry) >= self._minlen:
                cut = entry[:self._minlen]
                if cut in ngram_distribution:
                    ngram_distribution[cut] += data[entry]
                else:
                    ngram_distribution[cut] = data[entry]

        return ngram_distribution

    @property
    def ngram_tree(self):
        '''
        return a dictionary containing the following pairs,
        key: ngram string, for example, when minlen=5,
             the ngram string for word "university" is "unive".
        value: all words starting with the ngram,
               in the example, it is "unive".
        '''
        ngram_tree = dict()
        instance_d = Data(self._use_google_corpus)
        for entry in instance_d:
            if len(entry) >= self._minlen:
                cut = entry[:self._minlen]
                if cut in ngram_tree:
                    ngram_tree[cut].append(entry)
                else:
                    ngram_tree[cut] = [entry]

        return ngram_tree

class Scoring(object):
    """
    Methond that scores words by using probability from Google Trillion Corpus
    """
    def __init__(self):
        pass

    def get_unigram_score(self, word):
        """
        function that score single words
        2-word scoring to be added
        """
        if word in UNIGRAM_COUNTS:
            score = log10((UNIGRAM_COUNTS[word] / 1024908267229.0))
        else:
            score = log10((10.0 / (1024908267229.0 * 10 ** len(word))))
        #print "get_unigram_score's args->{0}; RESULT->{1}".format(word, score)
        return score

class IntersectCheck(object):
    """
    Method that checks intersection between words
    """
    def __init__(self):
        '''
        taks no arguments
        '''
        self.tu1 = None
        self.tu2 = None

    def check(self, tuple_0, tuple_1):
        '''
        finds intersection of two words
        input: position tuples
        return: boolean values showing whether intersection detected
        '''
        self.tu1 = tuple_0
        self.tu2 = tuple_1
        word1 = range(self.tu1[0], self.tu1[1]+1)
        word2 = range(self.tu2[0], self.tu2[1]+1)
        tmp_xs = set(word1)
        #print "returning {}".format(tmp_xs.intersection(word2))
        return tmp_xs.intersection(word2)

class WordSegment(object):
    '''
    def __init__(self, string, casesensitive = False):
        self._casesensitive = casesensitive
        self._string = string
        #facebook
        self.lst = ((0, 3, 0, 1), (1, 3, 0, 1), (4, 7, 1, 1), (6, 7, 1, 4), (0, 7, 0, 1))
    '''

    def __init__(self, min_length=2, casesensitive=False, use_google_corpus=False):
        self._minlen = min_length
        self._string = ''
        self._use_google_corpus = use_google_corpus
        self._casesensitive = casesensitive
        corpus = ConstructCorpus(self._minlen, self._use_google_corpus)
        self.ngram_distribution = corpus.ngram_distribution
        self.ngram_tree = corpus.ngram_tree
        self.score_tool = Scoring()
        self.lst = []

    def _divide(self):
        """
        Iterator finds ngrams(with its position in string) and their suffix.
        An example input of string "helloworld" yields the following tuples,
        (('hello',(0,5)), 'world')
        (('ellow',(1,6)), 'orld')
        (('llowo',(2,7)), 'rld')
        (('lowor',(3,8)), 'ld')
        (('oworl',(4,9)), 'd')
        (('world',(5,10)), '')
        """
        counter = 0
        for cut_point in range(self._minlen, len(self._string)+1):
            yield (
                (self._string[counter:cut_point], (counter, counter+self._minlen)),
                self._string[cut_point:]
                )
            counter += 1

    def penaltize(self, current, prev):
        '''
        function that imposes penalty to any gap between words
        example input is shown below
        #[[(1, 3), 'ace', (0,), -4.964005188761728], [(0, 3), 'face', (0,), -4.0926128161036965], [(4, 5), 'bo', (1, 2), -5.2144070696039995], [(4, 6), 'boo', (1, 2), -5.50655106498099], [(4, 7), 'book', (1, 2), -3.490909555336102], [(0, 7), 'facebook', (0,), -6.671146108616224]]
        '''
        penalty = -10
        #starting point penalty
        if prev == 0:
            gap = penalty  * (self.lst[current-1][0][0] - 0)
            #print "starting point gap found current->{0}, prev->{1}, GAP: {2}".format(current, prev, gap)
        elif self.lst[current-1][0][0] - self.lst[prev-1][0][1] == 1:
            gap = 0
            #print "seamless one found current->{0}, prev->{1}, GAP: {2}".format(current, prev, gap)
        else:
            gap = 0
            #print "Non-seamless one found current->{0}, prev->{1}, GAP: {2}".format(current, prev, gap)
        return gap

    def _init_graph(self, meaningful_words):
        '''
        function that creates graph for each requesting string
        below is an example input of this function
        [[(0, 2), 'fac', -14.0106849963], [(1, 3), 'ace', -4.964005188761728], [(0, 3), 'face', -4.0926128161036965], [(2, 4), 'ceb', -14.0106849963], [(4, 5), 'bo', -5.2144070696039995], [(3, 5), 'ebo', -14.0106849963], [(4, 6), 'boo', -5.550655106498099], [(4, 7), 'book', -3.490909555336102], [(0, 7), 'facebook', -6.671146108616224], [(3, 7), 'ebook', -16.0106849963], [(5, 7), 'ook', -14.0106849963]]
        '''
        word_graph = nx.Graph()
        word_graph.add_nodes_from(meaningful_words)
        inter = IntersectCheck()
        for each in meaningful_words:
            for each_2 in meaningful_words:
                if each == each_2:
                    continue
                elif inter.check(each[0], each_2[0]):
                    if (each[0], each_2[0]) in word_graph.edges():
                        continue
                    else:
                        word_graph.add_edge(each, each_2)
        return word_graph

    def _find_components(self, meaningful_words):
        '''
        function that finds the components in the graph.
        each component represents overlaping words
        for example, in the example below, except the word "anotherword",
        all rest words have at least one character contained in other words.
        They will become one component in the who string-level graph

        Example input is a list like this: [((0, 3),"face"), ((0, 7),"facebook"), ((1, 3),"ace"),
        ((4, 6),"boo"), ((4, 7),"book"), ((6, 7), "ok"), ((8, 19),"anotherword")]
        '''
        component_graph = nx.Graph()
        component_graph = self._init_graph(meaningful_words)
        components = []
        components = list(nx.connected_component_subgraphs(component_graph))
        return components

    def suffix_penaltize(self, current, suffix):
        """
        function that imposes penalties to suffix
        """
        if suffix == None:
            return 0
        #let default penalty = 10
        penalty = -10
        inter = IntersectCheck()
        if inter.check(self.lst[current-1][0], self.lst[suffix-1][0]):
            if suffix == len(self.lst):
                gap = penalty  * (self.lst[suffix-1][0][1] - self.lst[-1][0][1])
            #print "[start-1-jump] from {0} to {1}".format(self.lst[suffix-1][1], self.lst[current-1][1])
            else:
                gap = 0
                #print "[0-jump]non-starting overlapping suffix penality paid"
        else:
            gap = penalty  * (self.lst[suffix-1][0][1] - self.lst[current-1][0][0] - 1)
            #print "non-overlapping suffix penality paid"

        return gap

    def _opt_component(self, component):
        """
        function that finds optimal segmentation for each component
        """
        #the sorted list is [((1, 4), 'ace', -4.964005188761728), ((0, 4), 'face', -4.0926128161036965), ((4, 6), 'bo', -5.2144070696039995), ((4, 7), 'boo', -5.550655106498099), ((4, 8), 'book', -3.490909555336102), ((0, 8), 'facebook', -6.671146108616224)]
        meaningful_words = component.nodes()
        meaningful_words.sort(key=lambda x: x[0][1])
        #print "old meaningful list is {}".format(meaningful_words)
        old_meaningful_words = copy.deepcopy(meaningful_words)
        meaningful_words = []
        for each in old_meaningful_words:
            meaningful_words.append(list(each))
        #print "new meaningful list is {}".format(meaningful_words)
        tmp_lst = []
        inter = IntersectCheck()
        for tu1 in meaningful_words:
            pos = 0
            prev_list = []
            for tu2 in meaningful_words:
                if not inter.check(tu1[0], tu2[0]) and tu1[0][0] == tu2[0][1] + 1:
                    prev_list.append(pos+1 if pos is not None else 1)
                    #print "prev list appended {}".format(pos+1)
                if pos == None:
                    pos = 1
                else:
                    pos += 1
                    #print "for {0}, the non-intersected word positions are
                    #: {1}, words are,".format(tu1, tmp_lst)
            if prev_list:
                prev_list.reverse()
                tu1.insert(2, tuple(prev_list))
            else:
                tu1.insert(2, (0,))

        self.lst = meaningful_words

        j = len(self.lst)
        #print "j has length of {}".format(j)

        def _add(input1, input2, input3):
            """
            function that adds up 3 inputs
            """
            if input1 is None:
                return input2 + input3
            else:
                return input1 + input2 + input3

        def opt(j, memo):
            """
            Recurrence using Dynamic programming
            """
            #print "j is {}".format(j)
            if j == 0:
                return None

            if memo[j-1] is None:
                memo[j-1] = max(
                    #choose j
                    _add(opt(self.lst[(j-1)][2][0], memo), self.lst[(j-1)][3], self.penaltize(j, self.lst[(j-1)][2][0])),
                    #not choose j and jump to j-1 only when nesrest overlpping word has the same finish position
                    opt(j-1, memo) if self.lst[(j-2)][0][1] == self.lst[(j-1)][0][1] else None
                    )
                return memo[j-1]

            else:
                return memo[j-1]

        tmp_lst = []
        #create a memo table for dynamic programming
        memo = [None] * j
        ending_words = []
        counter = 1
        for each in self.lst:
            if each[0][1] == self.lst[-1][0][1]:
                ending_words.append(counter)
            counter += 1

        tmp_lst.append(opt(j, memo))
        #print tmp_lst

        new_lst = []
        pos = 0
        for each in tmp_lst:
            new_lst.append((each, ))
        #print memo

        def find_path(j, path):
            """
            find the optimal segmentation from the memo list
            """
            #print "working on {}".format(self.lst[(j-1)][1])
            #print "j is {}".format(j)

            if j == 0:
                pass
            elif memo[j-1] == memo[j-2] if j-2 >= 0 else memo[0]:
                if j != 1:
                    find_path(j-1, path)
                elif j == 1:
                    path.append(((self.lst[0][0][0], self.lst[0][0][1]), self.lst[0][1]))
            else:
                #if p(j) exists
                if len(self.lst[(j-1)][2]) > 0:
                    tmp_i = 0
                    #if p(j) == 1
                    if len(self.lst[(j-1)][2]) == 1:
                        path.append(((self.lst[j-1][0][0], self.lst[j-1][0][1]), self.lst[j-1][1]))
                        #print "[single P]jumped to {}".format(self.lst[j-1][2][tmp_i])
                        find_path(self.lst[j-1][2][tmp_i], path)
                    #if p(j) > 1
                    elif len(self.lst[(j-1)][2]) > 1:
                        prev_list = self.lst[(j-1)][2][:]
                        prev_list = list(prev_list)
                        prev_list.reverse()
                        p_list = []
                        #get the p, whose memo value is max
                        for i in xrange(len(self.lst[(j-1)][2])):
                            p_list.append(memo[self.lst[(j-1)][2][i]-1])
                            #print "p_list is {}".format(p_list)
                        max_p = max(p_list)

                        prev_list = self.lst[(j-1)][2][:]
                        #print "prev_list is {}".format(prev_list)
                        for i in xrange(len(self.lst[(j-1)][2])):
                            #print memo[prev_list[i]-1]
                            if memo[prev_list[i]-1] == max_p:
                                #tmp_p = memo[self.lst[(j-1)][2][i]-1]
                                tmp_i = i
                                break
                        #print "best i is {}".format(tmp_i)
                        #print "tmpi is {}".format(tmp_i)
                        #path.append(self.lst[(tmp_p - 1)][1])
                        path.append(((self.lst[j-1][0][0], self.lst[j-1][0][1]), self.lst[j-1][1]))
                        #print "jumped to {}".format(prev_list[tmp_i])
                        find_path(prev_list[tmp_i], path)
                else:
                    find_path(j-1, path)


        result = tmp_lst[0]
        path = []
        max_v = [i for i, j in enumerate(memo) if j == result]
        j = max_v[-1] + 1
        find_path(j, path)
        path.reverse()
        #print "for node {0}, the path is {1}".format(j, path)
        words_list = []
        for each in path:
            words_list.append(each[1])
        return ((path[0][0][0], self.lst[-1][0][1]), words_list)

    #public interface
    def segment(self, text):
        """
        public interface
        input: string, typically a sentence without spaces
        output: list of optimal words
        """
        if self._casesensitive == False:
            self._string = text.lower()
            self._string = self._string.strip("'")
        else:
            #for current version, only supports lowercase version
            pass

        candidate_list = []
        pair_dic = dict()
        for prefix, suffix in self._divide():
            pair_dic[prefix] = suffix
            if prefix[0] in self.ngram_distribution:
                candidate_list.append(
                    (self.ngram_distribution[prefix[0]], prefix)
                )
            else:
                #means this prefix was not likely
                #to be a part of meaningful word
                pass

        candidate_list.sort(reverse=True)
        #now candidate list is [(2345234, ("hello",(0,5)))]
        #print "candidate list is:"
        #print candidate_list
        meaningful_words = []
        #meaningful_words is [((0, 10),"helloworld"),...]
        for each in candidate_list:
            #(17507324569.0, ('in', (8, 10)))
            for word in self.ngram_tree[each[1][0]]:
                if word in each[1][0] + pair_dic[each[1]]:
                    if self._string[each[1][1][0]:each[1][1][0]+len(word)] == word:
                        meaningful_words.append(((each[1][1][0],
                                                  each[1][1][0]+len(word)-1), word,
                                                 self.score_tool.get_unigram_score(word)))

        #sort the list in order of position in original text
        meaningful_words.sort(key=lambda x: x[0][1])

        #find components from the original input string
        components = []
        components = self._find_components(meaningful_words)

        post_components = []
        for each in components:
            post_components.append(self._opt_component(each))
        #print "{}components found".format(len(post_components)
        #print "post_components is {}".format(post_components)
        meaningful_pos_lst = []
        for each in post_components:
            #print "each is {}".format(each)
            meaningful_pos_lst += range(int(each[0][0]), int(each[0][1]+1))
        meaningful_pos_lst.sort()
        #print "DEBUG meaningful_pos_lst is {}".format(meaningful_pos_lst)

        non_meaning_pos_lst = []
        for pos in xrange(len(self._string)):
            if pos in meaningful_pos_lst:
                continue
            else:
                non_meaning_pos_lst.append(pos)

        non_meaningful_range = []
        non_meaningful_range = [
            as_range(g) for _, g in groupby(non_meaning_pos_lst, key=lambda n, c=count(): n-next(c))
            ]

        meaningful_dic = dict()

        overall_pos_lst = []
        for each in non_meaningful_range:
            overall_pos_lst.append(each)
        for component in post_components:
            overall_pos_lst.append(component[0])
            meaningful_dic[component[0]] = component[1]

        #print "meaningful_dic is {}".format(meaningful_dic)
        #print "self._string is {}".format(self._string)
        overall_pos_lst.sort()
        #print "overall_pos_lst is {}".format(overall_pos_lst)
        return_lst = []
        overall_pos_lst.sort()
        for each in overall_pos_lst:
            if each in meaningful_dic:
                return_lst.extend(meaningful_dic[each])
            else:
                return_lst.append(self._string[each[0]:each[1]+1])

        #print "RESULT: {}\n".format(return_lst)
        return return_lst


w = WordSegment()
print w.segment('thisisatest')
'''
Test Cases
w = WordSegment(use_google_corpus=True)
#w = WordSegment(use_google_corpus=False)
w.segment('facebookingirl')
w.segment('facebook')
w.segment('whoiswatching')
w.segment('acertain')
w.segment('theyouthevent')
w.segment('baidu')
w.segment('google')
w.segment('from')
print w.segment('MARGARETAREYOUGRIEVINGOVERGOLDENGROVEUNLEAVINGLEAVESLIKETHETHINGSOFMANYOUWITHYOURFRESHTHOUGHTSCAREFORCANYOUAHASTHEHEARTGROWSOLDERITWILLCOMETOSUCHSIGHTSCOLDERBYANDBYNORSPAREASIGHTHOUGHWORLDSOFWANWOODLEAFMEALLIEANDYETYOUWILLWEEPANDKNOWWHYNOWNOMATTERCHILDTHENAMESORROWSSPRINGSARETHESAMENORMOUTHHADNONORMINDEXPRESSEDWHATHEARTHEARDOFGHOSTGUESSEDITISTHEBLIGHTMANWASBORNFORITISMARGARETYOUMOURNFOR') 
w.segment('pressinginvestedthebecomethemselves')
'''

