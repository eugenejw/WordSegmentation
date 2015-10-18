import sys
from os.path import join, dirname, realpath
from itertools import groupby
#import networkx as nx
from itertools import groupby, count
from math import log10
from functools import wraps
import inspect

if sys.hexversion < 0x03000000:
    range = xrange

def parse_file(filename):
    '''
    Global function that parses file and form a dictionary.
    '''
    with open(filename) as fptr:
        lines = (line.split('\t') for line in fptr)
        return dict((word, float(number)) for word, number in lines)

UNIGRAM_COUNTS = parse_file(join(dirname(realpath(__file__)), 'corpus', 'unigrams.txt'))

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
    def __init__(self):
        pass

    def get_unigram_score(self, word):

        if word in UNIGRAM_COUNTS:
            score = log10((UNIGRAM_COUNTS[word] / 1024908267229.0))
        else:
            score = log10((10.0 / (1024908267229.0 * 10 ** len(word))))
#        print "get_unigram_score's args->{0}; RESULT->{1}".format(word, score)
        return score


class IntersectCheck(object):
    def __init__(self):
            pass
        
    def check(self, tuple_0, tuple_1):
        '''                                                                                                                                                                 
        finds intersection of two words                                                                                                                                     
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
        self.lst= []

        
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
        #[[(1, 3), 'ace', (0,), -4.964005188761728], [(0, 3), 'face', (0,), -4.0926128161036965], [(4, 5), 'bo', (1, 2), -5.2144070696039995], [(4, 6), 'boo', (1, 2), -5.50655106498099], [(4, 7), 'book', (1, 2), -3.490909555336102], [(0, 7), 'facebook', (0,), -6.671146108616224]] 
        penalty = -10
        if prev == 0:
            gap = penalty  * (self.lst[current-1][0][0] - 0)
#            print "starting point gap found current->{0}, prev->{1}, GAP: {2}".format(current, prev, gap)

        elif self.lst[current-1][0][0] - self.lst[prev-1][0][1] == 1:
            gap = 0
#            print "seamless one found current->{0}, prev->{1}, GAP: {2}".format(current, prev, gap)

        else:
            gap = 0#penalty * (self.lst[current-1][0][0] - self.lst[prev-1][0][1])
#            print "Non-seamless one found current->{0}, prev->{1}, GAP: {2}".format(current, prev, gap)
        
        return gap
    
    def suffix_penaltize(self, current, suffix):
        if suffix == None:
            return 0 
        penalty = -10
        inter = IntersectCheck()
        if inter.check(self.lst[current-1][0], self.lst[suffix-1][0]):

            if suffix == len(self.lst):
                gap = penalty  * (self.lst[suffix-1][0][1] - self.lst[-1][0][1])
#                print "[start-1-jump] from {0} to {1}".format(self.lst[suffix-1][1], self.lst[current-1][1])
            else:
                gap = 0
#                print "[0-jump]non-starting overlapping suffix penality paid"
        else:
            
            gap = penalty  * (self.lst[suffix-1][0][1] - self.lst[current-1][0][0] - 1)
#            print "non-overlapping suffix penality paid"

        return gap
        
    def segment(self, text):
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
        print "candidate list is:"
        print candidate_list
        meaningful_words = []
        #meaningful_words is [((0, 10),"helloworld"),...]
        for each in candidate_list:
            #(17507324569.0, ('in', (8, 10)))
            for word in self.ngram_tree[each[1][0]]:
                if word in each[1][0] + pair_dic[each[1]]:
                    if self._string[each[1][1][0]:each[1][1][0]+len(word)] == word:
                        meaningful_words.append([(each[1][1][0], each[1][1][0]+len(word)-1), word, self.score_tool.get_unigram_score(word)])
        #sort the list in order of position in original text
        meaningful_words.sort(key=lambda x: x[0][1])
        #the sorted list is [((1, 4), 'ace', -4.964005188761728), ((0, 4), 'face', -4.0926128161036965), ((4, 6), 'bo', -5.2144070696039995), ((4, 7), 'boo', -5.550655106498099), ((4, 8), 'book', -3.490909555336102), ((0, 8), 'facebook', -6.671146108616224)]
        print meaningful_words
        tmp_lst = []
        inter = IntersectCheck()        
        for tu1 in meaningful_words:
            pos = 0
            prev_list = []
            for tu2 in meaningful_words:
               if not inter.check(tu1[0], tu2[0]) and tu1[0][0] == tu2[0][1] + 1:
                   prev_list.append(pos+1 if pos is not None else 1)
#                   print "prev list appended {}".format(pos+1)
               if pos == None:
                   pos = 1
               else:
                   pos += 1
#            print "for {0}, the non-intersected word positions are: {1}, words are,".format(tu1, tmp_lst)
            if prev_list:
                prev_list.reverse()
                tu1.insert(2, tuple(prev_list))
            else:
                tu1.insert(2, (0,))
#            tu1 = tuple(tu1)

        print meaningful_words
        #        return meaningful_words
        self.lst = meaningful_words
        #[[(1, 3), 'ace', (0,), -4.964005188761728], [(0, 3), 'face', (0,), -4.0926128161036965], [(4, 5), 'bo', (1, 2), -5.2144070696039995], [(4, 6), 'boo', (1, 2), -5.550655106498099], [(4, 7), 'book', (1, 2), -3.490909555336102], [(0, 7), 'facebook', (0,), -6.671146108616224]]
        
        j = len(self.lst)  #6
        print "j has length of {}".format(j)
        
        def _add(a, b, c):
            if a is None:
                return b + c
            else:
                return a + b + c
        def _add_2(a, b):
            if a is None:
                return b
            else:
                return a + b
        
        
        def opt(j, memo, suffix):
#            print "j is {}".format(j)
            if j == 0:
                return None
            
            if memo[j-1] is None:    
                memo[j-1] = max(
                    _add(opt(self.lst[(j-1)][2][0], memo, j), self.lst[(j-1)][3], self.penaltize(j, self.lst[(j-1)][2][0])),
#                    _add_2(opt(j-1, memo, j), self.suffix_penaltize(j, suffix)) if (j-1) in ending_words else None
                    opt(j-1, memo, j) if self.lst[(j-2)][0][1] == self.lst[(j-1)][0][1] else None
                    )
                return memo[j-1]

            else:
                return memo[j-1]

        tmp_lst = []
        memo = [None] * j
        ending_words = []
        count = 1
        for each in self.lst:
            if each[0][1] == self.lst[-1][0][1]:
                ending_words.append(count)
            count += 1
#        print "ending words are {}".format(ending_words)

#        while (j > 0):
        suffix = None
        tmp_lst.append(opt(j, memo, suffix))
#        j -= 1
        print tmp_lst
        print memo
                
        #[None, -14.714070147634635, -5.473683269420768, -3.7562511751772165, -5.839083271995247, -5.839083271995247, -8.520496534646679, -5.219687455935545, -5.219687455935545, -7.302519552753576, -7.302519552753576, -10.231246001516666, -10.231246001516666, -10.231246001516666, None, -10.551432729757513, -10.551432729757513, -10.551432729757513, -10.551432729757513]
        
        def solution(j, path):
            print "working on {}".format(self.lst[(j-1)][1])
            if j == 0:
                pass
            else:
                tmp_i = 0
                if len(self.lst[(j-1)][2]) > 1:
#                    print "len is now: {}".format(len(self.lst[(j-1)][2]))
                    tmp_p = None
                    for i in xrange(len(self.lst[(j-1)][2])):
                        print memo[self.lst[(j-1)][2][i]-1]
                        if memo[self.lst[(j-1)][2][i]-1] > tmp_p:
                            tmp_p = memo[self.lst[(j-1)][2][i]-1]
                            tmp_i = i
                
                
                if self.lst[(j-1)][3] + memo[(self.lst[(j-1)][2][tmp_i]) if self.lst[(j-1)][2][tmp_i]) != 0 else -1]  >= memo[j-2] if j-2 < 0 else memo[0]:
                    print "value of word:{0}<--{1} + its p:{2}'s value<--{3} is greater than its alter word{4}'s vale<--{5}".format(self.lst[(j-1)][1], self.lst[(j-1)][3], self.lst[self.lst[(j-1)][2][tmp_i]-1][1], memo[self.lst[(j-1)][2][tmp_i]-1], self.lst[j-2][1], memo[j-2])
                    path.append(self.lst[(j-1)][1])
                    solution(self.lst[(j-1)][2][tmp_i], path)
                else:
                    solution(j-1, path)

        j = len(self.lst)
#        while (j > 0):
        path = []
        solution(j, path)
        print "for node {0}, the path is {1}".format(j, path)
#            j -= 1
        return max(tmp_lst)

        

w = WordSegment()
w.segment('whoiswatching')
#w.segment('facebook')

#[[(1, 2), 're', (0,), -3.3763613545002444], [(1, 3), 'res', (0,), -4.714070147634635], [(0, 3), 'pres', (0,), -5.473683269420768], [(0, 4), 'press', (0,), -3.7562511751772165], [(5, 6), 'in', (4,), -2.0828320968180307], [(4, 6), 'sin', (3, 2), -4.7568720472963255], [(5, 7), 'ing', (4,), -4.764245359469461], [(0, 7), 'pressing', (0,), -5.219687455935545], [(4, 7), 'sing', (3, 2), -4.964814564774299], [(8, 9), 'in', (9, 8, 7), -2.0828320968180307], [(7, 9), 'gin', (6, 5), -5.719891144995784], [(8, 13), 'invest', (9, 8, 7), -5.0115585455811225], [(10, 13), 'vest', (11, 10), -5.440256832732823], [(11, 13), 'est', (0,), -4.246418105099501], [(12, 14), 'ste', (0,), -5.139639429862561], [(8, 15), 'invested', (9, 8, 7), -5.331745273821968], [(13, 15), 'ted', (0,), -5.013907093936451], [(10, 15), 'vested', (11, 10), -5.71636275049824], [(14, 15), 'ed', (14, 13, 12), -4.296448041283057]]        
        
    
