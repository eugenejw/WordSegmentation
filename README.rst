Python Word Segmentation
========================

WordSegmentation is an Apache2 licensed module for English word
segmentation, written in pure-Python, and based on a trillion-word corpus.

Inspired by Grant Jenks' https://pypi.python.org/pypi/wordsegment.
Based on word weighing algorithm from the chapter "Natural Language Corpus Data" by Peter Norvig
from the book "Beautiful Data" (Segaran and Hammerbacher, 2009).

Data files are derived from the Google Web Trillion Word Corpus, as described
by Thorsten Brants and Alex Franz, and `distributed`_ by the Linguistic Data
Consortium. 

Features
--------

- Pure-Python
- Segmentation Algorithm Using Divide and Conquer so that there is NO max length limit set to input text.
- Segmentation Algotrithm used Dynamic Programming to achieve a polynomial time complexity.
- Used Google Trillion Corpus to do scoring for the word segmentation.
- Developed on Python 2.7
- Tested on CPython 2.6, 2.7, 3.4.

Quickstart
----------

Installing WordSegment is simple with
`pip <http://www.pip-installer.org/>`_::

    $ pip install wordsegmentation

Tutorial
--------

In your own Python programs, you'll mostly want to use `segment` to divide a
phrase into a list of its parts::

    >>> from wordsegmentation import Wordsegment
    >>> ws = WordSegment(use_google_corpus=True)
    
    >>> ws.segment('universityofwashington')
    ['university', 'of', 'washington']
    >>> ws.segment('thisisatest')
    ['this', 'is', 'a', 'test']  
    >>> segment('thisisatest')
    ['this', 'is', 'a', 'test']


Bug Report
------------
`Weihan@Github <https://github.com/eugenejw/WordSegmentation>`:


Tech Details
------------

In the code, the segmentation algorithm consists of the following steps,
1)divide and conquer -- safely divide the input string into substring. This way we solved the length limit which will dramatically slow down the performance. 
for example, "facebook123helloworld" will be treated as 3 sub-problems -- "facebook", "123", and "helloworld".                 
2)for each sub-string. I used dynamic programming to calculate and get the optimal words.
3)combine the sub-problems, and return the result for the original string.      

Segmentation algorithm used in this module, has achieved a time-complexity of O(n^2).       

By comparison to existing segmentation algorithms, this module does better on following aspects,
1)can handle very long input. There is no arbitary max lenght limit set to input string.
2)segmentation finished in polynomial time via dynamic programming.
3)by default, the algorithm uses a filtered Google corpus, which contains only English words that could be found in dictionary.

An extreme example is shown below::
   >>>ws.segment('MARGARETAREYOUGRIEVINGOVERGOLDENGROVEUNLEAVINGLEAVESLIKETHETHINGSOFMANYOUWITHYOURFRESHTHOUGHTSCAREFORCANYOUAHASTHEHEARTGROWSOLDERITWILLCOMETOSUCHSIGHTSCOLDERBYANDBYNORSPAREASIGHTHOUGHWORLDSOFWANWOODLEAFMEALLIEANDYETYOUWILLWEEPANDKNOWWHYNOWNOMATTERCHILDTHENAMESORROWSSPRINGSARETHESAMENORMOUTHHADNONORMINDEXPRESSEDWHATHEARTHEARDOFGHOSTGUESSEDITISTHEBLIGHTMANWASBORNFORITISMARGARETYOUMOURNFOR')
   ['margaret', 'are', 'you', 'grieving', 'over', 'golden', 'grove', 'un', 'leaving', 'leaves', 'like', 'the', 'things', 'of', 'man', 'you', 'with', 'your', 'fresh', 'thoughts', 'care', 'for', 'can', 'you', 'a', 'has', 'the', 'he', 'art', 'grows', 'older', 'it', 'will', 'come', 'to', 'such', 'sights', 'colder', 'by', 'and', 'by', 'nor', 'spa', 're', 'a', 'sigh', 'though', 'worlds', 'of', 'wan', 'wood', 'leaf', 'me', 'allie', 'and', 'yet', 'you', 'will', 'weep', 'and', 'know', 'why', 'now', 'no', 'matter', 'child', 'the', 'name', 'sorrows', 'springs', 'are', 'the', 'same', 'nor', 'mouth', 'had', 'non', 'or', 'mind', 'expressed', 'what', 'he', 'art', 'heard', 'of', 'ghost', 'guessed', 'it', 'is', 'the', 'blight', 'man', 'was', 'born', 'for', 'it', 'is', 'margaret', 'you', 'mourn', 'for']


WordSegment License
-------------------

Copyright 2015 Weihan Jiang

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
