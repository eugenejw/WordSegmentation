import re
import time
import itertools, enchant

dictionary_us = enchant.Dict("en_US")
dictionary_gb = enchant.Dict("en_GB")
dictionary_au = enchant.Dict("en_US")
dictionary_ca = enchant.Dict("en_US")
white_list = ['january', 'february', 'march', 'april','may','june','july','august','september','october','november','december','america','american']
two_letter_words = "aa ab ad ah ai am an ar as at aw ax ba be bi bo by de do ed eh em en et ex go ha he hi ho id if in is it la ma me mi mm mo my na no od oe of oh on op or os ox pa pe pi re so to uh um un up ur us we wo yo".split()

def filter_corpus():
    with open('filtered_unigram_v2_allow_upper.txt', 'w') as output_file:
      with open("unigrams.txt.original") as f:
        count = 0
        for line in f:
            count += 1
#            if float(count)%float(1000) == 0:
#                print "working on line: {0}".format(line)
#                print "{0}% completed!\n".format(100*(float(count)/float(333333)))
            pattern = re.search(r'(\w+)[\t](.*)', line)
           # try:
#           time.sleep(1)
#           print "herehere\n"
            word = pattern.group(1).strip()
            f_count = pattern.group(2).strip()
#            print "word is {0}".format(word)
#            print "its length is {0}".format(len(word))
                
#            if dictionary_us.check(word) or dictionary_gb.check(word) or word in white_list
            if len(word)==1:
                continue
            if len(word)==2:
                if word in two_letter_words:
#                    print "wiring word {0} to file".format(word)
                    output_file.write(line)
                else:
                    continue

                
            else:

                if dictionary_us.check(word) or dictionary_gb.check(word) or dictionary_us.check(word.capitalize()) or dictionary_us.check(word.upper()):
                    if len(word)==3:
                        print word
                    #               time.sleep(1)
#                    print "wiring word {0} to file".format(word)
                    output_file.write(line)   
                else:
                    pass
#                new_f_count = str(float(f_count)*0.1)
#                new_line = word + ' ' + '\t' + new_f_count + '\n'
#                output_file.write(new_line)
            #except:
            #    with open("filer_error.txt", "a") as error_file:
            #        error_file.write("error met when dealing with: {0}".format(line))
                
filter_corpus()
            



