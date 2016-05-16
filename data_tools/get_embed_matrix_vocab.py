import sys
import os
from utils import Vocab
import numpy as np
import pickle

"""
This file will convert the preprocessed data files into a train, val and
test set. We will partition our data into 80% train, 10% val and 10% test.
We will also leave out an entire topic -- Argentina -- and use 50% of it as val
and 50% of it as test. The rest of val and test will be sampled uniformly
at random from the rest of the data.
"""

def get_argentina_filenames(fname):
    """
    Gets the file names corresponding to topic Argentina.
    """
    out = []
    f = open(fname, 'r')
    for line in f:
        sp_line = line.split()
        topic = sp_line[0].split('=')[1]
        if topic == 'argentina':
            out.append(sp_line[1].split('=')[1])
    return out

if __name__ == "__main__":
    root_dir = '/media/sf_cs224d/cs224d-project/'
    # Pass the directory of the MPQA as a command line argument.
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]

    file_by_topic = '/media/sf_cs224d/cs224d-project/database.mpqa.1.2/Release-Full-ByTopic'
    if len(sys.argv) > 2:
        file_by_topic = sys.argv[2]

    argentina_filenames = get_argentina_filenames(file_by_topic)
    print len(argentina_filenames)
    print argentina_filenames

    docs_dir = os.path.join(root_dir, 'data_tools', 'parsed')
    dir_list = os.listdir(docs_dir)
    #Create a set of all words
    all_words = set()
    vocab = Vocab()
    count_files = 0
    for dir_ in dir_list:
        print 'In directory:', dir_
        curr_dir = os.path.join(docs_dir, dir_)
        file_list = os.listdir(curr_dir)
	print curr_dir	
        for fname in file_list:
            count_files += 1
            full_fname = os.path.join(curr_dir, fname)
            f = open(full_fname, 'r')
	    for line in f:
		sp_line = line.split()
		word = sp_line[0]
                all_words.add(word) #Add word to set 
		vocab.add_word(word)
    print "Total number of unique words (case sensitive):", len(all_words)
    print "Total number of parsed files looked at:", count_files
    
    dir_for_all_words = os.path.join(root_dir, 'data_tools')
    file_for_all_words = os.path.join(dir_for_all_words, 'all_words.txt')
    all_words_f = open(file_for_all_words, 'w')
    for word in all_words:
	all_words_f.write(word + '\n')

    glove_dir = os.path.join(root_dir, 'glove.840B.300d')
    glove_f = open(os.path.join(glove_dir, 'glove.840B.300d.txt'), 'r')
    embedding_matrix = np.zeros((len(vocab.word_to_index),300))
    
    
    count = 0
    for line in glove_f:
	line_sp = line.split()
    	word = line_sp[0]
        line_sp_vec = [float(line_num) for line_num in line_sp[1:]] 
        if word in vocab.word_to_index:
	    index = vocab.word_to_index[word]
	    embedding_matrix[index,:] = line_sp_vec
    	    #print "Word found that matches list of words in vocab!"
	    #print "The word is ", word
	count += 1
	if count%5000 == 0:
		print "Current word:" , word
		print count
   
    
    np.save('embedding_matrix', embedding_matrix)
    pickle.dump(vocab, open('vocab.pickle','w'))
    	     





    
'''
with open(full_fname, 'r') as f:
	doc_str = f.read()
            anns_subsubdir = os.path.join(anns_subdir, fname)
            anns_fname = os.path.join(anns_subsubdir, 'gateman.mpqa.lre.1.2')
            with open(anns_fname, 'r') as ann_f:
                # print anns_fname
                annotations = []
                for line in ann_f:
                    if line.startswith('#'):  # Ignore comment lines
                        continue
                    sp_line = line.split('\t')
                    ann = sp_line[3]
                    if ann not in ['GATE_direct-subjective',
                                   'GATE_expressive-subjectivity']:
                        continue
                    start, end = [int(a) for a in sp_line[1].split(',')]
                    extra_info = sp_line[4]
                    annotations.append((start, end, ann, doc_str[start:end],
                                        extra_info))
                annotations = sorted(annotations, key=lambda x: x[0])
                words_with_offsets = split_with_start_end_byte(doc_str)
                outputs = tag_words_with_ann(words_with_offsets, annotations)
                with open(os.path.join(out_dir, fname), 'w') as out_f:
                    for out in outputs:
                        out_f.write('\t'.join([str(o) for o in out]) + '\n')
'''
