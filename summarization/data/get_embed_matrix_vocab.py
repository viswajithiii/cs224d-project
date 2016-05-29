import sys
import os
from utils import Vocab
import numpy as np
import pickle


if __name__ == "__main__":

    #Create a set of all words
    all_words = set()
    vocab = Vocab()
    count_files = 0
    for name in ['test', 'train', 'val']:
        filename = name + '_tokens.txt'
        f = open(filename, 'r')
        for line in f:
            sp_line = line.strip().split()
            for token in sp_line:
                all_words.add(token)
                vocab.add_word(token)
        f.close()

    glove_dir = '/media/sf_kickstarter/CS224D/Project/glove.840B.300d'
    glove_f = open(os.path.join(glove_dir, 'glove.840B.300d.txt'), 'r')
    embedding_matrix = np.zeros((len(vocab.word_to_index),300))


    count = 0
    for line in glove_f:
        line_sp = line.strip().split()
        word = line_sp[0]
        line_sp_vec = [float(line_num) for line_num in line_sp[1:]]
        if word in vocab.word_to_index:
            line_sp_vec = [float(line_num) for line_num in line_sp[1:]]
            index = vocab.word_to_index[word]
            embedding_matrix[index,:] = line_sp_vec
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
