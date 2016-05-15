import sys
import os
from pprint import pprint
import re
from copy import deepcopy


def process_single_quotes(words):
    """
    Processes single quotes. Removes them
    from the beginning and the end.
    """

    new_words = []

    append_single_quote = False
    for i in range(len(words)):
        word = deepcopy(words[i])
        if "'" in word and len(word) > 1:
            if word[0] == "'":
                new_words.append(word[0])
                word = word[1:]
            if word[-1] == "'":
                append_single_quote = True
                word = word[:-1]
            if len(word) > 2 and word[-2] == "'":
                new_words.append(word[:-2])
                new_words.append(word[-2:])
            else:
                new_words.append(word)
            if append_single_quote:
                new_words.append("'")
                append_single_quote = False
        else:
            new_words.append(word)
    return new_words


def split_with_start_end_byte(s):
    """
    Splits the string s on white space,
    and which every substring, it outputs the
    start and end byte in the original string s.

    Start byte inclusive, end byte exclusive.
    """
    s.replace('`', "'")
    words = re.findall(r"[\w']+|[.,!?;\"]", s)
    words = process_single_quotes(words)

    offsets = []
    running_offset = 0
    for word in words:
        word_offset = s.index(word, running_offset)
        word_len = len(word)
        running_offset = word_offset + word_len
        offsets.append((word, word_offset, running_offset))

    return offsets


def tag_words_with_ann(words_with_offsets, annotations):

    ann_dict = {'GATE_direct-subjective': 'DSE',
                'GATE_expressive-subjectivity': 'ESE'
                }

    in_annotation = None
    curr_ann_idx = 0
    outputs = []
    for word, start, end in words_with_offsets:
        if curr_ann_idx >= len(annotations):
            outputs.append((word, 'O'))
            continue
        if in_annotation is None:
            if annotations[curr_ann_idx][0] <= start:
                in_annotation = ann_dict[annotations[curr_ann_idx][2]]
                tag = 'B' + in_annotation
            else:
                tag = 'O'
        else:
            tag = 'I' + in_annotation
        if annotations[curr_ann_idx][1] <= end:
            in_annotation = None
            curr_ann_idx += 1
            while curr_ann_idx < len(annotations) and \
                    annotations[curr_ann_idx][0] >= annotations[
                                                        curr_ann_idx][1] - 1:
                # This while skips the nonsense annotations with zero or one
                # character.
                curr_ann_idx += 1

        outputs.append((word, tag))
    return outputs


if __name__ == "__main__":
    split_with_start_end_byte("This is 'the' shit's")
    root_dir = '/media/sf_kickstarter/CS224D/Project/database.mpqa.1.2'
    # Pass the directory of the MPQA as a command line argument.
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]

    docs_dir = root_dir + '/docs'
    anns_dir = root_dir + '/man_anns'
    if not os.path.exists('./parsed'):
        os.makedirs('./parsed')
    out_root_dir = './parsed'

    dir_list = os.listdir(docs_dir)
    for dir_ in dir_list:
        print 'In directory:', dir_
        docs_subdir = os.path.join(docs_dir, dir_)
        anns_subdir = os.path.join(anns_dir, dir_)
        out_dir = os.path.join(out_root_dir, dir_)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        file_list = os.listdir(docs_subdir)
        for fname in file_list:
            docs_fname = os.path.join(docs_subdir, fname)
            with open(docs_fname, 'r') as f:
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
