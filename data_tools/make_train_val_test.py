import sys
import os

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
    root_dir = '/media/sf_kickstarter/CS224D/Project/cs224d-project/data_tools/parsed'
    # Pass the directory of the MPQA as a command line argument.
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]


    file_by_topic = '/media/sf_kickstarter/CS224D/Project/database.mpqa.1.2/Release-Full-ByTopic'
    if len(sys.argv) > 2:
        file_by_topic = sys.argv[2]

    argentina_filenames = get_argentina_filenames(file_by_topic)
    print len(argentina_filenames)
    print argentina_filenames
    assert False
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
