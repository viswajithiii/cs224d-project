import sys
import os
from pprint import pprint

root_dir = '/media/sf_kickstarter/CS224D/Project/database.mpqa.1.2'
# Pass the directory of the MPQA as a command line argument.
if len(sys.argv) > 1:
    root_dir = sys.argv[1]

docs_dir = root_dir + '/docs'
anns_dir = root_dir + '/man_anns'

dir_list = os.listdir(docs_dir)
for dir_ in dir_list:
    docs_subdir = os.path.join(docs_dir, dir_)
    anns_subdir = os.path.join(anns_dir, dir_)
    file_list = os.listdir(docs_subdir)
    for fname in file_list:
        docs_fname = os.path.join(docs_subdir, fname)
        with open(docs_fname, 'r') as f:
            doc_str = f.read()
        anns_dir = os.path.join(anns_subdir, fname)
        anns_fname = os.path.join(anns_dir, 'gateman.mpqa.lre.1.2')
        with open(anns_fname, 'r') as ann_f:
            print anns_fname
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
            pprint(annotations[:20])
            curr_annotation = 0
            for i, byte in enumerate(doc_str):
                print repr(byte),'|',
            assert False
    break
