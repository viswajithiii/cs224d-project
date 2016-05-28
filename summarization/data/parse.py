import pickle
import nltk

for data in ['test', 'train', 'val']:
    filename = data + '_tokens.txt'
    f = open(filename, 'w')
    curr_claims = pickle.load(open(data + '_claims.pickle', 'r'))
    for claim in curr_claims:
        arg_ids = claim['_argument_sentences'].keys()
        arg_nums = [int(arg_num.split('_')[-1]) for arg_num in arg_ids]
        sorted_arg_nums, sorted_arg_ids = zip(*sorted(zip(arg_nums, arg_ids)))
        # print sorted_arg_nums
        # print sorted_arg_ids
        for arg_id in sorted_arg_ids:  
            tokens = nltk.word_tokenize(claim['_argument_sentences'][arg_id].encode("ascii", "ignore"))
            f.write(' '.join(tokens) + ' ')
        f.write('\t')
        f.write(' '.join(nltk.word_tokenize(claim['_claim'].encode("ascii", "ignore"))))
        f.write('\n')
    f.close()


