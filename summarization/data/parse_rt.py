import pickle
import nltk
import random


N_CLAIMS_PER_MOVIE = 10

for data in ['test', 'train', 'val']:
    filename = data + '_tokens_rt.txt'
    f = open(filename, 'w')
    curr_movies = pickle.load(open(data + '_movies.pickle', 'r'))
    for movie in curr_movies:
        arg_ids = movie['_critics'].keys()
        chosen_arg_ids = random.sample(arg_ids, min(N_CLAIMS_PER_MOVIE, len(arg_ids)))
        f.write(' '.join(nltk.word_tokenize(movie['_movie_name'].encode("ascii", "ignore").replace('_',' '))) + ' ')
        for arg_id in chosen_arg_ids:
            tokens = nltk.word_tokenize(movie['_critics'][arg_id].encode("ascii", "ignore"))
            f.write(' '.join(tokens) + ' ')
        f.write('\t')
        f.write(' '.join(nltk.word_tokenize(movie['_critic_consensus'].encode("ascii", "ignore"))))
        f.write('\n')
    f.close()
