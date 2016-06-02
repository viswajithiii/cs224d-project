import json
import random
import pickle

# RottenTomatoes files

f_rt = open('rottentomatoes.json', 'r')
rt_dump = json.load(f_rt)

# Create set of debates
set_movies = set(movie['_movie_id'] for movie in rt_dump)


# Total number of movies: 3731
# Use different movies for train, test, and val.
nMovies = len(set_movies)
print nMovies
nTrain = 2985
nVal = 373
nTest = 373
assert nMovies == nTrain + nVal + nTest
train_movies = set(random.sample(set_movies, nTrain))
set_movies = set_movies - train_movies
test_movies = set(random.sample(set_movies , nTest))
set_movies = set_movies - test_movies
val_movies = set_movies

# Each debate has multiple claims
# Total number of claims: 2259

test_movies = [movie for movie in rt_dump if movie['_movie_id']
               in test_movies]
val_movies = [movie for movie in rt_dump if movie['_movie_id']
               in val_movies]
train_movies = [movie for movie in rt_dump if movie['_movie_id']
                in train_movies]
assert len(test_movies) + len(train_movies) + len(val_movies) == len(
    rt_dump)

pickle.dump(train_movies, open('train_movies.pickle', 'w'))
pickle.dump(val_movies, open('val_movies.pickle', 'w'))
pickle.dump(test_movies, open('test_movies.pickle', 'w'))
