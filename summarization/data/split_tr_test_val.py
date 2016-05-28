import json
import random
import pickle

# IDebate files

f_idebate = open('idebate.json')
iDebateDump = json.load(f_idebate)

# Create set of debates
set_debates = set(claim['_debate_id'] for claim in iDebateDump)


# Total number of debates: 667
# Use different debates for train, test, and val.
nDebates = len(set_debates)
nTrain = 450
nVal = 67
nTest = 150
assert nDebates == nTrain + nVal + nTest
train_debates = set(random.sample(set_debates, nTrain))
set_debates = set_debates - train_debates
test_debates = set(random.sample(set_debates, nTest))
set_debates = set_debates - test_debates
val_debates = set_debates

# Each debate has multiple claims
# Total number of claims: 2259

test_claims = [claim for claim in iDebateDump if claim['_debate_id']
               in test_debates]
train_claims = [claim for claim in iDebateDump if claim['_debate_id']
                in train_debates]
val_claims = [claim for claim in iDebateDump if claim['_debate_id']
              in val_debates]
assert len(test_claims) + len(train_claims) + len(val_claims) == len(
    iDebateDump)

pickle.dump(train_claims, open('train_claims.pickle', 'w'))
pickle.dump(val_claims, open('val_claims.pickle', 'w'))
pickle.dump(test_claims, open('test_claims.pickle', 'w'))

