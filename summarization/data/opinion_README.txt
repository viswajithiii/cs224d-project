Movie Review and Online Argument Corpus (first released on April 2016)
URL: http://www.ccs.neu.edu/home/luwang/

This corpus is distributed together with:

Neural Network-Based Abstract Generation for Opinions and Arguments
Lu Wang and Wang Ling.
Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2016.


==== Content ====

I. Description of the datasets
II. Contact


==== I. Description of the datasets ====

There are two main datasets under directory /PATH/TO/opinion_abstracts:

1) RottenTomatoes: The movie critics and consensus crawled from http://rottentomatoes.com/

2) IDebate: The arguments crawled from http://idebate.org/

======== 1) RottenTomatoes ========

We collect professional critics (usually in the form of one sentence selected by rottentomatoes' editors) and their consensus (constructed by editors) for 3,731 movies. 

File rottentomatoes.json contains a list of objects, where each object corresponds to a movie. It has fields of "_movie_name", "_movie_id", "_critics", and "_critic_consensus".


======== 2) IDebate ========

We collect 2,259 claims for 676 debates.

File idebate.json contains a list of objects, where each object corresponds to a claim with a set of arguments relevant to that claim. It has fields of "_debate_name", "_debate_id", "_claim", "_claim_id", "_argument_sentences".



==== II. Contact ====

Should you have any questions, please contact luwang@ccs.neu.edu (Lu Wang).



