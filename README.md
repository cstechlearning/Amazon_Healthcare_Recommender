# Amazon_Healthcare_Recommender

Author: G.X.

https://jerryguangxu.github.io/Amazon_Healthcare_Recommender/



Steps:

Applied NLTK to process raw text for 50000 Amazon healthcare products including customer reviews, including noun singularization, removing stop-words, etc.
Keywords extraction from text for each product using RAKE algorithm. in order to build up simple search engine.
Product recommendation by determining product similarity via latent semantic indexing (LSI) algorithm using Gensim library.
Created a browser-based front end for product search and recommendation.
limitation:

Since this is no-server search engine, with all searchable contents included in .json file, so the number of products are quite limited.
