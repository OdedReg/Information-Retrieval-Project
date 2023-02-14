# Information-Retrieval-Project
Search engine on 6 million documents from Wikipedia

# Final-Project

Intro
=====
In this project we built a search engine for 6 million (approximetely) of English Wikipedia.

Content
==========

<!--ts-->
- [search_frontend](#search_frontend)
- [create_indexes](#create_indexes)
<!--te-->


## search_frontend
This file contains 5 search methods that process queries and return results from the entire Wikipedia corpus:

- `search` - The main method, returns up to a 100 of our best search results for the query. The search results ordered from best to worst where each element is a tuple (wiki_id, title).

- `search_body` - A method that returns up to 100 search results for the query, using TF-IDF and Cosine Similarity on the body of the articles.

- `search_title` - A method that returns all search results that contain a query word in the title of articles, ordered in descending order of the number of distinct query words that appear in the title.

- `search_anchor` - A method that returns all search results that contain a query word in the anchor text of articles, ordered in
descending order of the the number of query words that appear in anchor text linking to the page.

- `get_pagerank` - A method that returns PageRank values for a list of provided wiki article IDs.

- `get_pageview` - A method that returns the number of page views for a list of provided wiki article IDs.

## create_indexes

- `create_index` - The main method to create inverted index instance for title/body/anchor.

- `create_index_for_5_func` - A method to create inverted index instance according to the required rules for the project (without stemming and with a given regex).
