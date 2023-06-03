# The cluster security option "Allow API access to all Google Cloud services"
# under Manage Security → Project Access when setting up the cluster
!gcloud dataproc clusters list --region us-central1

# Imports & Setup
!pip install -q google-cloud-storage==1.43.0
!pip install -q graphframes

import nltk
import pyspark
import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
import builtins
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm
import operator
from contextlib import closing
import json
from io import StringIO
import matplotlib.pyplot as plt
import math
import hashlib
from functools import reduce
from operator import add

!ls -l /usr/lib/spark/jars/graph*

from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf, SparkFiles
from pyspark.sql import SQLContext
from graphframes import *

# install ngrok to emulate public IP / address
!wget -N https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip -O ngrok-stable-linux-amd64.zip
!unzip -u ngrok-stable-linux-amd64.zip
!./ngrok authtoken 2JidbvmcPvLbBy9Dk8OkRfpco9S_2sH521Z4ghkATHtEJiWRt

# install a ngrok python package and a version of flask that works with it in
# colab
!pip -q install flask-ngrok
!pip -q install flask==0.12.2
!pip -q install flask_restful

# download nltk stopwords
nltk.download('stopwords')

# Put your bucket name below and make sure you can access it without an error

bucket_name = ''
full_path = f"gs://{bucket_name}/"
paths=[]

client = storage.Client()
blobs = client.list_blobs(bucket_name)
for b in blobs:
    if '.parquet' in b.name:
        paths.append(full_path+b.name)


parquetFile = spark.read.parquet(*paths)

# Create Regex and Stemmer
BLOCK_SIZE = 1999998
NUM_BUCKETS = 124

english_stopwords = frozenset(stopwords.words('english'))

website_regx = "\\b((?:www\.)?\w+\.com)\\b"
dates_with_slash = "\\b((?:0?[1-9]|[12][0-9]|3[01])\/(?:0[1-9]|1[0-2])(?:\/\d{4})?)"
date_regex = "((?:January|March|April|May|June|July|August|September|October|November|December|Jan|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s(?:0?[1-9]|[12][0-9]|3[01])(?:\,\s)(?:[12][0-9]{3}))|((?:0?[1-9]|[12][0-9]|3[01])\s(?:Jan(?:uary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s(?:(?:[12][0-9]{3})|\d{3}))|((?:February|Feb)\s(?:0?[1-9]|[12][0-9])(?:\,\s)(?:[12][0-9]{3}))|((?:0?[1-9]|[12][0-9])\s(?:Feb)\s(?:[12][0-9]{3}))"
time_regex = "(\\b(?:1[012]|0?[1-9])\.[0-5][0-9](?:AM|PM))\\b|(\\b(?:1[012]|0?[1-9])(?:\:)?[0-5][0-9](?:\s?)(?:a\.m(?:\.)?|p\.m(?:\.)?|am|pm))|(\\b(?:2[0-3]|[01]?[0-9])\:(?:[0-5]?[0-9])(?:\:(?:[0-5]?[0-9]))?\\b)"
number_regex = "(?<![\w+\+\-\.\,])([+-]?\d{1,3}(?:\,\d{3})*(?:\.\d+)?)(?!\d*\%|\d*\.[a-zA-Z]+|\d*\,\S|\d*\.\S|\d*[a-zA-Z]+|\d)"
precent_regex = "(?<![\w+\+\-\.\,])([+-]?\d{1,3}(?:\,?\d{3})*(?:\.\d+)?\%)(?!\d*\%|\d*\.[a-zA-Z]+|\d*\,\S|\d*\.\S|\d*[a-zA-Z]+)"
word_like_usa_regex = "\\b([A-Za-z]\.[A-Za-z]\.[A-Za-z])\\b|\\b([A-Za-z]\.[A-Za-z]\.(?![\w]))"
word_regex ="(?<![\-\w])(\w+(?:\w*\-?)*\\'?\w*)(?!\w*\%)"
html_regex = "<(“[^”]*”|'[^’]*’|[^'”>])*>"

RE_WORD = re.compile(r"\b((?:www\.)?\w+\.com)\b|\b((?:0?[1-9]|[12][0-9]|3[01])\/(?:0[1-9]|1[0-2])(?:\/\d{4})?)|((?:January|March|April|May|June|July|August|September|October|November|December|Jan|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s(?:0?[1-9]|[12][0-9]|3[01])(?:\,\s)(?:[12][0-9]{3}))|((?:0?[1-9]|[12][0-9]|3[01])\s(?:Jan(?:uary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s(?:(?:[12][0-9]{3})|\d{3}))|((?:February|Feb)\s(?:0?[1-9]|[12][0-9])(?:\,\s)(?:[12][0-9]{3}))|((?:0?[1-9]|[12][0-9])\s(?:Feb)\s(?:[12][0-9]{3}))|(\b(?:1[012]|0?[1-9])\.[0-5][0-9](?:AM|PM))\b|(\b(?:1[012]|0?[1-9])(?:\:)?[0-5][0-9](?:\s?)(?:a\.m(?:\.)?|p\.m(?:\.)?|am|pm))|(\b(?:2[0-3]|[01]?[0-9])\:(?:[0-5]?[0-9])(?:\:(?:[0-5]?[0-9]))?\b)|(?<![\w+\+\-\.\,])([+-]?\d{1,3}(?:\,\d{3})*(?:\.\d+)?)(?!\d*\%|\d*\.[a-zA-Z]+|\d*\,\S|\d*\.\S|\d*[a-zA-Z]+|\d)|(?<![\w+\+\-\.\,])([+-]?\d{1,3}(?:\,?\d{3})*(?:\.\d+)?\%)(?!\d*\%|\d*\.[a-zA-Z]+|\d*\,\S|\d*\.\S|\d*[a-zA-Z]+)|\b([A-Za-z]\.[A-Za-z]\.[A-Za-z])\b|\b([A-Za-z]\.[A-Za-z]\.(?![\w]))|(?<![\-\w])(\w+(?:\w*\-?)*\'?\w*)(?!\w*\%)|<(“[^”]*”|'[^’]*’|[^'”>])*>", re.UNICODE)
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
all_stopwords = english_stopwords.union(corpus_stopwords)
stemmer = PorterStemmer()

RE_WORD_5_Func = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


# get PageViews
def GetPageViews(p):
    '''
    create a dictionary with doc_id: page_views and write it in a pickle file.

    Parameters:
  -----------
    p: path of file that includes the page views
    '''
    pv_name = p.name
    pv_temp = f'{p.stem}-4dedup.txt'
    pv_clean = f'{p.stem}.pkl'
    # Download the file (2.3GB)
    !wget -N $pv_path
    # Filter for English pages, and keep just two fields: article ID (3) and monthly
    # total number of page views (5). Then, remove lines with article id or page
    # view values that are not a sequence of digits.
    !bzcat $pv_name | grep "^en\.wikipedia" | cut -d' ' -f3,5 | grep -P "^\d+\s\d+$" > $pv_temp
    # Create a Counter (dictionary) that sums up the pages views for the same
    # article, resulting in a mapping from article id to total page views.
    wid2pv = Counter()
    with open(pv_temp, 'rt') as f:
      for line in f:
        parts = line.split(' ')
        wid2pv.update({int(parts[0]): int(parts[1])})
    # write out the counter as binary file (pickle it)
    with open(pv_clean, 'wb') as f:
      pickle.dump(wid2pv, f)


# get PageRank
def generate_graph(pages):
  ''' Compute the directed graph generated by wiki links.
  Parameters:
  -----------
    pages: RDD
      An RDD where each row consists of one wikipedia articles with 'id' and
      'anchor_text'.
  Returns:
  --------
    edges: RDD
      An RDD where each row represents an edge in the directed graph created by
      the wikipedia links. The first entry should the source page id and the
      second entry is the destination page id. No duplicates should be present.
    vertices: RDD
      An RDD where each row represents a vetrix (node) in the directed graph
      created by the wikipedia links. No duplicates should be present.
  '''

  edges = pages.map(lambda x: {(x[0],y[0]) for y in x[1]}).flatMap(lambda x: x)
  vertices = pages.map(lambda x: {(y[0],y[0]) for y in [(x[0], x[0])] + x[1]}).flatMap(lambda x: x).reduceByKey(lambda x,y: x)
  return edges, vertices


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


def token2bucket_id(token):
    """ hash token to bucket_id """
    return int(_hash(token), 16) % NUM_BUCKETS


# Writer and reader classes for the files
class MultiFileWriter:
    """ Sequential binary writer to multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, name, bucket_name, prefix):
        self._base_dir = Path(base_dir)
        self._name = name
        self._file_gen = (open(self._base_dir / f'{name}_{i:03}.bin', 'wb')
                          for i in itertools.count())
        self._f = next(self._file_gen)
        # Connecting to google storage bucket.
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.prefix = prefix

    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            # if the current file is full, close and open a new one.
            if remaining == 0:
                self.upload_to_gcp()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((self._f.name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()

    def upload_to_gcp(self):
        '''
            The function saves the posting files into the right bucket in google storage.
        '''
        self._f.close()
        file_name = self._f.name
        blob = self.bucket.blob(self.prefix + "/" + f"{file_name}")
        blob.upload_from_filename(file_name)


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self):
        self._open_files = {}

    def read(self, locs, n_bytes, prefix):
        b = []
        client = storage.Client()
        bucket = client.get_bucket('oded_318963386')
        for f_name, offset in locs:
            blob = bucket.get_blob(prefix + f'/{f_name}')
            if blob == None:
                continue
            pl_bin = blob.download_as_bytes()
            pl_to_read = pl_bin[offset: builtins.min(offset + n_bytes, BLOCK_SIZE)]
            n_read = builtins.min(n_bytes, BLOCK_SIZE - offset)
            b.append(pl_to_read)
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


TUPLE_SIZE = 6  # We're going to pack the doc_id and tf values in this
# many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
DL = Counter()


class InvertedIndex:
    def __init__(self, docs={}):
        """ Initializes the inverted index and add documents to it (if provided).
        Parameters:
        -----------
          docs: dict mapping doc_id to list of tokens
        """
        # stores document frequency per term
        self.df = Counter()
        # stores total frequency per term
        self.term_total = Counter()
        # stores posting list per term while building the index (internally),
        # otherwise too big to store in memory.
        self._posting_list = defaultdict(list)
        # mapping a term to posting file locations, which is a list of
        # (file_name, offset) pairs. Since posting lists are big we are going to
        # write them to disk and just save their location in this list. We are
        # using the MultiFileWriter helper class to write fixed-size files and store
        # for each term/posting list its list of locations. The offset represents
        # the number of bytes from the beginning of the file where the posting list
        # starts.
        self.posting_locs = defaultdict(list)
        self.doc_to_norm = {}
        self.DL_for_index = {}
        self.len_DL_for_index = 0

        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        """ Adds a document to the index with a given `doc_id` and tokens. It counts
            the tf of tokens, then update the index (in memory, no storage
            side-effects).
        """
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name):
        """ Write the in-memory index to disk. Results in the file:
            (1) `name`.pkl containing the global term stats (e.g. df).
        """
        #### GLOBAL DICTIONARIES ####
        self._write_globals(base_dir, name)

    def _write_globals(self, base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)

    def __getstate__(self):
        """ Modify how the object is pickled by removing the internal posting lists
            from the object's state dictionary.
        """
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    def posting_lists_iter(self):
        """ A generator that reads one posting list from disk and yields
            a (word:str, [(doc_id:int, tf:int), ...]) tuple.
        """
        with closing(MultiFileReader()) as reader:
            for w, locs in self.posting_locs.items():
                b = reader.read(locs[0], self.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(self.df[w]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
                yield w, posting_list

    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def delete_index(base_dir, name):
        path_globals = Path(base_dir) / f'{name}.pkl'
        path_globals.unlink()
        for p in Path(base_dir).rglob(f'{name}_*.bin'):
            p.unlink()

    @staticmethod
    def write_a_posting_list(b_w_pl, bucket_name, prefix):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl

        with closing(MultiFileWriter(".", bucket_id, bucket_name, prefix)) as writer:
            for w, pl in list_w_pl:
                # convert to bytes
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                # write to file(s)
                locs = writer.write(b)
                # save file locations to index
                posting_locs[w].extend(locs)
            writer.upload_to_gcp()
            InvertedIndex._upload_posting_locs(bucket_id, posting_locs, bucket_name, prefix)
        return bucket_id

    @staticmethod
    def _upload_posting_locs(bucket_id, posting_locs, bucket_name, prefix):
        with open(f"{bucket_id}_posting_locs.pickle", "wb") as f:
            pickle.dump(posting_locs, f)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_posting_locs = bucket.blob(prefix + f"/{bucket_id}_posting_locs.pickle")
        blob_posting_locs.upload_from_filename(f"{bucket_id}_posting_locs.pickle")



# BM25 and TF-IDF classes

def sum_dict_values(dict_values):
    return reduce(add, dict_values.values())


class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    index: inverted index
    """

    def __init__(self, index, DL, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.DL = DL
        self.N = len(self.DL)
        self.AVGDL = sum_dict_values(self.DL) / self.N
        self.words = list(self.index.term_total.keys())

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, queries, prefix, N=100):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
        d = {}
        for key in queries:
            Q = queries[key]
            self.idf = self.calc_idf(Q)
            for term in np.unique(Q):
                if term in self.words:
                    list_of_doc = read_posting_list(self.index, term, prefix)
                    for doc_id, freq in list_of_doc:
                        d[doc_id] = d.get(doc_id, 0) + self._score(term, doc_id, freq)
        return d

    def _score(self, term, doc_id, freq):
        """
        This function calculate the bm25 score for given query and document.

        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.

        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        doc_len = self.DL[str(doc_id)]
        numerator = self.idf[term] * freq * (self.k1 + 1)
        denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
        score += (numerator / denominator)
        return score

class tf_idf_from_index:
    """
    Best Match tf-idf.
    ----------

    index: inverted index
    """

    def __init__(self, index):
        self.index = index
        self.words = list(self.index.term_total.keys())
        self.doc_to_norm = self.index.doc_to_norm
        self.DL_for_index = self.index.DL_for_index
        self.len_DL_for_index = self.index.len_DL_for_index


# Functions for create_index

def tokenizer_5_func(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [token.group() for token in RE_WORD_5_Func.finditer(text.lower()) if
                      token.group() not in all_stopwords]
    return list_of_tokens


# tokenize assignment 4
def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [stemmer.stem(token.group()) for token in RE_WORD.finditer(text.lower()) if
                      token.group() not in all_stopwords]
    return list_of_tokens


def tokenize_body(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [stemmer.stem(token.group()) for token in RE_WORD_5_Func.finditer(text.lower()) if
                      token.group() not in all_stopwords]
    return list_of_tokens


def get_tokenize_words(text, id):
  ''' Count the length of `text` that is not included in
  `all_stopwords` and returns a tuple with doc_id: len(doc_id):
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    tuple with doc_id: len(doc_id)
  '''
  tokens = tokenize(text)
  return (id, tokens)


def calc_text_length(text, id):
  ''' Count the length of `text` that is not included in
  `all_stopwords` and returns a tuple with doc_id: len(doc_id):
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    tuple with doc_id: len(doc_id)
  '''
  tokens = tokenize(text)
  return (id, len(tokens))


def word_count(text, id):
  ''' Count the frequency of each word in `text` (tf) that is not included in
  `all_stopwords` and return entries that will go into our posting lists.
  Parameters:
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    List of tuples
      A list of (token, (doc_id, tf)) pairs
      for example: [("Anarchism", (12, 5)), ...]
  '''
  tokens = tokenize(text)
  # YOUR CODE HERE
  d = {}
  for word in tokens:
      d[word] = d.get(word, 0) + 1

  l = []
  for key, val in d.items():
    l.append((key,(id,val)))
  return l

def word_count_body(text, id):
  ''' Count the frequency of each word in `text` (tf) that is not included in
  `all_stopwords` and return entries that will go into our posting lists.
  Parameters:
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    List of tuples
      A list of (token, (doc_id, tf)) pairs
      for example: [("Anarchism", (12, 5)), ...]
  '''
  tokens = tokenize_body(text)
  # YOUR CODE HERE
  d = {}
  for word in tokens:
      d[word] = d.get(word, 0) + 1

  l = []
  for key, val in d.items():
    l.append((key,(id,val)))
  return l

def calc_text_length_for_5_func(text, id):
  ''' Count the length of `text` that is not included in
  `all_stopwords` and returns a tuple with doc_id: len(doc_id):
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    tuple with doc_id: len(doc_id)
  '''
  tokens = tokenizer_5_func(text)
  return (id, len(tokens))


def word_count_for_5_func(text, id):
  ''' Count the frequency of each word in `text` (tf) that is not included in
  `all_stopwords` and return entries that will go into our posting lists.
  Parameters:
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    List of tuples
      A list of (token, (doc_id, tf)) pairs
      for example: [("Anarchism", (12, 5)), ...]
  '''
  tokens = tokenizer_5_func(text)
  # YOUR CODE HERE
  d = {}
  for word in tokens:
      d[word] = d.get(word, 0) + 1

  l = []
  for key, val in d.items():
    l.append((key,(id,val)))
  return l

def reduce_word_counts(unsorted_pl):
  ''' Returns a sorted posting list by wiki_id.
  Parameters:
  -----------
    unsorted_pl: list of tuples
      A list of (wiki_id, tf) tuples
  Returns:
  --------
    list of tuples
      A sorted posting list.
  '''
  # YOUR CODE HERE
  return sorted(unsorted_pl, key = lambda x: x[0])

def calculate_df(postings):
  ''' Takes a posting list RDD and calculate the df for each token.
  Parameters:
  -----------
    postings: RDD
      An RDD where each element is a (token, posting_list) pair.
  Returns:
  --------
    RDD
      An RDD where each element is a (token, df) pair.
  '''
  # YOUR CODE HERE
  return postings.map(lambda x: (x[0],int(np.sum([1 for tup in x[1]]))))


def calculate_term_total(postings):
  ''' Takes a posting list RDD and calculate the df for each token.
  Parameters:
  -----------
    postings: RDD
      An RDD where each element is a (token, posting_list) pair.
  Returns:
  --------
    RDD
      An RDD where each element is a (token, term_total_frequency) pair.
  '''
  # YOUR CODE HERE
  return postings.map(lambda x: (x[0],int(np.sum([tup[1] for tup in x[1]]))))


def partition_postings_and_write(postings, i, prefix):
  ''' A function that partitions the posting lists into buckets, writes out
  all posting lists in a bucket to disk, and returns the posting locations for
  each bucket. Partitioning should be done through the use of `token2bucket`
  above. Writing to disk should use the function  `write_a_posting_list`, a
  static method implemented in inverted_index_colab.py under the InvertedIndex
  class.
  Parameters:
  -----------
    postings: RDD
      An RDD where each item is a (w, posting_list) pair.
  Returns:
  --------
    RDD
      An RDD where each item is a posting locations dictionary for a bucket. The
      posting locations maintain a list for each word of file locations and
      offsets its posting list was written to. See `write_a_posting_list` for
      more details.
  '''
  # YOUR CODE HERE
  list_of_words_per_bucket = postings.map(lambda x: (token2bucket_id(x[0]), (x[0],x[1])))
  buckets = list_of_words_per_bucket.groupByKey().map(lambda x: InvertedIndex.write_a_posting_list(x, bucket_name, prefix))
  return buckets

def doc_to_norm_body(text, id, DL_for_index, len_DL, w2df_dict):
    ''' Calculate the document norm.
      Parameters:
      -----------
        text: str
          Text of one document
        id: int
          Document id
        DL_for_index: dict
          Documents length dictionary
        len_DL: int
          DL length
        w2df_dict: dict
          A dictionary with words as keys and document frequencies as values
      Returns:
      --------
        Tuple (id, norm(document))
      '''
    tokens = tokenizer_5_func(text)
    tf = {}
    if len(tokens) == 0:
        return id, 0
    for term in tokens:
        tf[term] = tf.get(term, 0) + 1
    doc_length = DL_for_index[id]
    list_of_tf_idf = [reduce_doc_to_norm(tf, doc_length, len_DL, w2df_dict[term]) for term, tf in tf.items() if term in w2df_dict]
    return id, np.linalg.norm(list_of_tf_idf)

def doc_to_norm(text, id, DL_for_index, len_DL, w2df_dict):
    ''' Calculate the document norm.
      Parameters:
      -----------
        text: str
          Text of one document
        id: int
          Document id
        DL_for_index: dict
          Documents length dictionary
        len_DL: int
          DL length
        w2df_dict: dict
          A dictionary with words as keys and document frequencies as values
      Returns:
      --------
        Tuple (id, norm(document))
      '''
    tokens = tokenizer_5_func(text)
    tf = {}
    if len(tokens) == 0:
        return id, 0
    for term in tokens:
        tf[term] = tf.get(term, 0) + 1
    doc_length = DL_for_index[id]
    list_of_tf_idf = [reduce_doc_to_norm(tf, doc_length, len_DL, w2df_dict[term]) for term, tf in tf.items()]
    return id, np.linalg.norm(list_of_tf_idf)

def reduce_doc_to_norm(tf, doc_length, len_DL, df):
    return (tf / doc_length) * math.log(len_DL / df, 10)

def save_DL(DL, name):
    with open(f'{name}.pkl', 'wb') as handle:
        pickle.dump(DL, handle)

def save_titles_dict(titles_dict, name):
    with open(f'{name}.pkl', 'wb') as handle:
        pickle.dump(titles_dict, handle)

def save_other(other, name):
    with open(f'{name}.pkl', 'wb') as handle:
        pickle.dump(other, handle)

def read_for_search_frontend(name):
    with open(f'{name}.pkl', 'rb') as handle:
      return pickle.load(handle)


# Create indexes

docs_length_dict_body = {}
docs_length_dict_title = {}
docs_length_dict_anchor = {}


def create_index(doc_pairs, index_name, i, filter_from_n_docs):
    # word counts map
    prefix = index_name
    word_counts = []
    if prefix == 'body':
        # tokenize without stemming
        word_counts = doc_pairs.flatMap(lambda x: word_count_body(x[0], x[1]))
    else:
        # complex tokenize and stemming
        word_counts = doc_pairs.flatMap(lambda x: word_count(x[0], x[1]))
    postings = word_counts.groupByKey().mapValues(reduce_word_counts)
    docs_length_dict = word_counts.map(lambda x: x[1]).reduceByKey(lambda a, b: a + b).collectAsMap()
    DL.update(Counter(docs_length_dict))
    if prefix == 'body':
        docs_length_dict_body = docs_length_dict
    elif prefix == 'title':
        docs_length_dict_title = docs_length_dict
    else:
        docs_length_dict_anchor = docs_length_dict

    # filtering postings and calculate df
    postings_filtered = postings.filter(lambda x: len(x[1]) > filter_from_n_docs)
    w2df = calculate_df(postings_filtered)
    w2df_dict = w2df.collectAsMap()

    w2term_total = calculate_term_total(postings_filtered)
    w2term_total_dict = w2term_total.collectAsMap()
    # partition posting lists and write out
    posting_locs_list = partition_postings_and_write(postings_filtered, i, prefix).collect()

    # merge the posting locations into a single dict and run more tests
    super_posting_locs = defaultdict(list)
    for blob in client.list_blobs(bucket_name, prefix=f'{prefix}'):
        if not blob.name.endswith("pickle"):
            continue
        with blob.open("rb") as f:
            posting_locs = pickle.load(f)
            for k, v in posting_locs.items():
                super_posting_locs[k].extend(v)

    # Create inverted index instance
    inverted = InvertedIndex()
    # Adding the posting locations dictionary to the inverted index
    inverted.posting_locs = super_posting_locs
    # Add the token - df dictionary to the inverted index
    inverted.df = w2df_dict
    inverted.term_total = w2term_total_dict
    inverted.doc_to_norm = {}
    inverted.DL_for_index = {}
    inverted.len_DL_for_index = 0
    # write the global stats out
    inverted.write_index('.', index_name)
    return inverted


dict_doc_to_norm_body = {}
dict_doc_to_norm_title = {}
dict_doc_to_norm_anchor = {}


def create_index_for_5_func(doc_pairs, index_name, i, filter_from_n_docs):
    # word counts map
    prefix = index_name
    word_counts = doc_pairs.flatMap(lambda x: word_count_for_5_func(x[0], x[1]))
    postings = word_counts.groupByKey().mapValues(reduce_word_counts)
    docs_length_dict = word_counts.map(lambda x: x[1]).reduceByKey(lambda a, b: a + b).collectAsMap()
    DL.update(Counter(docs_length_dict))
    # filtering postings and calculate df
    postings_filtered = postings.filter(lambda x: len(x[1]) > filter_from_n_docs)
    w2df = calculate_df(postings_filtered)
    w2df_dict = w2df.collectAsMap()

    w2term_total = calculate_term_total(postings_filtered)
    w2term_total_dict = w2term_total.collectAsMap()
    # partition posting lists and write out
    posting_locs_list = partition_postings_and_write(postings_filtered, i, prefix).collect()

    dict_doc_to_norm = {}
    if prefix == 'body_5_func':
        dict_doc_to_norm_body = docs_length_dict
    elif prefix == 'title_5_func':
        dict_doc_to_norm_title = docs_length_dict
    else:
        dict_doc_to_norm_anchor = docs_length_dict

    # merge the posting locations into a single dict and run more tests
    super_posting_locs = defaultdict(list)
    for blob in client.list_blobs(bucket_name, prefix=f'{prefix}'):
        if not blob.name.endswith("pickle"):
            continue
        with blob.open("rb") as f:
            posting_locs = pickle.load(f)
            for k, v in posting_locs.items():
                super_posting_locs[k].extend(v)

    # Create inverted index instance
    inverted = InvertedIndex()
    # Adding the posting locations dictionary to the inverted index
    inverted.posting_locs = super_posting_locs
    # Add the token - df dictionary to the inverted index
    inverted.df = w2df_dict
    inverted.term_total = w2term_total_dict
    inverted.doc_to_norm = dict_doc_to_norm
    inverted.DL_for_index = docs_length_dict
    inverted.len_DL_for_index = len(docs_length_dict)
    # write the global stats out
    inverted.write_index('.', index_name)
    return inverted


def get_titles_dict(parquetFile):
    return parquetFile.select("id", "title").rdd.collectAsMap()


def get_doc_pairs(parquetFile):
    doc_text_pairs = parquetFile.select("text", "id").rdd
    doc_title_pairs = parquetFile.select("title", "id").rdd
    data_anchor_pairs = parquetFile.select("anchor_text", "id").rdd

    data_anchor_pairs = data_anchor_pairs.map(lambda x: {(x[1], y[1]) for y in x[0]}).flatMap(
        lambda x: x).groupByKey().mapValues(lambda x: " ".join(x)).map(lambda x: (x[1], x[0]))

    return doc_text_pairs, doc_title_pairs, data_anchor_pairs

# get page views
pv_path = 'https://dumps.wikimedia.org/other/pageview_complete/monthly/2021/2021-08/pageviews-202108-user.bz2'
p = Path(pv_path)
GetPageViews(p)
# upload to gs
pageview_src = f'{p.stem}.pkl'
pageview_dst = f'gs://{bucket_name}/{pageview_src}'
!gsutil cp $pageview_src $pageview_dst

# get page rank
pages_links = spark.read.parquet("gs://wikidata20210801_preprocessed/*").select("id", "anchor_text").rdd
edges, vertices = generate_graph(pages_links)
edgesDF = edges.toDF(['src', 'dst']).repartition(124, 'src')
verticesDF = vertices.toDF(['id']).repartition(124, 'id')
g = GraphFrame(verticesDF, edgesDF)
pr_results = g.pageRank(resetProbability=0.15, maxIter=6)
pr = pr_results.vertices.select("id", "pagerank").rdd
prank = pr.collectAsMap()

# write out the PageRank dict as binary file (pickle it)
with open('pr.pkl', 'wb') as f:
    pickle.dump(prank, f)

# upload to gs
pagerank_src = 'pr.pkl'
pagerank_dst = f'gs://{bucket_name}/{pagerank_src}'
!gsutil cp $pagerank_src $pagerank_dst


titles_dict = get_titles_dict(parquetFile)
save_titles_dict(titles_dict, 'titles_dict')

# upload to gs
titles_dict_src = 'titles_dict.pkl'
titles_dict_dst = f'gs://{bucket_name}/{titles_dict_src}'
!gsutil
cp $titles_dict_src $titles_dict_dst

doc_text_pairs, doc_title_pairs, data_anchor_pairs = get_doc_pairs(parquetFile)

body_index = create_index(doc_text_pairs, 'body', 0, 50)
title_index = create_index(doc_title_pairs, 'title', 1, 0)
anchor_index = create_index(data_anchor_pairs, 'anchor', 2, 0)

save_DL(DL, 'DL')

bm25_body = BM25_from_index(body_index, DL)
bm25_title = BM25_from_index(title_index, DL)
bm25_anchor = BM25_from_index(anchor_index, DL)

save_other(docs_length_dict_body, 'docs_length_dict_body')
save_other(docs_length_dict_title, 'docs_length_dict_title')
save_other(docs_length_dict_anchor, 'docs_length_dict_anchor')

save_other(bm25_body, 'bm25_body')
save_other(bm25_title, 'bm25_title')
save_other(bm25_anchor, 'bm25_anchor')

# # upload to gs
src = 'docs_length_dict_body.pkl'
dst = f'gs://{bucket_name}/{src}'
!gsutil
cp $src $dst

src = 'docs_length_dict_title.pkl'
dst = f'gs://{bucket_name}/{src}'
!gsutil
cp $src $dst

src = 'docs_length_dict_anchor.pkl'
dst = f'gs://{bucket_name}/{src}'
!gsutil
cp $src $dst

DL_src = 'DL.pkl'
DL_dst = f'gs://{bucket_name}/{DL_src}'
!gsutil
cp $DL_src $DL_dst

src = 'body.pkl'
dst = f'gs://{bucket_name}/{src}'
!gsutil
cp $src $dst

src = 'title.pkl'
dst = f'gs://{bucket_name}/{src}'
!gsutil
cp $src $dst

src = 'anchor.pkl'
dst = f'gs://{bucket_name}/{src}'
!gsutil
cp $src $dst

src = 'bm25_body.pkl'
dst = f'gs://{bucket_name}/{src}'
!gsutil
cp $src $dst

src = 'bm25_title.pkl'
dst = f'gs://{bucket_name}/{src}'
!gsutil
cp $src $dst

src = 'bm25_anchor.pkl'
dst = f'gs://{bucket_name}/{src}'
!gsutil
cp $src $dst

# save inverted indexes for the 5 search function which need different tokenize function
DL = Counter()

body_index_5_func = create_index_for_5_func(doc_text_pairs, 'body_5_func', 3, 50)
title_index_5_func = create_index_for_5_func(doc_title_pairs, 'title_5_func', 4, 0)
anchor_index_5_func = create_index_for_5_func(data_anchor_pairs, 'anchor_5_func', 5, 0)

save_DL(DL, 'DL_5_func')

tf_idf_body = tf_idf_from_index(body_index_5_func)
tf_idf_title = tf_idf_from_index(title_index_5_func)
tf_idf_anchor = tf_idf_from_index(anchor_index_5_func)

save_other(tf_idf_body, 'tf_idf_body')
save_other(tf_idf_title, 'tf_idf_title')
save_other(tf_idf_anchor, 'tf_idf_anchor')

# upload to gs
DL_5_func_src = 'DL_5_func.pkl'
DL_5_func_dst = f'gs://{bucket_name}/{DL_5_func_src}'
!gsutil
cp $DL_5_func_src $DL_5_func_dst

src = 'body_5_func.pkl'
dst = f'gs://{bucket_name}/{src}'
!gsutil
cp $src $dst

src = 'title_5_func.pkl'
dst = f'gs://{bucket_name}/{src}'
!gsutil
cp $src $dst

src = 'anchor_5_func.pkl'
dst = f'gs://{bucket_name}/{src}'
!gsutil
cp $src $dst

src = 'tf_idf_body.pkl'
dst = f'gs://{bucket_name}/{src}'
!gsutil
cp $src $dst

src = 'tf_idf_title.pkl'
dst = f'gs://{bucket_name}/{src}'
!gsutil
cp $src $dst

src = 'tf_idf_anchor.pkl'
dst = f'gs://{bucket_name}/{src}'
!gsutil
cp $src $dst