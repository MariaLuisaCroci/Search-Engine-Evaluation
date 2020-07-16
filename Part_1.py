import os, os.path
import whoosh
from whoosh import index
from whoosh.index import create_in
from whoosh.qparser import *
from whoosh import scoring
from whoosh.scoring import Frequency, TF_IDF, BM25F
from whoosh.fields import *
from whoosh.analysis import KeywordAnalyzer, SimpleAnalyzer, StandardAnalyzer, StemmingAnalyzer, FancyAnalyzer, NgramAnalyzer, LanguageAnalyzer
import csv
import time
from bs4 import BeautifulSoup
import re
import string
import urllib
import pandas as pd
import statistics
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data_path = 'Cranfield_DATASET\\'
doc_path = 'Cranfield_DATASET\\DOCUMENTS\\'


# creating a list of tuple analyzer = [(analyzer_object, 'analyzer_name')]

analyzer = [(KeywordAnalyzer(), 'KeywordAnalyzer'), (SimpleAnalyzer(), 'SimpleAnalyzer'),
            (StandardAnalyzer(), 'StandardAnalyzer'), (StemmingAnalyzer(), 'StemmingAnalyzer'),
            (FancyAnalyzer(), 'FancyAnalyzer'), (NgramAnalyzer(4), 'NgramAnalyzer'),
           (LanguageAnalyzer('en'), 'LanguageAnalyzer')]

# creating a list of tuple scoring_list = [('scoring_function', 'scoring_function_name')]
scoring_list = [('scoring.Frequency()', 'Frequency'), ('scoring.TF_IDF()', 'TF_IDF'), ('scoring.BM25F()', 'BM25F')]


# dictionary  queries = {query_id : query}

file = open(data_path + 'cran_Queries.tsv', 'r')
reader = csv.reader(file, delimiter = '\t')
queries = {r[0]: r[1] for r in reader}
file.close()

#remove the head of the tsv from the dict
del queries["Query_ID"]

# dictionary SE_dict = {config_name : {query_id :{doc_id :  rank}}}
#the following dictionary is a nested dictionary and it is created in order to map to each configuration the results (documents) for each query
#it also contains the rank for each document

SE_dict = {}

start = time.time()

for elem in analyzer:
    if not os.path.exists(data_path + elem[1]):
        os.mkdir(data_path  + elem[1])

    # the schema must contain the doc id, the title and the body
    schema = Schema(id = ID(stored = True),
                    title = TEXT(analyzer = elem[0], stored = True),
                    body = TEXT(analyzer = elem[0]))

    # creating the index and open it
    ix = index.create_in(data_path + elem[1], schema)
    ix = index.open_dir(data_path + elem[1])
    writer = ix.writer()

    # filling the schema
    for i in range(1, 1401):
        h = open(doc_path + '______' + str(i) + '.html', "r")
        f = h.read()
        soup = BeautifulSoup(f)

        tit = soup.find('title') #get the title from the doc
        tit = ''.join(tit.findAll(text = True))
        tit = tit.replace('\n', ' ').replace(" .", "")

        body = soup.find('body') #get the body from the doc_id
        body = ''.join(body.findAll(text = True))
        body = body.replace('\n  ', '').replace("\n", " ")
        body = ' '.join(word.strip(string.punctuation) for word in body.split())

        # adding to the schema the doc id, the title and the body
        writer.add_document(id = str(i), title = tit, body = body)

    writer.commit()
    h.close()

    for score in scoring_list:
        # create the empty dictionary that we fill up during the loop
        query_doc = {}

        for q in queries:

            input_query = queries[q]

            # search both the title and body fields
            qp = MultifieldParser(["title", "body"], ix.schema)
            # parsing the query
            persed_query = qp.parse(input_query)

            searcher = ix.searcher(weighting = eval(score[0]))

            results = searcher.search(persed_query, limit = 10)

            # create an empty dictionary used for the value(s) of query_doc
            ranking = {}

            for hit in results:
            # save the result of the doc_id and the rank as dictonary
                ranking.update({int(hit['id']) : int(hit.rank) + 1})

            # save the rank with the query id
            query_doc.update({int(q) : ranking})

        SE_dict["SE_" + elem[1] + "_" + score[1]] = query_doc

end = time.time()

#print(round((end - start)/60, 2))

# open the GT as dataframe and map each query with a set of documents
# dictionary gt_dict = {query_id : [relevant_docs]}

gt_dict = pd.read_csv(data_path + 'cran_Ground_Truth.tsv', sep = "\t")
gt_dict = gt_dict.groupby("Query_id")["Relevant_Doc_id"].apply(list).to_dict()


# this is the dictionary containing the relevant_docs with correspondant rank (sorted by rank)
# dictionary {config : {query_id : {doc_id1 : rank1, doc_id2, rank_2, ...}}}

SE_gt_ideal = {}

for config in SE_dict:

    qq = {}

    for query_id in SE_dict[config]:

        rank_gt = {}

        for doc_id in SE_dict[config][query_id]:
            if doc_id in gt_dict[query_id]:

                rank_gt.update({int(doc_id): SE_dict[config][query_id][doc_id]})

                qq.update({int(query_id): rank_gt})

    SE_gt_ideal[config] = qq


# dictionary SE_gt = {config :  {query_id : {doc_id1 : rank1, doc_id2, rank_2, ...}}}
# this dict has the same structure of SE_gt_ideal, but the documents are not sorted by the rank

SE_gt = {}

for config in SE_dict:

    qq = {}

    for query in SE_dict[config]:

        rr = {}

        for doc_id in gt_dict[query]:
            if doc_id in SE_dict[config][query]:

                rr.update({int(doc_id): SE_dict[config][query][doc_id]})

                qq.update({int(query): rr})

    SE_gt[config] = qq


# dictionary rank_dict = {config : [reciprocal_rank_per_query]}

rank_dict = {}

for analyzer in SE_gt_ideal:
    rank_list = []
    for doc in SE_gt_ideal[analyzer].values():
        rank_list.append(1/min(doc.values()))
    rank_dict[analyzer] = rank_list


# dictionary  mrr_dict = {config : mrr_score}

mrr_dict = {}
for key in rank_dict:
    mrr_dict[key] = round(sum(rank_dict[key])/222, 2)




# henceforward we'll use only the search engine configurations with MRR >= 0.32
# dictionary accept_configuration = {accept_config : mrr_score}

accept_configuration = {}

for config in mrr_dict:
    if mrr_dict[config] >= 0.32:
        accept_configuration[config] = mrr_dict[config]


# R-precision dict
# dictionary  precision_dict = {accept_config : {query : R-precision}}

precision_dict = {}

for config in accept_configuration:

    precision_dict[config] = {}

    for query in SE_gt_ideal[config]:
        count = 0
        for rank in SE_gt_ideal[config][query].values():
            if rank <= len(gt_dict[query]):
                count += 1
        precision_dict[config][query] = round(count/len(gt_dict[query]), 2)


# dictionary table_dict = {accept_config : {'mean' : mean_value, 'min' : min_value, 'first' : first_value, 'median' : median_value, 'third' : third_value}}
# it contains the relative descriptive statistics of the R precision for each accept configuration

table_dict = {}

for config in precision_dict:
    table_dict[config] = {}
    mean = round(np.average(list(precision_dict[config].values())), 2)
    minimum = min(precision_dict[config].values())
    first = round(np.quantile(list(precision_dict[config].values()), 0.25), 2)
    median = round(np.median(list(precision_dict[config].values())), 2)
    third = round(np.quantile(list(precision_dict[config].values()), 0.75), 2)
    maximum = max(precision_dict[config].values())

    table_dict[config].update({'mean' : mean, 'min' : minimum, 'first_quartile' : first,  'median' : median, 'third_quartile' : third, 'max' : maximum})

# ndgc_dict = { k : {config_name : avg_ndcg}}
# it contain the average nDCG for k = [1, 10] and for each accept configuration

ndcg_dict = {}

for k in range(1, 11):

    sub_dict = {}

    for config in accept_configuration:

        qq = {}

        for query in SE_dict[config]:

            dcg = 0
            idcg = 0

            if query in SE_gt[config]:
                results = list(SE_dict[config][query])[:k]
                relevance = []

                for i in range(len(results)):

                    if results[i] in gt_dict[query]:
                        relevance.append(int(1))
                    else:
                        relevance.append(int(0))

                for i in range(len(relevance)):

                    if relevance[i] == 1:
                        dcg += 1/np.log2(i+2)
                    else:
                        dcg += 0

                sorted_li = sorted(relevance, reverse = True)

                for i in range(len(sorted_li)):

                    if sorted_li[i] == 1:

                        idcg += 1/np.log2(i+2)

                try:
                    qq.update({int(query): round(dcg/idcg, 2)})

                except ZeroDivisionError:
                    qq.update({int(query): 0})
        sub_dict.update({config: round(np.mean(list(qq.values())), 2)})

    ndcg_dict.update({int(k): fazz_one})


# plot the nDCG@k

#k = list(ndcg_dict.keys())

#markers = ["o", "v", "s", "X", "P", "d", "*", "p", "^", "8", "D", "H", "h"]

#plt.figure(figsize = (10, 8))

#for config, mark in zip(accept_configuration, markers):
#    av = []

#    for key in ndcg_dict:
#        av.append(ndcg_dict[key][config])
#    plt.plot(k, av, marker = mark, markersize = 10, linestyle = "-", label = config)

#plt.xlabel('k')
#plt.ylabel('Average nDCG')
#plt.legend()
