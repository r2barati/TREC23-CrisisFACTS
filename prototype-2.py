from transformers import pipeline
!pip install --upgrade git+https://github.com/allenai/ir_datasets.git@crisisfacts # install ir_datasets (crisisfacts branch)

credentials = {
    "institution": "", # University, Company or Public Agency Name
    "contactname": "", # Your Name
    "email": "", # A contact email address
    "institutiontype": "" # Either 'Research', 'Industry', or 'Public Sector'
}

# Write this to a file so it can be read when needed
import json
import os

home_dir = os.path.expanduser('~')

!mkdir -p ~/.ir_datasets/auth/
with open(home_dir + '/.ir_datasets/auth/crisisfacts.json', 'w') as f:
    json.dump(credentials, f)

# Event numbers as a list
eventNoList = [
    "001", # Lilac Wildfire 2017
    "002", # Cranston Wildfire 2018
    "003", # Holy Wildfire 2018
    "004", # Hurricane Florence 2018
    "005", # 2018 Maryland Flood
    "006", # Saddleridge Wildfire 2019
    "007", # Hurricane Laura 2020
    "008", # Hurricane Sally 2020
    "009", # Beirut Explosion, 2020
    "010", # Houston Explosion, 2020
    "011", # Rutherford TN Floods, 2020
    "012", # TN Derecho, 2020
    "013", # Edenville Dam Fail, 2020
    "014", # Hurricane Dorian, 2019
    "015", # Kincade Wildfire, 2019
    "016", # Easter Tornado Outbreak, 2020
    "017", # Tornado Outbreak, 2020 Apr
    "018", # Tornado Outbreak, 2020 March
]
\import requests

# Gets the list of days for a specified event number, e.g. '001'
def getDaysForEventNo(eventNo):

    # We will download a file containing the day list for an event
    url = "http://trecis.org/CrisisFACTs/CrisisFACTS-"+eventNo+".requests.json"

    # Download the list and parse as JSON
    dayList = requests.get(url).json()

    # Print each day
    # Note each day object contains the following fields
    #   {
    #      "eventID" : "CrisisFACTS-001",
    #      "requestID" : "CrisisFACTS-001-r3",
    #      "dateString" : "2017-12-07",
    #      "startUnixTimestamp" : 1512604800,
    #      "endUnixTimestamp" : 1512691199
    #   }

    return dayList

for day in getDaysForEventNo(eventNoList[0]):
    print(day["dateString"])

eventsMeta = {}

for eventNo in eventNoList: # for each event
    dailyInfo = getDaysForEventNo(eventNo) # get the list of days
    eventsMeta[eventNo]= dailyInfo

    print("Event "+eventNo)
    for day in dailyInfo: # for each day
        print("  crisisfacts/"+eventNo+"/"+day["dateString"], "-->", day["requestID"]) # construct the request string

    print()

import ir_datasets

# download the first day for event 001 (this is a lazy call, it won't download until we first request a document from the stream)
dataset = ir_datasets.load('crisisfacts/001/2017-12-07')

for item in dataset.docs_iter()[:10]: # create an iterator over the stream containing the first 10 items
    print(item)

# download the second day for event 009, first 2023 event
dataset = ir_datasets.load('crisisfacts/009/2020-08-04')

for item in dataset.docs_iter()[:10]: # create an iterator over the stream containing the first 10 items
    print(item)

import pandas as pd

# Convert the stream of items to a Pandas Dataframe
itemsAsDataFrame = pd.DataFrame(dataset.docs_iter())

# Create a filter expression
is_reddit =  itemsAsDataFrame['source_type']=="Reddit"

# Apply our filter
itemsAsDataFrame[is_reddit]

# Create a filter expression
is_twitter =  itemsAsDataFrame['source_type']=="Twitter"

# Apply our filter
itemsAsDataFrame[is_twitter]

# Create a filter expression
is_fb =  itemsAsDataFrame['source_type']=="Facebook"

# Apply our filter
itemsAsDataFrame[is_fb]

# Create a filter expression
is_news =  itemsAsDataFrame['source_type']=="News"

# Apply our filter
itemsAsDataFrame[is_news]


import pandas as pd

pd.DataFrame(dataset.queries_iter())

!pip install python-terrier # install pyTerrier

import pyterrier as pt

# Initalize pyTerrier if not started
if not pt.started():
    pt.init()

# Ask pyTerrier to download the dataset, the 'irds:' header tells pyTerrier to use ir_datasets as the data source
pyTerrierDataset = pt.get_dataset('irds:crisisfacts/009/2020-08-04')

# To create the index, we use an 'indexer', this interates over the documents in the collection and adds them to the index
# The paramters of this call are:
#  Index Storage Path: "None" (some index types write to disk, this would be the directory to write to)
#  Index Type: type=pt.index.IndexingType(3) (Type 3 is a Memory Index)
#  Meta Index Fields: meta=['docno', 'text'] (The index also can store raw fields so they can be attached to the search results, this specifies what fields to store)
#  Meta Index Lengths: meta_lengths=[40, 200] (pyTerrier allocates a fixed amount of storage space per field, how many characters should this be?)
indexer = pt.IterDictIndexer("None", type=pt.index.IndexingType(3), meta=['docno', 'text'], meta_lengths=[40, 200])

# Trigger the indexing process
index = indexer.index(pyTerrierDataset.get_corpus_iter())

retriever = pt.BatchRetrieve(index, wmodel="DFReeKLIM", metadata=["docno", "text"])

pd.DataFrame(retriever.search("injuries"))

# All of the above codes are provided by project at https://colab.research.google.com/github/crisisfacts/utilities/blob/main/00-Data/00-CrisisFACTS.Downloader.ipynb


!pip install gensim

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Prepare the data for Doc2Vec
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(itemsAsDataFrame['text'].str.split())]

# Train a Doc2Vec model
model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)

# Generate embeddings for all documents
document_embeddings = np.array([model.infer_vector(doc.words) for doc in documents])

# Function to calculate cosine similarity between two vectors
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Generate an embedding for a query
query = "injuries"
query_embedding = model.infer_vector(query.split())

# Calculate similarity scores between the query and all documents
similarity_scores = [cosine_similarity(query_embedding, doc_embedding) for doc_embedding in document_embeddings]

# Print the top 10 documents with the highest similarity scores
top_docs = np.argsort(similarity_scores)[::-1][:10]

print(itemsAsDataFrame.iloc[top_docs])


 
# Instantiate a Summarizer model
summarizer_object = pipeline("summarization", model = "facebook/bart-large-cnn")
# Let's say we want to summarize the top 10 documents related to "injuries"
top_docs_text = itemsAsDataFrame.iloc[top_docs]['text']
# Concatenate the text of the top documents into one string
text = " ".join(article_texts)

# Use the model to summarize the text
new_summary = summarizer_object(text, max_length = 120, min_length = 80)

print(new_summary)

import json

def process_request(request):
    # This is a placeholder function.
    return [
        {"doc_id": "doc1", "score": 0.9},
        {"doc_id": "doc2", "score": 0.8},
        # ... more documents ...
    ]

results = []
for eventNo in eventNoList:
    dailyInfo = getDaysForEventNo(eventNo)
    for day in dailyInfo:
        request_id = day["requestID"]
        ranked_docs = process_request(request_id)
        for rank, doc in enumerate(ranked_docs, start=1):
            result = {
                "run_id": "run1",  # placeholder
                "event_id": day["eventID"],
                "request_id": request_id,
                "doc_id": doc["doc_id"],
                "rank": rank,
                "score": doc["score"],
                "run_type": "automatic"  # placeholder
            }
            results.append(result)
            
# Write the results to the output file
with open('submission.json', 'w') as f:
    for result in results:
        json.dump(result, f)
        f.write('\n')  # write a newline character after each JSON object

with open('submission.json', 'r') as f:
    for line in f:
        print(line)
        

import numpy as np

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    return np.mean(r)

def dcg_at_k(r, k):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.
    """
    r = np.asarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    """Score is normalized discounted cumulative gain (ndcg)"""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def expected_reciprocal_rank(r):
    """ERR is the expected reciprocal rank"""
    p = 1.0
    for i in range(len(r)):
        rank = i+1
        R = (2**r[i]-1) / 2**max(r)
        p *= R
        ERR = p / rank
    return ERR

# Assumption for relevance scores for a query
relevance_scores = [3, 2, 3, 0, 0, 1, 2, 3, 2, 0]

# Calculate the metrics:
print("P@10:", precision_at_k(relevance_scores, 10))
print("nDCG@10:", ndcg_at_k(relevance_scores, 10))
print("ERR:", expected_reciprocal_rank(relevance_scores))


 
