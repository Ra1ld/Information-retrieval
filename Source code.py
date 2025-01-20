# -*- coding: utf-8 -*-

## CRAWLER - WIkIPEDIA

import requests
from bs4 import BeautifulSoup
import re

def fetch_wikipedia_page(url):
    """
    Fetches and parses a Wikipedia page given its URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad requests (4xx and 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def extract_links(soup):
    """
    Extracts and returns a list of valid Wikipedia links from a page.
    """
    base_url = "https://en.wikipedia.org"
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        # Filter for Wikipedia article links
        if href.startswith('/wiki/') and not re.search(r':', href):
            full_url = base_url + href
            links.append(full_url)
    return links

def crawl_wikipedia(start_url, max_depth=2, max_pages=10):
    """
    Crawls Wikipedia starting from `start_url` up to `max_depth` levels.
    """
    parsed_paragraph = []
    visited = set()
    to_visit = [(start_url, 0)]  # (URL, depth)
    result = []

    while to_visit and len(result) < max_pages:
        current_url, depth = to_visit.pop(0)
        if depth > max_depth or current_url in visited:
            continue

        print(f"Visiting: {current_url} at depth {depth}")
        soup = fetch_wikipedia_page(current_url)
        
        if soup is None:
            continue

        paragraphs = soup.find_all('p')
        #append in one list
        text = [p.text.strip() for p in paragraphs] 
        parsed_paragraph.append(" ".join(text))

        visited.add(current_url)
        result.append(current_url)

        links = extract_links(soup)
        for link in links:
            if link not in visited:
                to_visit.append((link, depth + 1))
    
    return (result, parsed_paragraph)

#CRAWLER - MAIN CODE
start_url = "https://en.wikipedia.org/wiki/Web_scraping"
(crawled_pages, paragraphs) = crawl_wikipedia(start_url, max_depth=2, max_pages=20)
print("\nCrawled Pages:")
for page in crawled_pages:
    print(page)
    
import pandas as pd

df = pd.DataFrame({
    "URL": crawled_pages,
    "Text": paragraphs
})

# Save to CSV
df.to_csv('wikipedia_data.csv', index=False, encoding='utf-8')



## CREATE INVERTED INDEX

# Specify the path to your CSV file
csv_file_path = "wikipedia_data.csv"

# Load the CSV file into a DataFrame
data = pd.read_csv(csv_file_path, header=0)

# Display the loaded data
data.head()


# Access the column as a list
column_data = data['Text'].tolist() 



from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Downloading important data for nltk to work
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Επεξεργασία Κειμένου
def preprocess_text(text):
    # Αφαίρεση ειδικών χαρακτήρων και αριθμών
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Μετατροπή σε πεζά
    tokens = [word.lower() for word in tokens]

    # Αφαίρεση stop-words
    stop_words = set(stopwords.words("english"))  
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

# Εφαρμογή της συνάρτησης preprocess_text στη στήλη κειμένου
cleaned_text = []
for item in column_data:
    cleaned_text.append(preprocess_text(item))



from collections import defaultdict

def create_inverted_index(documents):
    inverted_index = defaultdict(list)  # Κάθε όρος αντιστοιχεί σε λίστα εγγράφων
    for doc_id, tokens in enumerate(documents):
        
        # Εισαγωγή λέξεων στη δομή
        for word in set(tokens):  # Χρησιμοποιούμε set για να αποφύγουμε επαναλήψεις
            inverted_index[word].append(doc_id)
    return inverted_index



# Δημιουργία του ευρετηρίου από τη στήλη "cleaned_text"
inverted_index = create_inverted_index(cleaned_text)

# Προβολή παραδείγματος από το ευρετήριο
for word, doc_ids in list(inverted_index.items())[:5]:
    print(f"{word}: {doc_ids}")

import json
# Save the inverted index to a JSON file
with open("inverted_index.json", "w", encoding="utf-8") as f:
    json.dump(inverted_index, f, ensure_ascii=False, indent=4)
    

## USE INVERTED INDEX FOR SEARCH
import json

with open("inverted_index.json", "r", encoding="utf-8") as f:
    inverted_index = json.load(f)
    
import pandas as pd

csv_file_path = "wikipedia_data.csv"

# Load the CSV file into a DataFrame
data = pd.read_csv(csv_file_path, header=0)    
#it has the urls of the docs
urls = data['URL'].tolist() 
documents = data['Text'].tolist()



# Προεπεξεργασία ερωτήματος
def preprocess_query(query):
    stop_words = set(stopwords.words("english")) 
    #it should not remove AND OR NOT
    stop_words = [word for word in stop_words if word not in ['and', 'or', 'not']]
    tokens = word_tokenize(query)
    tokens = [re.sub(r"[^a-zA-Z]", "", token.lower()) for token in tokens]
    tokens = [word for word in tokens if word and word not in stop_words]
    return tokens

# Boolean λειτουργίες
def boolean_and(postings1, postings2):
    return list(set(postings1) & set(postings2))

def boolean_or(postings1, postings2):
    return list(set(postings1) | set(postings2))

def boolean_not(postings, total_docs):
    return list(set(total_docs) - set(postings))

# Αναζήτηση - Boolean Retrieval
# Αναζήτηση - Boolean Retrieval
def search(query, inverted_index, total_docs):
    # Χωρισμός σε λέξεις και προεπεξεργασία
    query_tokens = preprocess_query(query)
    
    # Ανάλυση ερωτήματος για Boolean λειτουργίες
    results = set(range(total_docs))  # Όλα τα έγγραφα αρχικά
    operation = "AND"  # Default λειτουργία
    
    for token in query_tokens:
        if token.upper() in ["AND", "OR", "NOT"]:
            operation = token.upper()
        else:
            postings = inverted_index.get(token, [])
            
            if operation == "AND":
                results = boolean_and(results, postings)
            elif operation == "OR":
                results = boolean_or(results, postings)
            elif operation == "NOT":
                results = boolean_not(postings, range(total_docs))
    
    return sorted(results)


def compute_tf(doc, word):
    words = doc.split()
    word_count = words.count(word)
    total_words = len(words)
    tf = word_count / total_words
    return tf

import math


# Υπολογισμός της αντίστροφης συχνότητας εγγράφου (IDF)
def compute_idf(inverted_index, word, total_docs):
    doc_freq = len(inverted_index.get(word, []))
    if doc_freq == 0:
        return 0
    idf = math.log(total_docs / doc_freq)
    return idf

# Υπολογισμός TF-IDF για μια συγκεκριμένη λέξη σε ένα συγκεκριμένο έγγραφο
def compute_tfidf(doc_id, word, documents, inverted_index):
    total_docs = len(documents)
    
    # Υπολογισμός TF για το έγγραφο
    tf = compute_tf(documents[doc_id], word)
    
    # Υπολογισμός IDF για τη λέξη
    idf = compute_idf(inverted_index, word, total_docs)
    
    # Υπολογισμός TF-IDF
    tfidf = tf * idf
    return tfidf

import numpy as np
    
def search_tf_idf(query, inverted_index, documents):
    total_docs = len(documents)
    result_docs = search(query, inverted_index, total_docs)

    query_tokens = preprocess_query(query)
    q_scores = []
    for q in query_tokens:
        if q not in ['and', 'or', 'not']:
            scores = []
            for i in result_docs:
                scores.append(compute_tfidf(i,q,documents,inverted_index))
            q_scores.append(scores)
    
    q_scores = np.array(q_scores)        
    final_scores = np.mean(q_scores,axis=0)
    
    
    sorted_terms, sorted_scores = zip(*sorted(zip(result_docs, final_scores), key=lambda x: x[1], reverse=True))
    
    # Convert back to lists 
    sorted_terms = list(sorted_terms)
    sorted_scores = list(sorted_scores)
    return(sorted_terms, sorted_scores)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def vector_space_retrieve(query, inverted_index, documents):
    total_docs = len(documents)
    result_docs = search(query, inverted_index, total_docs)
    
    
    # Calculate the cosine similarity between the query and the documents
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    
    
    # Get the TF-IDF vector of the query (it will be a sparse matrix)
    query_tfidf = query_vector.toarray()

    # Compute cosine similarity between the query and all documents in the result set
    document_vectors = tfidf_matrix[result_docs]
    
    # Compute cosine similarity of query with each document in the result set
    similarities = cosine_similarity(query_tfidf, document_vectors)
    
    
    sorted_terms, sorted_scores = zip(*sorted(zip(result_docs, list(similarities[0])), key=lambda x: x[1], reverse=True))
    
    # Convert back to lists 
    sorted_terms = list(sorted_terms)
    sorted_scores = list(sorted_scores)
    
    return(sorted_terms, sorted_scores)

from collections import Counter

def build_inverted_index_freq(documents):
        inverted_index = {}
        for doc_id, doc in enumerate(documents):
            terms = doc.split()
            term_freq = Counter(terms)
            for term, freq in term_freq.items():
                if term not in inverted_index:
                    inverted_index[term] = []
                inverted_index[term].append((doc_id, freq))
        return inverted_index
    
def calculate_bm25(query, inverted_index, documents):
    #params
    k1=1.5
    b=0.75
    
    doc_lengths = [len(doc.split()) for doc in documents]
    avg_doc_length = sum(doc_lengths) / len(documents)
    
    #calc IDF
    idf = {}
    total_docs = len(documents)
    for term in inverted_index:
        doc_freq = len(inverted_index[term])
        idf[term] = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
    
    total_docs = len(documents)
    result_docs = search(query, inverted_index, total_docs)
    
    scores = [0] * len(result_docs)  # Score for each document
    query_terms = preprocess_query(query)
    for term in query_terms:
        if term not in ['and', 'or', 'not']:
            if term in inverted_index:
                idf_t = idf.get(term, 0)
                for i, doc_id in enumerate(result_docs):
                    word_list = documents[doc_id].split()
                    term_freq = word_list.count(term)
                    doc_length = doc_lengths[doc_id]
                    tf = term_freq
                    score = idf_t * tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_length / avg_doc_length))
                    scores[i] += score
                    
    sorted_terms, sorted_scores = zip(*sorted(zip(result_docs, scores), key=lambda x: x[1], reverse=True))
    
    # Convert back to lists 
    sorted_terms = list(sorted_terms)
    sorted_scores = list(sorted_scores)
    
    return(sorted_terms, sorted_scores)                




query = "crawler AND bot"
total_docs = data.shape[0]  # Συνολικός αριθμός εγγράφων

# Αναζήτηση Boolean Retrieval
result_docs = search(query, inverted_index, total_docs)
print("Αποτελέσματα Boolean Retrieval:")
for i in result_docs:
    print(urls[i])

# Αναζήτηση TF-IDF
sorted_terms, sorted_scores = search_tf_idf(query, inverted_index, documents)

print("Αποτελέσματα TF-IDF:")
for i in range(len(sorted_terms)):
    print("%s \t %.6f" %( urls[sorted_terms[i]], sorted_scores[i]))

# Αναζήτηση vector space retrieve
sorted_terms, sorted_scores = vector_space_retrieve(query, inverted_index, documents)

print("Αποτελέσματα vector space retrieve:")
for i in range(len(sorted_terms)):
    print("%s  \t %.6f" %( urls[sorted_terms[i]], sorted_scores[i]))


# Αναζήτηση bm25
sorted_terms, sorted_scores = calculate_bm25(query, inverted_index, documents)

print("Αποτελέσματα BM25:")
for i in range(len(sorted_terms)):
    print("%s  \t %.6f" %( urls[sorted_terms[i]], sorted_scores[i]))
    

# EVALUATION - USE NEWSGROUP DATA

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

def scores(retrieved_results, relevant_results, M):
    # Δημιουργία των δυαδικών λιστών (true/false) για τα αποτελέσματα
    # Αν το έγγραφο είναι σχετικό, το σημειώνουμε ως 1, αλλιώς ως 0
    y_true = [1 if doc in relevant_results else 0 for doc in range(M)]  # Σχετικά έγγραφα
    y_pred = [1 if doc in retrieved_results else 0 for doc in range(M)]  # Επιστρεφόμενα έγγραφα

    # Υπολογισμός των μετρικών
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    map_score = average_precision_score(y_true, y_pred)

    # Εκτύπωση αποτελεσμάτων
    print(f"Ακρίβεια (Precision): {precision:.4f}")
    print(f"Ανάκληση (Recall): {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Μέση Ακρίβεια (MAP): {map_score:.4f}")
    
    
# Load the dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = newsgroups.data
labels = newsgroups.target
categories = newsgroups.target_names



import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

query = "space shuttle launch"
total_docs = len(documents)  # Συνολικός αριθμός εγγράφων
relevant_results = np.where(labels == categories.index('sci.space'))
relevant_results = list(relevant_results[0])    


top_k = 10
# Αναζήτηση Boolean Retrieval
sorted_terms = search(query, inverted_index, total_docs)
print("Αποτελέσματα Boolean Retrieval:")
for i in range(top_k):
    print(sorted_terms[i])

retrieved_results = list(sorted_terms[:top_k])
scores(retrieved_results, relevant_results, total_docs)

# Αναζήτηση TF-IDF
sorted_terms, sorted_scores = search_tf_idf(query, inverted_index, documents)
top_k = 10
print("Αποτελέσματα TF-IDF:")
for i in range(top_k):
    print("%s %.6f" %( sorted_terms[i], sorted_scores[i]))

retrieved_results = sorted_terms[:top_k]
scores(retrieved_results, relevant_results, total_docs)

# Αναζήτηση vector space retrieve
sorted_terms, sorted_scores = vector_space_retrieve(query, inverted_index, documents)

print("Αποτελέσματα vector space retrieve:")
for i in range(top_k):
    print("%s %.6f" %( sorted_terms[i], sorted_scores[i]))

retrieved_results = sorted_terms[:top_k]
scores(retrieved_results, relevant_results, total_docs)

# Αναζήτηση bm25
sorted_terms, sorted_scores = calculate_bm25(query, inverted_index, documents)

print("Αποτελέσματα BM25:")
for i in range(top_k):
    print("%s %.6f" %( sorted_terms[i], sorted_scores[i]))

retrieved_results = sorted_terms[:top_k]
scores(retrieved_results, relevant_results, total_docs)