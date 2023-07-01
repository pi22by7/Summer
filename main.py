import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx

nltk.download('punkt')
nltk.download('stopwords')


def extract_text_from_url(url, element, class_name):
    # Send a GET request to the URL
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for failed requests

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all elements with the specified tag and class
    elements = soup.find_all(element, class_=class_name)

    # Extract the text from the found elements
    text = ' '.join([element.get_text() for element in elements])

    return text


def summarize_text(text, num_sentences=3):
    # Preprocessing
    text = text.lower()

    # Sentence Tokenization
    sentences = sent_tokenize(text)

    # Stopword Removal
    stop_words = set(stopwords.words('english'))
    sentences = [sentence for sentence in sentences if sentence not in stop_words]

    # TextRank Algorithm

    # Word Tokenization
    words = [word_tokenize(sentence) for sentence in sentences]

    # Vector Representation (Using GloVe embeddings as an example)
    word_embeddings = {}
    with open('glove.6B.50d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = embedding

    sentence_vectors = []
    for sentence in words:
        if len(sentence) != 0:
            vectors = [word_embeddings.get(word, np.zeros((50,))) for word in sentence]
            vector = sum(vectors) / len(vectors)
        else:
            vector = np.zeros((50,))
        sentence_vectors.append(vector)

    # Similarity Matrix
    similarity_matrix = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 50),
                                                            sentence_vectors[j].reshape(1, 50))[0, 0]

    # Graph Creation
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    # Ranking
    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)

    # Selection
    selected_sentences = [sentence for _, sentence in ranked_sentences[:num_sentences]]

    # Summary Generation
    summary = " ".join(selected_sentences)

    return summary


# Example usage
url = "https://towardsdatascience.com/scraping-1000s-of-news-articles-using-10-simple-steps-d57636a49755"
element = "div"
class_name = "ch"

# Extract text from URL
html_text = extract_text_from_url(url, element, class_name)

# Generate summary
summary = summarize_text(html_text)

# Write raw text to file
with open("raw_text.txt", "w", encoding="utf-8") as file:
    file.write(html_text)

# Write summary to file
with open("summary.txt", "w", encoding="utf-8") as file:
    file.write(summary)

print("Raw text and summary have been written to files.")
