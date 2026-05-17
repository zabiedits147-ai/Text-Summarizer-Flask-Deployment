import threading
from nltk.corpus import stopwords
import numpy as np
import networkx as nx
import regex
from flask import Flask, request, jsonify, render_template
import nltk

def read_article(data):
    article = data.split(". ")
    sentences = []
    for sentence in article:
        review = regex.sub("[^A-Za-z0-9]",' ', sentence)
        sentences.append(review.replace("[^a-zA-Z]", " ").split(" "))        
    sentences.pop()     
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1 - nltk.cluster.util.cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

def generate_summary(file_name, top_n=10):
    stop_words = stopwords.words('english')
    summarize_text = []
    sentences = read_article(file_name)
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    a = ". ".join(summarize_text)
    return a

#----------FLASK-----------------------------#

result_store = {}

app = Flask(__name__)

@app.route('/templates', methods=['POST'])
def original_text_form():
    text = request.form['input_text']
    number_of_sent = request.form['num_sentences']

    def run_summary():
        result_store['summary'] = generate_summary(text, int(number_of_sent))

    thread = threading.Thread(target=run_summary)
    thread.start()
    thread.join()

    summary = result_store.get('summary', '')
    return render_template('index1.html', title="Summarizer", original_text=text, output_summary=summary, num_sentences=5)

@app.route('/')
def homepage():
    title = "TEXT summarizer"
    return render_template('index1.html', title=title)

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=7860)