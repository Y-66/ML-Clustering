############################################################
# IMPORTS
############################################################
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram

from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


############################################################
# MAIN PIPELINE
############################################################

def main():

    ########################################################
    # LOAD DATA
    ########################################################

    data = pd.read_csv("processed_data.csv")

    texts = data["Cleaned_Content"].astype(str)
    true_labels = data["Label"]

    tokenized_texts = [doc.split() for doc in texts]

    ########################################################
    # WORD2VEC
    ########################################################

    print("Training Word2Vec...")

    w2v_model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    )

    def document_vector(tokens):
        vectors = [
            w2v_model.wv[word]
            for word in tokens
            if word in w2v_model.wv
        ]
        if len(vectors) == 0:
            return np.zeros(100)

        return np.mean(vectors, axis=0)

    X = np.array([document_vector(doc) for doc in tokenized_texts])

    ########################################################
    # EM CLUSTERING
    ########################################################

    n_clusters = len(np.unique(true_labels))

    gmm = GaussianMixture(
        n_components=n_clusters,
        random_state=42
    )

    pred_clusters = gmm.fit_predict(X)

    ########################################################
    # SILHOUETTE
    ########################################################

    sil = silhouette_score(X, pred_clusters)
    print("Silhouette:", sil)

    ########################################################
    # KAPPA
    ########################################################

    encoder = LabelEncoder()
    true_encoded = encoder.fit_transform(true_labels)

    kappa = cohen_kappa_score(true_encoded, pred_clusters)
    print("Kappa:", kappa)

    ########################################################
    # COHERENCE  ✅ FIXED
    ########################################################

    dictionary = Dictionary(tokenized_texts)

    cluster_topics = []

    for c in range(n_clusters):

        cluster_docs = [
            texts.iloc[i]
            for i in range(len(pred_clusters))
            if pred_clusters[i] == c
        ]

        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=10
        )

        if len(cluster_docs) > 0:
            X_counts = vectorizer.fit_transform(cluster_docs)
            words = vectorizer.get_feature_names_out()
            cluster_topics.append(list(words))

    coherence_model = CoherenceModel(
        topics=cluster_topics,
        texts=tokenized_texts,
        dictionary=dictionary,
        coherence='c_v'
    )

    coherence = coherence_model.get_coherence()
    print("Coherence:", coherence)

    ########################################################
    # PCA VISUALIZATION
    ########################################################

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=X_2d[:,0],
        y=X_2d[:,1],
        hue=pred_clusters
    )
    plt.title("EM Clustering")
    plt.show()

    ########################################################
    # DENDROGRAM
    ########################################################

    linked = linkage(X[:200], method='ward')

    plt.figure(figsize=(10,6))
    dendrogram(linked, truncate_mode='level', p=5)
    plt.title("Dendrogram")
    plt.show()


############################################################
# WINDOWS SAFE ENTRY POINT ⭐⭐⭐⭐⭐
############################################################

if __name__ == "__main__":
    main()