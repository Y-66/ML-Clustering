############################################################
# EM (GMM) TEXT CLUSTERING USING TF-IDF (OPTIMIZED VERSION)
############################################################

import matplotlib
matplotlib.use("TkAgg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


def main():

    ########################################################
    # LOAD DATA
    ########################################################

    data = pd.read_csv("processed_data.csv")

    texts = data["Cleaned_Content"].astype(str)
    true_labels = data["Label"]

    tokenized_texts = [doc.split() for doc in texts]

    print("Dataset size:", len(texts))

    ########################################################
    # TF-IDF FEATURE ENGINEERING
    ########################################################

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1,2),
        min_df=5,
        max_df=0.7,
        sublinear_tf=True
    )

    X = tfidf.fit_transform(texts).toarray()

    print("TF-IDF Shape:", X.shape)

    ########################################################
    # SCALING + PCA (IMPORTANT FOR GMM)
    ########################################################

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=100, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)

    print("Reduced Shape:", X_reduced.shape)

    ########################################################
    # EM / GMM
    ########################################################

    n_clusters = len(np.unique(true_labels))

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='diag',   # improved stability
        random_state=42
    )

    pred_clusters = gmm.fit_predict(X_reduced)

    ########################################################
    # EVALUATION
    ########################################################

    sil = silhouette_score(X_reduced, pred_clusters)
    print("Silhouette Score:", sil)

    encoder = LabelEncoder()
    true_encoded = encoder.fit_transform(true_labels)

    kappa = cohen_kappa_score(true_encoded, pred_clusters)
    print("Kappa Score:", kappa)

    ########################################################
    # CONFUSION MATRIX (7.1 Disagreement Analysis)
    ########################################################

    cm = confusion_matrix(true_encoded, pred_clusters)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - GMM (TF-IDF)")
    plt.xlabel("Predicted Cluster")
    plt.ylabel("True Label")
    plt.savefig("Confusion_Matrix_TFIDF.png", dpi=300)
    plt.close()

    ########################################################
    # COHERENCE SCORE
    ########################################################

    dictionary = Dictionary(tokenized_texts)

    cluster_topics = []

    for c in range(n_clusters):

        cluster_docs = texts[pred_clusters == c]

        if len(cluster_docs) > 0:

            vectorizer = CountVectorizer(
                stop_words="english",
                max_features=10
            )

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
    print("Coherence Score:", coherence)

    ########################################################
    # PCA (2D) FOR VISUALIZATION
    ########################################################

    pca_vis = PCA(n_components=2, random_state=42)
    X_vis = pca_vis.fit_transform(X_reduced)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(
        X_vis[:,0],
        X_vis[:,1],
        c=pred_clusters,
        cmap="viridis",
        s=25
    )

    plt.title("GMM Clustering (TF-IDF + PCA)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter)
    plt.savefig("TFIDF_GMM_Clustering.png", dpi=300)
    plt.close()

    ########################################################
    # DENDROGRAM
    ########################################################

    linked = linkage(X_reduced[:200], method='ward')

    plt.figure(figsize=(10,6))
    dendrogram(linked, truncate_mode="level", p=5)
    plt.title("Hierarchical Dendrogram (TF-IDF)")
    plt.savefig("Dendrogram_TFIDF.png", dpi=300)
    plt.close()

    ########################################################
    # ERROR ANALYSIS
    ########################################################

    data["Predicted"] = pred_clusters
    data["True"] = true_encoded

    errors = data[data["Predicted"] != data["True"]]

    print("Misclustered Samples:", len(errors))

    if len(errors) > 0:

        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=10
        )

        X_err = vectorizer.fit_transform(
            errors["Cleaned_Content"]
        )

        print("Top confusing words:")
        print(vectorizer.get_feature_names_out())

    print("\n===== PIPELINE FINISHED SUCCESSFULLY =====")


if __name__ == "__main__":
    main()