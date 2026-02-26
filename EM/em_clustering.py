############################################################
# EM (GMM) TEXT CLUSTERING USING WORD2VEC (FULL VERSION)
############################################################

import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, cohen_kappa_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram

from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from matplotlib.patches import Ellipse


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

    print("Dataset size:", len(texts))

    ########################################################
    # WORD2VEC FEATURE ENGINEERING
    ########################################################

    print("\nTraining Word2Vec...")

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
    # EM / GMM
    ########################################################

    n_clusters = len(np.unique(true_labels))

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full',
        random_state=42
    )

    pred_clusters = gmm.fit_predict(X)

    ########################################################
    # EVALUATION
    ########################################################

    sil = silhouette_score(X, pred_clusters)
    print("Silhouette:", sil)

    encoder = LabelEncoder()
    true_encoded = encoder.fit_transform(true_labels)

    kappa = cohen_kappa_score(true_encoded, pred_clusters)
    print("Kappa:", kappa)

    ########################################################
    # CONFUSION MATRIX
    ########################################################

    cm = confusion_matrix(true_encoded, pred_clusters)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Word2Vec GMM")
    plt.xlabel("Predicted Cluster")
    plt.ylabel("True Label")
    plt.savefig("Confusion_Matrix_Word2Vec.png", dpi=300)
    plt.close()

    ########################################################
    # COHERENCE
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
    print("Coherence:", coherence)

    ########################################################
    # PCA FOR VISUALIZATION
    ########################################################

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8,6))
    plt.scatter(
        X_2d[:,0],
        X_2d[:,1],
        c=pred_clusters,
        cmap="viridis",
        s=25
    )

    plt.title("GMM Clustering (Word2Vec)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    ########################################################
    # GAUSSIAN ELLIPSES
    ########################################################

    def draw_ellipse(position, covariance):

        if covariance.shape == (2,2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1,0], U[0,0]))
            width, height = 2*np.sqrt(s)
        else:
            angle = 0
            width, height = 2*np.sqrt(covariance)

        for nsig in range(1,4):
            plt.gca().add_patch(
                Ellipse(
                    position,
                    nsig*width,
                    nsig*height,
                    angle=angle,
                    fill=False,
                    linewidth=2
                )
            )

    for i in range(gmm.n_components):

        mean_2d = pca.transform(
            gmm.means_[i].reshape(1,-1)
        )[0]

        cov_2d = (
            pca.components_
            @ gmm.covariances_[i]
            @ pca.components_.T
        )

        draw_ellipse(mean_2d, cov_2d)

    plt.savefig("Word2Vec_GMM_Clustering.png", dpi=300)
    plt.close()

    ########################################################
    # DENDROGRAM
    ########################################################

    linked = linkage(X[:200], method='ward')

    plt.figure(figsize=(10,6))
    dendrogram(linked, truncate_mode="level", p=5)
    plt.title("Hierarchical Dendrogram (Word2Vec)")
    plt.savefig("Dendrogram_Word2Vec.png", dpi=300)
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

    print("\n===== WORD2VEC PIPELINE FINISHED =====")


############################################################
# WINDOWS SAFE ENTRY
############################################################

if __name__ == "__main__":
    main()
############################################################
# ERROR ANALYSIS FUNCTION
############################################################

from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

def error_analysis(model_name, data, true_encoded, pred_clusters, texts, encoder):

    print(f"\n===== ERROR ANALYSIS: {model_name} =====")

    # Confusion Matrix
    cm = confusion_matrix(true_encoded, pred_clusters)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Cluster")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"Confusion_Matrix_{model_name}.png", dpi=300)
    plt.close()

    # Category agreement
    label_names = encoder.classes_
    category_agreement = {}

    for i, label in enumerate(label_names):
        correct = cm[i, i]
        total = cm[i].sum()
        acc = correct / total if total > 0 else 0
        category_agreement[label] = acc

    print("\nCategory-wise agreement:")
    for k, v in category_agreement.items():
        print(k, round(v, 3))

    # Worst category
    worst_category = min(category_agreement,
                         key=category_agreement.get)

    print("Worst clustered category:", worst_category)

    worst_index = list(label_names).index(worst_category)

    misclassified = data[
        (true_encoded == worst_index) &
        (pred_clusters != worst_index)
    ]

    print("Misclassified samples:",
          len(misclassified))

    # Top confusing words
    if len(misclassified) > 0:

        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=15
        )

        X_err = vectorizer.fit_transform(
            misclassified["Cleaned_Content"]
        )

        top_words = vectorizer.get_feature_names_out()

        print("Top confusing words:")
        print(top_words)

        # Save to txt file
        with open(f"Error_{model_name}_TopWords.txt", "w") as f:
            f.write("Worst Category: " + worst_category + "\n")
            f.write("Misclassified Samples: " +
                    str(len(misclassified)) + "\n\n")
            f.write("Top Confusing Words:\n")
            for w in top_words:
                f.write(w + "\n")