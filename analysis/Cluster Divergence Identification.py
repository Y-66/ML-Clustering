import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import math

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, cohen_kappa_score, confusion_matrix
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec


# ===================== 1. Load Data =====================

df = pd.read_csv('../processed_data.csv')
docs = df['Cleaned_Content'].fillna('').str.replace(r'\bpad\b', '', regex=True).tolist()
true_labels = df['Label'].tolist()

unique_labels = sorted(list(set(true_labels)))
label_to_id = {l: i for i, l in enumerate(unique_labels)}
y_true = np.array([label_to_id[l] for l in true_labels])


# ===================== 2. Train Word2Vec =====================

tokenized_docs = [doc.split() for doc in docs]

print("Training Word2Vec model...")
w2v_model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    epochs=20,
    seed=42
)


# ===================== 3. Document Embeddings =====================

def get_document_vector(doc_tokens, model):
    valid_words = [w for w in doc_tokens if w in model.wv.key_to_index]
    if not valid_words:
        return np.zeros(model.vector_size)
    return np.mean([model.wv[w] for w in valid_words], axis=0)

print("Generating document embeddings...")
X_w2v = np.array([get_document_vector(tokens, w2v_model) for tokens in tokenized_docs])


# ===================== 4. K-means =====================

k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
y_pred_clusters = kmeans.fit_predict(X_w2v)


# ===================== 5. Hungarian Label Matching =====================

def match_labels(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    mapping = {col: row for col, row in zip(col_ind, row_ind)}
    y_pred_matched = np.array([mapping[l] for l in y_pred])
    return y_pred_matched, mapping

y_pred_matched, mapping = match_labels(y_true, y_pred_clusters)


# ===================== 6. Evaluation =====================

sil_score = silhouette_score(X_w2v, y_pred_clusters)
kappa = cohen_kappa_score(y_true, y_pred_matched)

print("\n=== K-means + Word2Vec Evaluation ===")
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Kappa Score: {kappa:.4f}")


# ===================== 7. NPMI Coherence =====================

cluster_top_words = []
for i in range(k):
    centroid = kmeans.cluster_centers_[i]
    top_words = [w for w, _ in w2v_model.wv.similar_by_vector(centroid, topn=10)]
    cluster_top_words.append(top_words)

cv_vectorizer = CountVectorizer(
    vocabulary=set(w for words in cluster_top_words for w in words),
    binary=True
)
X_binary = cv_vectorizer.fit_transform(docs).toarray()
word2id = cv_vectorizer.vocabulary_
N = len(docs)

coherence_scores = []
for top_words in cluster_top_words:
    score = 0
    pairs = list(itertools.combinations(top_words, 2))
    for w1, w2 in pairs:
        if w1 not in word2id or w2 not in word2id:
            continue
        id1, id2 = word2id[w1], word2id[w2]
        p_w1 = X_binary[:, id1].sum() / N
        p_w2 = X_binary[:, id2].sum() / N
        p_w12 = (X_binary[:, id1] * X_binary[:, id2]).sum() / N
        if p_w12 > 0:
            pmi = math.log(p_w12 / (p_w1 * p_w2))
            npmi = pmi / -math.log(p_w12)
            score += npmi
    coherence_scores.append(score / len(pairs) if pairs else 0)

avg_coherence = np.mean(coherence_scores)
print(f"NPMI Coherence: {avg_coherence:.4f}")


# ===================== 8. Confusion Matrix =====================

cm = confusion_matrix(y_true, y_pred_matched)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=unique_labels,
            yticklabels=unique_labels)
plt.title("Confusion Matrix (K-means + Word2Vec)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_kmeans_word2vec.png", dpi=300)
plt.close()


# ===================== 9. Error Analysis =====================

print("\n=== Error Analysis ===")

category_agreement = {}

for i, label in enumerate(unique_labels):
    correct = cm[i, i]
    total = cm[i].sum()
    acc = correct / total if total > 0 else 0
    category_agreement[label] = acc

for k_label, v in category_agreement.items():
    print(f"{k_label}: {v:.3f}")

worst_category = min(category_agreement, key=category_agreement.get)
print("\nWorst clustered category:", worst_category)

worst_idx = label_to_id[worst_category]

misclassified_docs = [
    docs[i]
    for i in range(len(docs))
    if y_true[i] == worst_idx and y_pred_matched[i] != worst_idx
]

print("Misclassified samples:", len(misclassified_docs))

if len(misclassified_docs) > 0:
    vectorizer = CountVectorizer(stop_words='english', max_features=15)
    X_err = vectorizer.fit_transform(misclassified_docs)
    top_words = vectorizer.get_feature_names_out()

    print("Top confusing words:")
    print(top_words)


# ===================== 10. t-SNE Visualization =====================

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_w2v)

plt.figure(figsize=(6,5))
sns.scatterplot(
    x=X_tsne[:,0],
    y=X_tsne[:,1],
    hue=[unique_labels[i] for i in y_true],
    palette='Set1',
    alpha=0.8
)
plt.title("t-SNE (Word2Vec + K-means)")
plt.tight_layout()
plt.savefig("tsne_kmeans_word2vec.png", dpi=300)
plt.close()

print("\n===== Pipeline Finished =====")