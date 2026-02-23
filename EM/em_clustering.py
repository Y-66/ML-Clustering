# =====================================================
# EM TEXT CLUSTERING PIPELINE
# Gaussian Mixture Model (Expectation-Maximization)
# =====================================================
import matplotlib
matplotlib.use('Agg') # 强制使用非交互式后端，直接保存图片而不弹出窗口
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import cohen_kappa_score, silhouette_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer


# =====================================================
# 1. LOAD DATA
# =====================================================

# ⚠️ Put processed_data.csv in SAME folder as this file
DATA_PATH = "processed_data.csv"

df = pd.read_csv(DATA_PATH)

texts = df["Cleaned_Content"].astype(str)
labels = df["Label"]

print("Dataset size:", len(df))


# =====================================================
# 2. FEATURE ENGINEERING — TF-IDF
# =====================================================

print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=3000,
    stop_words="english",
    ngram_range=(1, 2),   # unigram + bigram
    min_df=3,
    max_df=0.8
)

X_tfidf = tfidf.fit_transform(texts)

print("TFIDF Shape:", X_tfidf.shape)


# =====================================================
# 3. DIMENSION REDUCTION (IMPORTANT FOR EM)
# =====================================================

print("\nApplying SVD dimensionality reduction...")

svd = TruncatedSVD(
    n_components=100,
    random_state=42
)

X_reduced = svd.fit_transform(X_tfidf)

print("Reduced Shape:", X_reduced.shape)


# =====================================================
# 4. EM CLUSTERING (Gaussian Mixture Model)
# =====================================================

print("\nRunning EM Clustering...")

n_clusters = len(np.unique(labels))

gmm = GaussianMixture(
    n_components=n_clusters,
    covariance_type="full",
    random_state=42,
    n_init=10
)

gmm.fit(X_reduced)

clusters = gmm.predict(X_reduced)
probabilities = gmm.predict_proba(X_reduced)

df["Cluster"] = clusters


# =====================================================
# 5. EVALUATION
# =====================================================

print("\n===== Evaluation =====")

le = LabelEncoder()
true_labels = le.fit_transform(labels)

# ---- Cohen Kappa ----
kappa = cohen_kappa_score(true_labels, clusters)
print("Cohen Kappa Score:", round(kappa, 4))

# ---- Silhouette ----
sil = silhouette_score(X_reduced, clusters)
print("Silhouette Score:", round(sil, 4))

print("\nCluster Distribution:")
print(pd.Series(clusters).value_counts())
# =====================================================



# =====================================================
# 7. ERROR ANALYSIS
# =====================================================

print("\n===== Error Analysis =====")

df["True_Label"] = true_labels

errors = df[df["True_Label"] != df["Cluster"]]

print("Misclustered samples:", len(errors))

# ---- EM confidence ----
confidence = probabilities.max(axis=1)
df["Confidence"] = confidence

uncertain_docs = df[df["Confidence"] < 0.4]

print("Low-confidence documents:", len(uncertain_docs))


# ---- Frequent confusing words ----
print("\nTop words causing confusion:")

vec = CountVectorizer(
    stop_words="english",
    max_features=1000
)

X_err = vec.fit_transform(errors["Cleaned_Content"])

word_freq = np.asarray(X_err.sum(axis=0)).flatten()
words = vec.get_feature_names_out()

top_words = sorted(
    zip(words, word_freq),
    key=lambda x: x[1],
    reverse=True
)[:10]

for w, f in top_words:
    print(w, f)


# =====================================================
# 8. SAVE RESULTS
# =====================================================

df.to_csv("EM_results.csv", index=False)

print("\n✅ Pipeline Finished Successfully!")
print("Generated files:")
print(" - EM_clusters.png")
print(" - True_labels.png")
print(" - EM_results.csv")